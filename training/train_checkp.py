import os
import time

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import numpy as np
import random

from training.dataset import CollatorForCLM, ParquetDataset
from training.model import Transformer, TransformerModelArgs
from training.utils import build_lr_scheduler, clip_grad_norm_, get_args, get_num_params, get_num_flop_per_token, init_logger, logger, PRECISION_STR_TO_DTYPE, set_default_dtype

def set_seed(seed: int):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  
def get_rng_state_dict():
  return {
    "python": random.getstate(),
    "numpy": np.random.get_state(),
    "torch": torch.get_rng_state(),
    "cuda": torch.cuda.get_rng_state_all(),
  }
  
def set_rng_state_dict(state: dict):
  random.setstate(state["python"])
  np.random.set_state(state["numpy"])
  torch.set_rng_state(state["torch"])
  torch.cuda.set_rng_state_all(state["cuda"])

def save_checkpoint(state: dict, ckpt_dir: str, step: int):
  os.makedirs(ckpt_dir, exist_ok=True)
  path = os.path.join(ckpt_dir, f"checkpoint.pt")
  with open(path, 'wb') as f:
    torch.save(state, f)
    f.flush()
    os.fsync(f.fileno())

def train(args):
  logger.info(f"FAv3?: {os.getenv('USE_FLASH_ATTENTION', '0')} | Validation?: {os.environ.get('VALIDATION_MODE', '0')} | Experiment args: {args}")
  # Init
  device = torch.device(f"cuda:{int(os.getenv('LOCAL_RANK', 0))}")
  model_dtype = PRECISION_STR_TO_DTYPE[args.model_dtype]
  
  if args.seed is not None:
    set_seed(args.seed)

  # Set up DataLoader
  logger.info("Setting up DataLoaders...")
  tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)
  train_ds = ParquetDataset(args.dataset, tokenizer, args.sequence_length, args.batch_size*args.training_steps)
  train_collator = CollatorForCLM(args.sequence_length, tokenizer.pad_token_id)
  train_dl = DataLoader(train_ds,
                        batch_size=args.batch_size,
                        collate_fn=train_collator)
  train_dl_iterator = iter(train_dl)

  # Set up Model
  logger.info("Setting up Model...")
  model_config = TransformerModelArgs(
        dim=args.sequence_length,
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        multiple_of=1024,
        rope_theta=500000,
        vocab_size=tokenizer.vocab_size,
        seq_len=args.sequence_length,
    )
  with set_default_dtype(model_dtype):
    # CPU
    model = Transformer(model_config)
    
  # check if file exists
  if args.load_checkpoint is not None and not os.path.exists(args.load_checkpoint):
    raise ValueError(f"Checkpoint {args.load_checkpoint} does not exist")
  
  # Optional: Resume from checkpoint
  if args.load_checkpoint:
    checkpoint_name = os.path.join(args.checkpoint_dir, "checkpoint.pt")
    logger.info(f"Loading checkpoint Model from {checkpoint_name}")
    # Note: this gives OOM error because during copy, you have 2 copies of the model in GPU memory.
    # Instead, load the model to CPU first and then move to GPU after loading
    tic = time.perf_counter()
    ckpt = torch.load(checkpoint_name, map_location="cpu", weights_only=False) # weights_only=False allows to load custom numpy pickles (UNSAFE)
    model.load_state_dict(ckpt["model"])
    toc = time.perf_counter()
    elapsed = toc - tic
    logger.info(f"Loaded model")
    
  model = model.to(device)
  
  if args.compile:
    logger.info("Using `torch.compile`")
    model = torch.compile(model, fullgraph=True)
  
  model.train()

  # Build Optimizers & LR Scheduler
  optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, fused=args.fused_optimizer)
  lr_scheduler = build_lr_scheduler(optimizer, args.lr_warmup_steps)

  # Utils
  num_flop_per_token = get_num_flop_per_token(
      get_num_params(model, exclude_embedding=True),
      model_config,
  )

  ntokens_since_last_log = 0
  ntraining_tokens_since_last_log = 0
  time_last_log = time.perf_counter()

  train_step = 0
  
  steps = []
  losses = []
  with open(args.loss_file, "w") as f:
    f.write("step,loss\n")
  
  # Optional: Resume from checkpoint
  if args.load_checkpoint:
    logger.info(f"Loading checkpoint optimiser, scheduler")
    tic = time.perf_counter()
    optimizer.load_state_dict(ckpt["optimizer"])
    lr_scheduler.load_state_dict(ckpt["scheduler"])
    set_rng_state_dict(ckpt["rng_state"])
    train_step = ckpt.get("step", 0)
    toc = time.perf_counter()
    elapsed = elapsed + toc - tic
    logger.info(f"Loaded optimiser, scheduler checkpoint at step {train_step}")
    logger.info(f"Checkpoint loaded in: {elapsed:.2f} seconds")
    
    train_dl_iterator = iter(train_dl) # Recreate iterator
    for _ in range(train_step):
      next(train_dl_iterator) # Skip to the current step

  # Warmup
  for _ in range(args.warmup):
    save_checkpoint(
        {
          "step": train_step,
          "model": model.state_dict(),
          "optimizer": optimizer.state_dict(),
          "scheduler": lr_scheduler.state_dict(),
          "rng_state": get_rng_state_dict(), # Loss is stateless, (crossentropy) so no need to save
        },
        args.checkpoint_dir,
        train_step,
      )

  logger.info("Starting training!")
  after_step_time = 0.0
  start_time = time.perf_counter()
  
  while train_step < args.training_steps:
    train_step_time = time.perf_counter()
    train_step += 1

    # Profiling
    if args.profile and args.profile_step_start == train_step:
      torch.cuda.cudart().cudaProfilerStart()
      torch.autograd.profiler.emit_nvtx(record_shapes=True).__enter__()

    input_ids, labels = next(train_dl_iterator)
    ntokens_since_last_log += args.batch_size * args.sequence_length
    num_items_in_batch = labels.ne(-100).sum()
    ntraining_tokens_since_last_log += num_items_in_batch
    input_ids = input_ids.to(device)
    labels = labels.to(device)

    optimizer.zero_grad()

    tic = time.perf_counter()
    logits = model(input_ids)
    fwd_time = time.perf_counter() - tic

    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1).float(), labels.flatten(0, 1), reduction="sum")
    loss = loss / num_items_in_batch
    del logits
    tic = time.perf_counter()
    loss.backward()
    bck_time = time.perf_counter() - tic
    
    # insert step and loss
    steps.append(train_step)
    losses.append(loss.item())

    # Clip gradients
    clip_grad_norm_(model.parameters(), args.grad_max_norm)

    optimizer.step()
    lr_scheduler.step()
    train_step_time = time.perf_counter() - train_step_time

    # Logging
    if (train_step == 1 or train_step % args.logging_frequency == 0):
      time_delta = time.perf_counter() - time_last_log
      # tokens per second per device, abbreviated as tps
      tps = ntokens_since_last_log / time_delta 
      mfu = 100 * num_flop_per_token * tps / 989e12
      tflops = num_flop_per_token * tps / 1e12
      training_tps = ntraining_tokens_since_last_log / time_delta

      logger.info(f"Step: {train_step} | "
                  f"Loss: {loss.item():.2f} | "
                  f"Tokens per second: {tps:.2f} | "
                  f"Training tokens per second (%): {100*training_tps/tps:.2f} | "
                  f"MFU (%): {mfu:.2f} | "
                  f"TFLOPs: {tflops:.7f} | "
                  f"Fwd time: {fwd_time:.7f} | "
                  f"Bck time: {bck_time:.7f} | "
                  f"Train Time: {train_step_time:.7f} | "
                  f"Previous After Step Time: {after_step_time:.7f}")
      ntokens_since_last_log = 0
      ntraining_tokens_since_last_log = 0
      time_last_log = time.perf_counter()
    
    after_step_time = time.perf_counter()
    # Profiling
    if args.profile and args.profile_step_end == train_step:
      torch.cuda.cudart().cudaProfilerStop()
      
    # Checkpointing
    if train_step % args.checkpoint_freq == 0 or train_step == args.training_steps:
      tic = time.perf_counter()
      save_checkpoint(
        {
          "step": train_step,
          "model": model.state_dict(),
          "optimizer": optimizer.state_dict(),
          "scheduler": lr_scheduler.state_dict(),
          "rng_state": get_rng_state_dict(), # Loss is stateless, (crossentropy) so no need to save
        },
        args.checkpoint_dir,
        train_step,
      )
      toc = time.perf_counter()
      logger.info(f"Checkpoint saved in {toc - tic:.4f} seconds")
      
      # save loss trace, the whole array
      with open(args.loss_file, "a") as f:
        for i in range(len(losses)):
          f.write(f"{steps[i]},{losses[i]}\n")
          
      losses = []
      steps = []
      
      logger.info(f"Checkpoint saved to {args.checkpoint_dir}")
    after_step_time = time.perf_counter() - after_step_time
    logger.info(f"Checkpoint saved in {after_step_time:.7f} seconds")

  logger.info(f"Training completed. Final after step time: {after_step_time:.7f} s")
  total_time = time.perf_counter() - start_time
  logger.info(f"Training run end: total time taken {total_time:.2f} seconds")

if __name__ == "__main__":
  init_logger()
  args = get_args()
  train(args)
