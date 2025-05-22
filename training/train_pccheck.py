import os
import time

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import torch.multiprocessing as mp
from training.dataset import CollatorForCLM, ParquetDataset
from training.model import Transformer, TransformerModelArgs
from training.utils import build_lr_scheduler, clip_grad_norm_, get_args, get_num_params, get_num_flop_per_token, init_logger, logger, PRECISION_STR_TO_DTYPE, set_default_dtype, set_seed, save_random_states, load_random_states, load_checkpoint, set_opt_state, set_model_state

from checkpointing.pccheck.chk_monitor import Chk_monitor
from checkpointing.pccheck_utils import initialize, get_total_size, set_storage

def train(args):
  logger.info(f"Experiment args: {args}")
  # Init
  device = torch.device(f"cuda:{int(os.getenv('LOCAL_RANK', 0))}")
  model_dtype = PRECISION_STR_TO_DTYPE[args.model_dtype]
  model_isfp32 = model_dtype == torch.float32
  non_blocking = args.non_blocking
  
  if args.seed is not None:
    set_seed(args.seed)

  # check if file exists
  if args.load_checkpoint is not None and not os.path.exists(args.checkpoint_dir):
    raise ValueError(f"Checkpoint dir {args.checkpoint_dir} does not exist")
  
  train_step = 0
  if args.load_checkpoint:
    # Load random state checkpoint
    state = load_random_states(args.checkpoint_dir)
    train_step = state["step"]
    logger.info(f"Training will resume from step: {train_step}")

  # Set up DataLoader
  logger.info("Setting up DataLoaders...")
  tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)
  train_ds = ParquetDataset(args.dataset, tokenizer, args.sequence_length, args.batch_size*args.training_steps)
  train_collator = CollatorForCLM(args.sequence_length, tokenizer.pad_token_id)
  train_dl = DataLoader(train_ds, batch_size=args.batch_size, collate_fn=train_collator)
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
    model = Transformer(model_config).to(device)
    model.train()

  # Build Optimizers & LR Scheduler
  optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, fused=args.fused_optimizer)
  lr_scheduler = build_lr_scheduler(optimizer, args.lr_warmup_steps)
  # for checkpoint
  mp.set_start_method("spawn", force=True)
  gpu_ar, model_size, opt_size = initialize(model, [optimizer])
  total_size = model_size + opt_size
  print(f"Model size: {model_size}, Optimizer size: {opt_size}, Total size: {total_size}")

  # Optional: Resume from checkpoint
  if args.load_checkpoint:
    logger.info(f"Loading checkpoint model, optimizer + scheduler")
    tic = time.perf_counter()
    lr_scheduler.load_state_dict(state["lr_scheduler"])
    load_checkpoint(ckpt_dir=args.checkpoint_dir, model=model, optimizer_list=[optimizer], max_async=args.max_async, total_size=model_size+opt_size, model_dtype=model_dtype, non_blocking=non_blocking)
    checkpoint_time = time.perf_counter() - tic
    if model_isfp32:
      set_storage(model, [optimizer], gpu_ar)
    torch.cuda.empty_cache()
    logger.info(f"Loaded optimizer, scheduler checkpoint at step {train_step} in {checkpoint_time:.2f} seconds")

    train_dl_iterator = iter(train_dl) # Recreate iterator
    for _ in range(train_step):
      next(train_dl_iterator) # Skip to the current step
  else:
    if model_isfp32:
      set_storage(model, [optimizer], gpu_ar)
    torch.cuda.empty_cache()
  
  ckpt_monitor = Chk_monitor(
      total_size=total_size,
      gpu_ar=gpu_ar,
      bsize=total_size,
      max_async=args.max_async,
      model=model.state_dict(),
      optimizer=optimizer.state_dict(),
  )

  # set_opt_state(gpu_ar, [optimizer], model_size, non_blocking=non_blocking)
  # if not model_isfp32:
  #   set_model_state(gpu_ar, model, non_blocking=non_blocking)
  # logger.info("Checkpointing warmup...")
  # for _ in range(args.warmup):
  #   ckpt_monitor.save()
  #   while ckpt_monitor.checkpoint_in_progress():
  #       time.sleep(4)

  logger.info("Checkpointing warmup done")

  if args.compile:
    logger.info("Using `torch.compile`")
    model = torch.compile(model, fullgraph=True)

  # Utils
  num_flop_per_token = get_num_flop_per_token(
      get_num_params(model, exclude_embedding=True),
      model_config,
  )

  steps = []
  losses = []
  if not os.path.exists(args.loss_file):
    with open(args.loss_file, "w") as f:
      f.write("step,loss\n")

  ntokens_since_last_log = 0
  ntraining_tokens_since_last_log = 0
  time_last_log = time.perf_counter()
  logger.info("------------------------------------------------------------Starting training!")
  while train_step < args.training_steps:
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

    logits = model(input_ids)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1).float(), labels.flatten(0, 1), reduction="sum")
    loss = loss / num_items_in_batch
    del logits
    loss.backward()

    # insert step and loss
    steps.append(train_step)
    losses.append(loss.item())

    # Clip gradients
    clip_grad_norm_(model.parameters(), args.grad_max_norm)

    while ckpt_monitor.gpu_copy_in_progress():
        continue

    optimizer.step()
    lr_scheduler.step()

    # Logging
    if (train_step == 1 or train_step % args.logging_frequency == 0):
      time_delta = time.perf_counter() - time_last_log
      # tokens per second per device, abbreviated as tps
      tps = ntokens_since_last_log / time_delta 
      mfu = 100 * num_flop_per_token * tps / 989e12
      tflops = num_flop_per_token * tps / 1e12
      training_tps = ntraining_tokens_since_last_log / time_delta

      logger.info(f"Step: {train_step} | Loss: {loss.item():.2f} | Tokens per second: {tps:.2f} | Training tokens per second (%): {100*training_tps/tps:.2f} | MFU (%): {mfu:.2f} | TFLOPs: {tflops:.2f}")
      ntokens_since_last_log = 0
      ntraining_tokens_since_last_log = 0
      time_last_log = time.perf_counter()
    
    # Profiling
    if args.profile and args.profile_step_end == train_step:
      torch.cuda.cudart().cudaProfilerStop()
      
    # Checkpointing
    if train_step % args.checkpoint_freq == 0 or train_step == args.training_steps:
      set_opt_state(gpu_ar, [optimizer], model_size, non_blocking=non_blocking)
      if not model_isfp32:
        set_model_state(gpu_ar, model, non_blocking=non_blocking)
      ckpt_monitor.save()
      save_random_states(step=train_step, ckpt_dir=args.checkpoint_dir, lr_scheduler=lr_scheduler)
      # save loss trace, the whole array
      with open(args.loss_file, "a") as f:
        for i in range(len(losses)):
          f.write(f"{steps[i]},{losses[i]}\n")
      losses = []
      steps = []
      
      logger.info(f"Checkpoint saved to {args.checkpoint_dir}")

  ckpt_monitor.kill_checkpoint()
  logger.info("Training completed")

if __name__ == "__main__":
  init_logger()
  args = get_args()
  os.sched_setaffinity(0, set(list(range(20, 70)))) # cuda:0 NUMA Node Cores
  train(args)
