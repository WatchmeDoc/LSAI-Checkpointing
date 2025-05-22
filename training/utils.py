import argparse
import functools
import logging
from checkpointing.pccheck.chk_checkpoint_pipeline import return_offset
from contextlib import contextmanager
import pickle
import os
import numpy as np
import random
import torch
from torch.optim.lr_scheduler import LambdaLR
import mmap
from copy import deepcopy
logger = logging.getLogger()

PRECISION_STR_TO_DTYPE = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
    "fp64": torch.float64,
}

# PR_ADDR_DATA = PR_ADDR + (max_async+3)*OFFSET_SIZE;
PAGE_SIZE = mmap.PAGESIZE

def set_seed(seed: int):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)

def save_random_states(step: int, lr_scheduler, ckpt_dir: str):
  def get_rng_state_dict():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }
  state = get_rng_state_dict()
  state["step"] = step
  state["lr_scheduler"] = lr_scheduler.state_dict()
  os.makedirs(ckpt_dir, exist_ok=True)
  ckpt_path = os.path.join(ckpt_dir, "rng_states.pt")
  with open(ckpt_path, "wb") as f:
    pickle.dump(state, f)
  logger.info(f"Random states checkpoint saved to {ckpt_path}")

def load_random_states(ckpt_dir: str):
    """
    Load random state checkpoint from a file and return the step.
    """
    def set_rng_state_dict(state: dict):
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])
        torch.cuda.set_rng_state_all(state["cuda"])

    ckpt_path = os.path.join(ckpt_dir, "rng_states.pt")
    with open(ckpt_path, "rb") as f:
        state = pickle.load(f)
    set_rng_state_dict(state)
    logger.info(f"Random state checkpoint loaded from {ckpt_path}")
    return state

def load_checkpoint(ckpt_dir: str, max_async: int, model: torch.nn.Module, optimizer_list: list[torch.optim.Optimizer], total_size: int, model_dtype: torch.dtype, non_blocking: bool = False):
    """
    Load model checkpoint from a file. Only loads model and optimizer states.
    """
    ckpt_path = os.path.join(ckpt_dir, "checkpoint_pccheck.pt")
    if not os.path.exists(ckpt_path):
        raise ValueError(f"Checkpoint {ckpt_path} does not exist")

    offset = return_offset(
        "checkpointing/pccheck/libtest_ssd.so",
        ckpt_path,
        max_async,
        total_size
    )

    with open(ckpt_path, "rb") as f:
        f.seek(offset, os.SEEK_SET)
        # convert numpy array into torch tensor
        payload = f.read(total_size * 4)
        data = torch.frombuffer(payload, dtype=torch.float32)
    # de-serialize the checkpoint into the model.
    # Perform the invert operation of set_storage
    start_idx = 0
    for name, ref in model.named_parameters():
        end_idx = start_idx + ref.numel()
        my_ar = data[start_idx:end_idx].reshape(ref.shape).to(model_dtype)
        with torch.no_grad():
            ref.copy_(my_ar, non_blocking=non_blocking)
        start_idx += ref.numel()

    for optimizer in optimizer_list:
        opt_state = optimizer.state_dict()
        opt_state_copy = deepcopy(opt_state)
        for name, _ in opt_state['state'].items():
            for k, ref in opt_state['state'][name].items():
                if (torch.is_tensor(ref)):
                    end_idx = start_idx + ref.numel()
                    t = data[start_idx:end_idx].reshape(ref.shape).to(model_dtype)
                    opt_state_copy['state'][name][k].copy_(t, non_blocking=non_blocking)
                    start_idx += ref.numel()
                elif (type(ref) == int or type(ref) == float):
                    opt_state_copy['state'][name][k] = ref
                    start_idx += 1
        # load learning rate
        opt_state_copy['param_groups'][0]['lr'] = data[start_idx]
        start_idx += 1
        optimizer.load_state_dict(opt_state_copy)
    if non_blocking:
        torch.cuda.synchronize()
    logger.info(f"Checkpoint loaded from {ckpt_path}")

def set_opt_state(gpu_ar, optimizer_list, offset, non_blocking: bool = False):
    print("Setting optimizer state...")
    start_idx = offset
    for optimizer in optimizer_list:
        opt_state = optimizer.state_dict()
        for name, _ in opt_state['state'].items():
            for _, ref in opt_state['state'][name].items():
                if (torch.is_tensor(ref)):
                    gpu_ar[start_idx:start_idx+ref.numel()].copy_(ref.to(torch.float32).flatten(), non_blocking=non_blocking)
                    start_idx += ref.numel()
                elif (type(ref) == int or type(ref) == float):
                    gpu_ar[start_idx] = ref
                    start_idx += 1
        # store learning rate
        gpu_ar[start_idx] = optimizer.param_groups[0]['lr']
        start_idx += 1
        if non_blocking:
            torch.cuda.synchronize()

def set_model_state(gpu_ar, model, non_blocking: bool = False):
    """
    Set the model state in the GPU array. Assumes gpu_ar is in fp32 and model is in anything but.
    """
    print("Setting model state...")
    start_idx = 0
    for _, ref in model.named_parameters():
        gpu_ar[start_idx:start_idx+ref.numel()].copy_(ref.to(torch.float32).flatten(), non_blocking=non_blocking)
        start_idx += ref.numel()
    if non_blocking:
        torch.cuda.synchronize()
    return start_idx


def init_logger():
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)

def get_num_params(model: torch.nn.Module, exclude_embedding: bool = False) -> int:
    num_params = sum(p.numel() for p in model.parameters())
    if exclude_embedding:
        num_params -= sum(
            sum(p.numel() for p in m.parameters())
            for m in model.children()
            if isinstance(m, torch.nn.Embedding)
        )
    return num_params


def get_num_flop_per_token(num_params: int, model_config) -> int:
    l, h, q, t = (
        model_config.n_layers,
        model_config.n_heads,
        model_config.dim // model_config.n_heads,
        model_config.seq_len,
    )
    # Reasoning behind the factor of 12 for the self-attention part of the formula:
    # 1. each self-attention has 2 matmul in the forward and 4 in the backward (6)
    # 2. the flash attention does 1 more matmul recomputation in the backward
    #    but recomputation should not be counted in calculating MFU           (+0)
    # 3. each matmul performs 1 multiplication and 1 addition                 (*2)
    # 4. we follow the convention and do not account for sparsity in causal attention
    flop_per_token = 6 * num_params + 12 * l * h * q * t

    return flop_per_token


def build_lr_scheduler(optimizer: torch.optim, warmup_steps: int):

    def linear_warmup_constant(
        warmup_steps: int, current_step: int
    ) -> float:
        """Computes linear warmup followed by linear decay.

        Per LambdaLR requirement, this is accomplished by returning
        a multiplicative factor to adjust the learning rate to
        create the desired schedule.
        """
        if current_step < warmup_steps:
            # linear warmup
            # 0-indexed step, hence + 1 adjustments
            current_step += 1
            curr_adjustment = float(current_step / (warmup_steps + 1))

        else:
            # constant
            curr_adjustment = 1

        return curr_adjustment

    lr_lambda = functools.partial(linear_warmup_constant, warmup_steps)
    return LambdaLR(optimizer, lr_lambda)
    
@torch.no_grad()
def clip_grad_norm_(parameters, grad_max_norm):
  grads = [p.grad for p in parameters if p.grad is not None]
  total_norm = torch.nn.utils.get_total_norm(grads, error_if_nonfinite=True)
  torch.nn.utils.clip_grads_with_norm_(parameters, grad_max_norm, total_norm)
  return total_norm

@contextmanager
def set_default_dtype(dtype: torch.dtype):
    """
    Context manager to set torch's default dtype.
    """
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(old_dtype)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="/capstor/store/cscs/ethz/large-sc/datasets/train_data.parquet",
        help="Path to a parquet file containing a 'text' column with documents (`str`)",
    )
    parser.add_argument(
        "--tokenizer-name-or-path",
        type=str,
        default="unsloth/Mistral-Nemo-Base-2407-bnb-4bit",
        help="A path to a directory containing vocabulary files required by the tokenizer or the model id of a predefined tokenizer hosted inside a model repo on the Hugging Face Hub.",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=2048,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--fused-optimizer",
        action='store_true',
        help="Set to fuse the optimizer for increased performance or not"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
    )
    parser.add_argument(
        "--lr-warmup-steps",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--training-steps",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--logging-frequency",
        type=int,
        default=5,
        help="Log every `--logging-frequency` steps"
    )
    parser.add_argument(
        "--profile",
        action='store_true',
        help="Profile the run using the NSYS profiler"
    )
    parser.add_argument(
        "--profile-step-start",
        type=int,
        default=10,
        help="Starting step to profile using the NSYS profiler"
    )
    parser.add_argument(
        "--profile-step-end",
        type=int,
        default=12,
        help="Last step to profile using the NSYS profiler"
    )
    parser.add_argument(
        "--grad-max-norm",
        type=float,
        default=1,
    )
    parser.add_argument(
        "--model-dtype",
        type=str,
        default="fp32",
        help="Model dtype for parameters, gradients and optimizer states. Default: fp32",
    )
    parser.add_argument(
        "--compile",
        action='store_true',
        help="Set to compile the model with `torch.compile`"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./checkpointing/checkpoints",
        help="Directory to save checkpoints"
    )
    parser.add_argument(
        "--loss-file",
        type=str,
        default="loss_trace.csv",
        help="Directory to save loss trace"
    )

    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=50,
        help="Save a checkpoint every N training steps"
    )
    parser.add_argument(
        "--max-async",
        type=int,
        default=1,
        help="Maximum parallelism when writing checkpoints to disk"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Warmup steps for checkpointing"
    )
    parser.add_argument(
        "--load-checkpoint",
        action='store_true',
        help="Loads the latest checkpoint from the checkpoint directory"
    )
    parser.add_argument(
        "--non-blocking",
        action='store_true',
        help="Enables non-blocking copy for model and optimizer states"
    )
    args = parser.parse_args()
    if not "/" in args.loss_file:
        args.loss_file = f"./results/{args.model_dtype}/sl_{args.sequence_length}/{args.loss_file}"
    return args