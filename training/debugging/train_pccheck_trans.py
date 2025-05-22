""" Basically most vision models. """

import os
from platform import node
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torchvision import models, datasets, transforms
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
from torch.multiprocessing import Pool, Process, set_start_method, Manager, Value, Lock
from datetime import timedelta
import random
import numpy as np
import time
import os
import argparse

import ctypes
from checkpointing.pccheck.chk_monitor import Chk_monitor
from checkpointing.pccheck_utils import initialize, get_total_size, set_storage

import os
import time

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from training.dataset import CollatorForCLM, ParquetDataset
from training.model import Transformer, TransformerModelArgs
from training.utils import build_lr_scheduler, clip_grad_norm_, get_args, get_num_params, get_num_flop_per_token, init_logger, logger, PRECISION_STR_TO_DTYPE, set_default_dtype, set_seed, save_random_states, load_random_states, load_checkpoint

from checkpointing.pccheck.chk_monitor import Chk_monitor
from checkpointing.pccheck_utils import initialize, get_total_size, set_storage

# preprocessing sources:
# https://github.com/pytorch/examples/blob/main/imagenet/main.py

parser = argparse.ArgumentParser(
    description="PyTorch ImageNet/CIFAR Training or inference using torchvision models"
)

parser.add_argument("--arch", default="resnet18", type=str, help="torchvision model")
parser.add_argument(
    "--batchsize", default=1, type=int, help="batch size for training"
)
parser.add_argument("--train_dir", default="", type=str, help="path to dataset")
parser.add_argument("--cfreq", default=50, type=int, help="Checkpoint Frequency")
parser.add_argument(
    "--psize", default=1, type=int, help="Number of chunks for pipeline"
)
parser.add_argument(
    "--bench_total_steps", default=1000, type=int, help="Number of steps to train for"
)

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
    default="bf16",
    help="Model dtype for parameters, gradients and optimizer states. Default: bf16",
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
    default="./checkpoints",
    help="Directory to save checkpoints"
)
parser.add_argument(
    "--load-checkpoint",
    action='store_true',
    help="Loads the latest checkpoint from the checkpoint directory"
)
args = parser.parse_args()


def train():

    print(f"Process with pid {os.getpid()}, args is {args}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)
    train_ds = ParquetDataset(args.dataset, tokenizer, args.sequence_length, args.batch_size*args.training_steps)
    train_collator = CollatorForCLM(args.sequence_length, tokenizer.pad_token_id)
    train_dl = DataLoader(train_ds,
                            batch_size=args.batch_size,
                            collate_fn=train_collator)
    train_dl_iterator = iter(train_dl)

    local_rank = 0
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    model = Transformer(TransformerModelArgs(
        dim=args.sequence_length,
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        multiple_of=1024,
        rope_theta=500000,
        vocab_size=tokenizer.vocab_size,
        seq_len=args.sequence_length,
    ))
    model = model.to(device)  # to GPU

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, fused=args.fused_optimizer)
    lr_scheduler = build_lr_scheduler(optimizer, args.lr_warmup_steps)

    metric_fn = torch.nn.CrossEntropyLoss().to(0)

    model.train()



    # for checkpoint
    mp.set_start_method("spawn", force=True)
    gpu_ar, total_size = initialize(model, [optimizer])


    # assume gpu_ar is big 1D GPU tensor
    set_storage(model, [optimizer], gpu_ar)
    torch.cuda.empty_cache()
    if args.cfreq > 0:
        chk_monitor = Chk_monitor(
            total_size,
            gpu_ar=gpu_ar,
            bsize=total_size,
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
        )
    else:
        chk_monitor = None

    batch_idx = 0
    steps_since_checkp = 0
    checkpoints = 0
    warmup = 3

    start_train_time = time.time()

    batch_idx = 0

    start_iter = time.time()
    while batch_idx < args.bench_total_steps:

        input_ids, labels = next(train_dl_iterator)

        num_items_in_batch = labels.ne(-100).sum()

        input_ids = input_ids.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(input_ids)
        loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1).float(), labels.flatten(0, 1), reduction="sum")
        loss = loss / num_items_in_batch
        del logits
        loss.backward()

        # Clip gradients
        clip_grad_norm_(model.parameters(), args.grad_max_norm)

        if chk_monitor:
            while chk_monitor.gpu_copy_in_progress():
                continue

        optimizer.step()
        lr_scheduler.step()

        # FOR CHECKING
        # grads = []
        # for group in optimizer.param_groups:
        #     for p in group['params']:
        #         if p.grad is not None:
        #             grads.append(p.grad[0][0])
        #             break

        if (batch_idx == warmup) or (
            (args.cfreq > 0) and steps_since_checkp == args.cfreq - 1
        ):
            if args.cfreq > 0:
                chk_monitor.save()
                steps_since_checkp = 0
                checkpoints += 1
            if batch_idx == warmup:
                print(f"Start clock!")
                start_train_time = time.time()
        else:
            steps_since_checkp += 1

        batch_idx += 1
        print(f"Step {batch_idx} took {time.time()-start_iter}")

        start_iter = time.time()

    end_train_time = time.time()
    total_train_time = end_train_time - start_train_time
    if chk_monitor:
        chk_monitor.kill_checkpoint()

    print(
        f"-- BENCHMARK ENDED: Total time: {total_train_time} sec, Number of iterations: {batch_idx}, Number of checkpoints: {checkpoints}"
    )
    print(f"EXECUTION TIME: {total_train_time} sec")
    print(f"THROUGHPUT IS {(args.bench_total_steps-warmup)/total_train_time}")


if __name__ == "__main__":
    args = parser.parse_args()
    os.sched_setaffinity(0, {0})
    train()
