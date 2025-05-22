import os
from platform import node
import sys
import torch
import argparse
import torch.multiprocessing as mp
import time

import ctypes
from checkpointing.pccheck.chk_checkpoint_pipeline import return_offset
from checkpointing.pccheck.chk_monitor import Chk_monitor


import mmap
PAGE_SIZE = 65536 #mmap.PAGESIZE

parser = argparse.ArgumentParser(
    description="Checks"
)
parser.add_argument('--load',
                    default=False,
                    action='store_true',
                    help='Load checkpoint')

def load_and_check(ckpt_path, max_async, total_size):
    offset = return_offset(
        "checkpointing/pccheck/libtest_ssd.so",
        ckpt_path,
        max_async,
        total_size
    )
    print(f"--------- offset is {offset}")
    gpu_ar_data = torch.load("./checkpointing/checkpoints/checkpoint_pccheck_ar.pt", map_location="cpu", weights_only=False)
    numel = gpu_ar_data.numel()
    with open(ckpt_path, "rb") as f:
        f.seek(offset, os.SEEK_SET)
        # convert numpy array into torch tensor
        payload = f.read(numel*4)
        pccheck_data = torch.frombuffer(payload, dtype=torch.float32)

    print(pccheck_data)
    print(gpu_ar_data)
    assert torch.equal(pccheck_data, gpu_ar_data)


def train(total_size):
    device = torch.device(f"cuda:{int(os.getenv('LOCAL_RANK', 0))}")
    mp.set_start_method("spawn", force=True)
    gpu_ar = torch.rand(total_size).to(device)

    chk_monitor = Chk_monitor(
        total_size,
        gpu_ar=gpu_ar,
        bsize=total_size,
        model=None,
        optimizer=None,
    )

    for i in range(5):
        print(f"Iteration {i}")
        time.sleep(1)

        while chk_monitor.gpu_copy_in_progress():
            continue
        # rand_ar = torch.rand(total_size).to(device)
        # gpu_ar.copy_(rand_ar)

        # start = time.time()
        # torch.save(gpu_ar, "checkpointing/checkpoints/checkpoint_pccheck_ar.pt")
        # f = open("checkpointing/checkpoints/checkpoint_pccheck_ar.pt", "a+")
        # os.fsync(f.fileno())
        # f.close()

        # print(f"******************* saving took {time.time()-start} sec")
        chk_monitor.save()

        while chk_monitor.gpu_copy_in_progress():
            continue

    if chk_monitor:
        chk_monitor.kill_checkpoint()

if __name__ == "__main__":
    args = parser.parse_args()
    os.sched_setaffinity(0, set(list(range(43, 71))))
    total_size = 3422751890 * 2

    if args.load:
        load_and_check(
            "./checkpointing/checkpoints/checkpoint_pccheck.pt", 1, total_size)
    else:
        train(total_size)
