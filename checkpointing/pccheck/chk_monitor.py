import json
from checkpointing.pccheck.chk_checkpoint_pipeline import Checkpoint
from torch.multiprocessing import Pool, Process, set_start_method, Manager, Value, Lock, Barrier


class Chk_monitor:

    def __init__(
        self,
        total_size,
        gpu_ar,
        ratio=2.0,
        bsize=None,
        is_distributed=False,
        rank=0,
        world_size=1,
        max_async=1,
        config="checkpointing/pccheck/pccheck_config.json",
        **kwargs,
    ):

        # only 1 background process
        self.lock = Lock()
        self.cp_in_progress = Value("i", 0)
        self.start = Value("i", 0)
        self.stop = Value("i", 0)
        self.barrier = Barrier(2)

        self.checkpoint_dict = {}

        for name, ref in kwargs.items():
            self.checkpoint_dict[name] = ref

        print(f"BSIZE IS {bsize}")
        with open(config, "r") as f:
            config = json.load(f)
        self.basic_file = config["basic_file"]
        self.max_async = max_async
        self.num_threads = config["num_threads"]
        
        chk = Checkpoint(
            total_size,
            config["num_threads"],
            self.basic_file,
            config["c_lib_path"],
            self.max_async,
            ratio=ratio,
            gpu_ar=gpu_ar,
            bsize=bsize,
            memory_saving=config["memory_saving"],
            is_distributed=is_distributed,
            rank=rank,
            world_size=world_size
        )

        self.chk_process = Process(
            target=chk.start_chk,
            args=[
                self.barrier,
                self.lock,
                self.checkpoint_dict,
                self.cp_in_progress,
                self.start,
                self.stop,
                config["gpu_copy"],
                config["is_sync"],
            ],
        )
        self.chk_process.start()
        self.barrier.wait()
        # print("Chk process started! PID is: ", self.chk_process.pid)

    def gpu_copy_in_progress(self):

        # return True if at least one of the background processes is copying
        with self.lock:
            if self.cp_in_progress.value == 1:
                return True

        return False

    def checkpoint_in_progress(self):
        with self.lock:
            if self.start.value == 1:
                return True

        return False

    def save(self):
        print(f"******************** CALL SAVE ********************")

        while True:
            with self.lock:
                if self.start.value == 0:
                    break

        with self.lock:
            self.cp_in_progress.value = 1
            self.start.value = 1

    def kill_checkpoint(self):

        with self.lock:
            self.stop.value = 1

        self.chk_process.join()
