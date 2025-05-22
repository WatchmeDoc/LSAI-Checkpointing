## Introduction
Training large-scale machine learning models requires both a lot of resources utilized for several weeks. As the scale of such deployments increases, the probability of failures also increases along with it. In our case, on the Alps cluster, there could be disruptions due to machine failures,
server maintenance which takes place weekly, or even time limitations that are assigned to each quota. Therefore, having a checkpointing mechanism to pick-up the model training from where it stopped
is essential in order to safely finalize a training process.

Standard checkpointing mechanisms, such as the one presented below, usually pause the training process and checkpoint the parameters to persistent storage.
This, however, induces large overhead, especially when applied in high frequency, yet frequent checkpoints are essential to avoid long recovery times (i.e. the time since the process restarts until it reaches the state it was when the partition took place).

### Implementations
In this project we implement the intra-node version of the checkpointing mechanism outlined in "[PCcheck: Persistent Concurrent Checkpointing for ML](https://anakli.inf.ethz.ch/papers/PCcheck_asplos25.pdf)" (2025, Strati et al) and compare it against a naive baseline checkpointing implementation that uses `torch.save()`. However, related work [2] has shown that `torch.save()` isn't enough to make sure that your changes have been persisted to disk and need to run `fsync()` right after.
While we adapted PCCheck such that it works on a single node, the original system supports checkpointing mechanism for distributed training, and can be adapted properly in the future. But, we will show below the performance benefits of using PCCheck over naive checkpointing only on a single node.

In both cases, the user can use the `--load-checkpoint` flag to load the latest persisted checkpoint and resume the training process.

## Baseline
We implement the baseline using a standard [`pytorch.save`](https://docs.pytorch.org/docs/stable/generated/torch.save.html), where we checkpoint the current training step, the model state, optimizer state and the learning rate scheduler state. We also checkpoint the dataloader state by storing the random states of the relevant libraries (python, numpy, torch, cuda), as well as the step number (`train_step`) we are currently at.

To load the checkpoint, we perform a standard [`pytorch.load`](https://docs.pytorch.org/docs/stable/generated/torch.load.html) to load the aforementioned states. Then, when we need to restore the dataloader state, we iterate `train_step` times through it without any actions.

The complete code implementation can be found under `/training/train_checkp.py`. To run it, the user can use:

```shell
python -m training.train_checkp.py --seed 4 --training-steps 151 --checkpoint-freq 50 --model-dtype fp32 --sequence-length 1024 --loss-file loss_trace_checkp.csv
```

## PCcheck Implementation
The above naive checkpointing mechanism induces high overhead in DNN training jobs, especially when applied in high frequency. PCCheck [1] is a framework for DNN training that supports multiple concurrent checkpoints in parallel. Concurrent checkpoints can help reduce idle GPU time and increase checkpoint frequency, since training does not have to wait for the previous checkpoint to finish, before initiating a new one. However, naively issuing concurrent checkpoints can increase CPU memory and storage overheads, as well as PCIe and storage bandwidth contention, which could degrade training throughput.

### How it works
In short, PCCheck uses a separate GPU buffer that keeps track of the optimizer and model states. When a checkpoint is issued, the model state and optimizer state dictionaries are copied to DRAM using the copy mechanism described on their paper, which in turn is asynchronously persisted to disk while the training loop proceeds unaffected. In order to further reduce stalls, PCCheck supports pipelining by splitting the data into multiple chunks and using multiple threads. The system also supports overlapping checkpoints (e.g. issuing a new checkpoint before the previous one has completed) through its `max_async` parameter, as well as out of the box support for failures during checkpointing, meaning that if the machine crashes during checkpoint, the last persisted checkpoint will be unaffected.

The reader may refer to the original paper for further implementation details, as well as guidance towards how to configure this system (number of threads, checkpoint frequency etc) in Section 3.4.

To run the training process with PCCheck, the user can use:
```shell
python -m training.train_pccheck --seed 4 --training-steps 151 --checkpoint-freq 50 --model-dtype fp32  --sequence-length 1024 --loss-file loss_trace_pccheck1.csv
```

### Issues
While attempting to make the system run on the Alps cluster, we encountered several issues that we had to work on, as the system was built and tested on x86 machines, while GH200 nodes run on Arm64 architecture.

First of all, just like any academic research paper, PCCheck codebase included various baselines, as well as support for other disk types such as Intel Optane Memory, features which interrupted our installation process as we didn't have root privileges on the cluster to install all the required tools. Thus, we had to figure out which parts were actually needed and which weren't to strip down the codebase as much as possible such that we can install it on our machines.

Secondly, as the system uses C++ on the background and some very low level primitives, such as barriers in assembly instructions, we had to make several changes and translate the x86 instructions into Arm64. We also had to profile the GH200 node thoroughly to find how it allocates mmap regions in order to fix the configured file offsets that PCCheck uses to find the data it stores.

Third, as the system was tested on much smaller models and machines, we faced several integer/float overflows in the C++ level, issues which required thorough profiling as they didn't just pop. However, this also presented an opportunity; given that 1 GH200 node holds a whooping 288 CPU cores, the system could further be optimized to take advantage of those massive computational resources.

The above issue brings us to the last issue we faced, regarding multithreading. In x86 machines, using multiple threads brought significant speedup on PCCheck, but this was not the case in Arm64, as it induced some undefined behavior (e.g. large increased overhead, straggling threads, but at some times it would indeed bring the desired speedup). To overcome this, we ... TODO

## Results

### Experiments Setup
We modified on the provided codebase, implementing our new features and changing the model configuration such that its dimension matches the sequence length. As described above, PCCheck uses an auxiliary GPU buffer of the same size as the model + optimizer parameters and thus the model couldn't fit with sequence length of 4096 in a single GPU. Thus, we compare with Transformer dimension = sequence length with values of 1024 and 2048, as well as FP32 and BF16

To test correctness, we launch two runs using a random seed (e.g. 4), with 150 training steps and checkpoints at 50-steps intervals. We then launch a third run with the same parameters but with a failure happening after step 50. Finally, we recover from the checkpoint and we resume training. If everything is set correctly, the 3 plotted lines should overlap 100%.

### Validation
We plot and compare the  losses in the following figure:

![[./training/loss_curves.png]]

From the figure and the log files, we verify that the losses are exactly the same for a given seed and regardless of failure recovery.

### Performance
#### Baseline

We report the following times using Python's ```perf_counter``` timing. Note that we do not consider the time taken to transfer the model from host memory to device memory.

Checkpoint Saving Time: $\approx$ 35 seconds
Checkpoint Loading Time: $\approx$ 12 seconds

## Future Work
There are still several things one may work on to extend this feature. The first and most important one is to profile deeper the use of multiple threads and CPU affinity, in order to leverage the large amount of CPU cores that are available in a single node.

The next important step is to also work on 


### Acknowledgements
We acknowledge that we used help from Generative AI tools, including ChatGPT and GitHub Copilot. We also express our gratitude to doctoral candidate Foteini Strati for her assistance to overcoming the aforementioned issues when adapting PCCheck to the new hardware.

**Team Members:** 
	- George Manos: georgios.manos@infk.ethz.ch
	- Luca Renna: luca.renna@infk.ethz.ch

## References
[1] F. Strati, M. Friedman, A. Klimovic, PCcheck: Persistent Concurrent Checkpointing for ML (ASPLOS 2025)
[2] Mohan, J., Phanishayee, A., & Chidambaram, V. (2021). CheckFreq: Frequent, Fine-Grained DNN Checkpointing. In M. K. Aguilera & G. Yadgar (Eds.), 19th USENIX Conference on File and Storage Technologies (FAST ’21), 203–216. USENIX Association.

