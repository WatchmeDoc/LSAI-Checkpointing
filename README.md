<div align="center">
    <header><h1>Large-Scale AI Engineering: PCCheck Integration
</h1></header>
    <a href="#">
    <img src="https://img.shields.io/badge/Python-3.12-306998">
    </a>
    <a href="#">
        <img src="https://img.shields.io/badge/Conda-25.1.1-44903d">
    </a>
    <br>
    <a href="https://github.com/eth-easl/pccheck"><strong>Check out PCCheck »</strong></a>
</div>
<br>

Project for the course [Large-Scale AI Engineering](https://ai.ethz.ch/education/lectures-and-seminars/large-scale-ai-engineering.html), carried out by Dr. Arnout Devos and Dr. Imanol Schlág.

The project's goal is to implement a feature in a shared codebase with other students. In this work, we integrated PCCheck system, an efficient checkpointing mechanism with minimal interference to the training process.

## Getting Started

PCCheck was developed for Linux Ubuntu 22.04, x86 architecture, and we made slight adaptations to support arm64 (which is the architecture in GH200 Nodes)

### Alps Cluster

If using the Alps cluster, first get an interactive shell on a debug node to build and test the project to make sure everything works:

```shell
srun --account=a-large-sc --container-writable --environment=my_env -p debug --pty bash
```

### Other

For other machines, you probably have to install several packages, using the `install_preq_at_vm.sh` script, including cuda drivers etc. As our primary focus is the Alps cluster, we haven't tested it.

### PCCheck Installation

First, compile the cpp code:

```shell
bash install.sh
```

Then, test that pccheck runs via simple example:

```shell
cd checkpointing && bash test_simple.sh && cd ../
```


## Team Members

- Luca Renna, luca.renna@inf.ethz.ch
- George Manos, georgios.manos@inf.ethz.ch


