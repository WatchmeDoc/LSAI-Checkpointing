#!/bin/bash

python -m training.train_pccheck_vgg16 --dataset imagenet  --batchsize 32 --arch vgg16 --cfreq 50 --bench_total_steps 500

rm -f checkpointing/checkpoints/*.pt
