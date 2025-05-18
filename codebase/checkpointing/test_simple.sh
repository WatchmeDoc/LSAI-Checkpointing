#!/bin/bash

python train_pccheck.py --dataset imagenet  --batchsize 32 --arch vgg16 --cfreq 50 --bench_total_steps 500 --max-async 1 \
 --num-threads 1 --c_lib_path pccheck/libtest_ssd.so

rm -f pccheck_checkpoint.chk

