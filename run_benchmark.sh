#!/bin/bash

# python -m training.train_pccheck_vgg16 --dataset imagenet  --batchsize 32 --arch vgg16 --cfreq 50 --bench_total_steps 500

# rm -f checkpointing/checkpoints/*.pt

python -m training.train_checkp.py --seed 4 --training-steps 151 --checkpoint-freq 50 --model-dtype fp32 --sequence-length 1024 --loss-file loss_trace_checkp1.csv
python -m training.train_checkp.py --seed 4 --training-steps 151 --checkpoint-freq 50 --model-dtype fp32 --sequence-length 1024 --loss-file loss_trace_checkp2.csv

python -m training.train_checkp.py --seed 4 --training-steps 151 --checkpoint-freq 50 --model-dtype fp32 --sequence-length 1024 --loss-file loss_trace_checkp3_fail50.csv &
sleep(120)
pkill -f train_checkp.py
python -m training.train_checkp.py --seed 4 --training-steps 151 --checkpoint-freq 50 --model-dtype fp32 --sequence-length 1024 --loss-file loss_trace_checkp3_fail50.csv --load-checkpoint

python -m training.train_pccheck --seed 4 --training-steps 151 --checkpoint-freq 50 --model-dtype fp32  --sequence-length 1024 --loss-file loss_trace_pccheck1.csv
python -m training.train_pccheck --seed 4 --training-steps 151 --checkpoint-freq 50 --model-dtype fp32  --sequence-length 1024 --loss-file loss_trace_pccheck2.csv
python -m training.train_pccheck --seed 4 --training-steps 151 --checkpoint-freq 50 --model-dtype fp32  --sequence-length 1024 --loss-file loss_trace_pccheck3_fail50.csv &
sleep(120)
pkill -f train_pccheck.py
python -m training.train_pccheck --seed 4 --training-steps 151 --checkpoint-freq 50 --model-dtype fp32  --sequence-length 1024 --loss-file loss_trace_pccheck3_fail50.csv --load-checkpoint