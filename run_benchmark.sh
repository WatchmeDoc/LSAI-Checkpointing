#!/bin/bash

# python -m training.train_pccheck_vgg16 --dataset imagenet  --batchsize 32 --arch vgg16 --cfreq 50 --bench_total_steps 500

# rm -f checkpointing/checkpoints/*.pt

python training/train_checkp.py --seed 4 --loss-file results/fp32/loss_trace_checkp1.csv --training-steps 151 --model-dtype fp32
python training/train_checkp.py --seed 4 --loss-file results/fp32/loss_trace_checkp2.csv --training-steps 151 --model-dtype fp32
python training/train_checkp.py --seed 4 --loss-file results/fp32/loss_trace_checkp3_fail.csv --training-steps 151 --model-dtype fp32 &
sleep(60)
pkill -f train_checkp.py
python training/train_checkp.py --seed 4 --loss-file results/fp32/loss_trace_checkp3_fail.csv --training-steps 151 --model-dtype fp32 --load-checkpoint

python -m training.train_pccheck --seed 4 --loss-file results/fp32/loss_trace_pccheck1.csv --training-steps 151 --model-dtype fp32
python -m training.train_pccheck --seed 4 --loss-file results/fp32/loss_trace_pccheck2.csv --training-steps 151 --model-dtype fp32
python -m training.train_pccheck --seed 4 --loss-file results/fp32/loss_trace_pccheck3_fail.csv --training-steps 151 --model-dtype fp32 &
sleep(60)
pkill -f train_pccheck.py
python -m training.train_pccheck --seed 4 --loss-file results/fp32/loss_trace_pccheck3_fail.csv --training-steps 151 --model-dtype fp32 --load-checkpoint

