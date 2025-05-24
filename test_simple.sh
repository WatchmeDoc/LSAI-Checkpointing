#!/bin/bash

python -m training.train_pccheck --seed 4 --training-steps 151 --checkpoint-freq 50 --model-dtype fp32  --sequence-length 2048 --loss-file loss_trace_pccheck.csv
rm -f checkpointing/checkpoints/*.pt
