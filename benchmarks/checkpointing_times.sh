#!/bin/bash

sequence_lengths=(1024 2048)
data_types=("fp32" "bf16")

for dtype in "${data_types[@]}"; do
    for seq_len in "${sequence_lengths[@]}"; do

        echo "==== RUNNING BASELINE FOR $seq_len WITH $dtype ===="

        python -m training.torch_save_baseline \
            --seed 4 \
            --training-steps 10 \
            --checkpoint-freq 1 \
            --model-dtype $dtype \
            --sequence-length $seq_len \
            --warmup 0

        echo "==== BASELINE END ===="
        echo "==== RUNNING PCCHECK FOR $seq_len WITH $dtype ===="

        python -m training.train_pccheck \
            --seed 4 \
            --training-steps 10 \
            --checkpoint-freq 1 \
            --model-dtype $dtype \
            --sequence-length $seq_len \
            --warmup 0
        
        echo "==== PCCHECK END ===="
    done
done

echo "Benchmarking complete."