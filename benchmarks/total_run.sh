#!/usr/bin/env bash

sequence_lengths=(1024 2048)
data_types=("fp32" "bf16")

overall_start=$(date +%s)

for dtype in "${data_types[@]}"; do
    for seq_len in "${sequence_lengths[@]}"; do
        combo_start=$(date +%s)
        echo "==== RUNNING BASELINE FOR $seq_len WITH $dtype ===="

        start=$(date +%s)
        python -m training.torch_save_baseline \
            --seed 4 \
            --training-steps 400 \
            --checkpoint-freq 20 \
            --model-dtype "$dtype" \
            --sequence-length "$seq_len" \
            --warmup 0
        echo "Baseline duration: $(( $(date +%s) - start )) s"
        echo "==== BASELINE END ===="

        echo "==== RUNNING PCCHECK FOR $seq_len WITH $dtype ===="
        start=$(date +%s)
        python -m training.train_pccheck \
            --seed 4 \
            --training-steps 400 \
            --checkpoint-freq 20 \
            --model-dtype "$dtype" \
            --sequence-length "$seq_len" \
            --warmup 0
        echo "PCCHECK duration: $(( $(date +%s) - start )) s"
        echo "==== PCCHECK END ===="

        echo
    done
done

echo "Benchmarking complete"
