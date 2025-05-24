#!/bin/bash

sequence_lengths=(1024 2048)
data_types=("fp32" "bf16")
cd .. 
mkdir -p logs
for dtype in "${data_types[@]}"; do
    for seq_len in "${sequence_lengths[@]}"; do
        LOGFILE="logs/nocheck_dtype_${dtype}_sl_${seq_len}.log"

        echo "==== RUNNING BASELINE FOR $seq_len WITH $dtype ===="

        python -m training.train_checkp \
            --seed 4 \
            --training-steps 800 \
            --checkpoint-freq 900 \
            --model-dtype "$dtype" \
            --sequence-length "$seq_len" \
         > "$LOGFILE" 2>&1

        echo "==== BASELINE END ===="
        # LOGFILE="logs/pccheck_dtype_${dtype}_sl_${seq_len}_t1.log"

        # echo "==== RUNNING PCCHECK FOR $seq_len WITH $dtype ===="

        # python -m training.train_pccheck \
        #     --seed 4 \
        #     --training-steps 800 \
        #     --checkpoint-freq 40 \
        #     --model-dtype "$dtype" \
        #     --sequence-length "$seq_len" \
        #  > "$LOGFILE" 2>&1

        # echo "==== PCCHECK END ===="

    done
done

echo "Benchmarking complete."
