#!/usr/bin/env bash
set -euo pipefail

# Create logs directory if it doesn't exist
mkdir -p logs

# Fixed settings
SEED=4
TRAIN_STEPS=800
MODEL_DTYPE=bf16
SEQ_LEN=2048
LOSS_FILE=loss_trace_pccheck.csv

# Loop over flash attention flags and checkpoint frequencies
for CKPT_FREQ in 40 900; do
  for FLASH in 0 1; do
    for RUN in 1 2 3; do
      LOGFILE="logs/flash${FLASH}_ckpt${CKPT_FREQ}_run${RUN}.log"
      echo "Running: FLASH=$FLASH, CKPT_FREQ=$CKPT_FREQ, run #$RUN → $LOGFILE"
      
      USE_FLASH_ATTENTION="$FLASH" \
      python -m training.train_pccheck \
        --seed "$SEED" \
        --training-steps "$TRAIN_STEPS" \
        --checkpoint-freq "$CKPT_FREQ" \
        --model-dtype "$MODEL_DTYPE" \
        --sequence-length "$SEQ_LEN" \
        --loss-file "$LOSS_FILE" \
      > "$LOGFILE" 2>&1
    done
  done
done

echo "All runs completed. Logs are in ./logs/"