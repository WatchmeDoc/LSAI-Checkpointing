#!/usr/bin/env bash
set -euo pipefail

# Output CSV
OUTPUT="benchmarks/summary.csv"
echo "checkpointing_name,dtype,sequence_length,num_threads,time" > "$OUTPUT"

# Loop over all log files
for logfile in logs/*.log; do
  filename=$(basename "$logfile")
  base="${filename%.log}"

  # Parse filename into components
  if [[ $base =~ ^([^_]+)_dtype_([^_]+)_sl_([^_]+)_t([0-9]+)$ ]]; then
    checkpointing_name="${BASH_REMATCH[1]}"
    dtype="${BASH_REMATCH[2]}"
    sequence_length="${BASH_REMATCH[3]}"
    num_threads="${BASH_REMATCH[4]}"

    # Extract the last "total time taken" value
    time=$(grep "Training run end" "$logfile" \
           | tail -n1 \
           | awk '{print $(NF-1)}')

    # Append a CSV line
    echo "${checkpointing_name},${dtype},${sequence_length},${num_threads},${time}" \
      >> "$OUTPUT"
  else
    echo "Warning: skipping '$filename' (unexpected format)" >&2
  fi
done

echo "Done – summary written to $OUTPUT"