#!/usr/bin/env bash
set -euo pipefail

# Temporary storage
TMPFILE=$(mktemp)
echo "Initializing"

# Collect name,time pairs
for logfile in logs/*.log; do
  echo "logfile: " "$logfile"
  # derive the config name, e.g. "flash0_ckpt40"
  base=$(basename "$logfile" .log)
  name=${base%_run*}
  # extract the floating-point time value (second-to-last token)
  time=$(grep "Training run end" "$logfile" \
         | tail -n1 \
         | awk '{print $(NF-1)}')
  echo "$name,$time" >> "$TMPFILE"
done
echo "Collected run times from logs into $TMPFILE"
# Compute average and std. dev. using awk (requires gawk for sqrt())
awk -F, '
{
  cnt[$1]++
  sum[$1]+=$2
  sumsq[$1]+=$2*$2
}
END {
  print "name,avg_time,std_time"
  for (cfg in cnt) {
    mean = sum[cfg] / cnt[cfg]
    var  = sumsq[cfg]/cnt[cfg] - mean*mean
    if (var < 0) var = 0          # guard against tiny negatives
    std  = sqrt(var)
    # format to two decimal places
    printf "%s,%.2f,%.2f\n", cfg, mean, std
  }
}' "$TMPFILE" > summary.csv

# Cleanup
rm "$TMPFILE"

echo "Wrote summary.csv with average & std-dev of run times."