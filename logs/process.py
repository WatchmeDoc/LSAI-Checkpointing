import re
import os
from typing import List
import pandas as pd

def parse_files(file_paths: List[str]) -> pd.DataFrame:
    # Regex to match time components from training step lines
    time_pattern = re.compile(
        r"Before Fwd Time: ([\d.]+) \| "
        r"Fwd time: ([\d.]+) \| "
        r"Bck time: ([\d.]+) \| "
        r"After Bck Time: ([\d.]+) \| "
        r"GPU Copy Time: ([\d.]+) \| "
        r"Train Time: ([\d.]+) \| "
        r"Previous After Step Time: ([\d.]+)"
    )

    # Regex for final after step time
    final_after_step_pattern = re.compile(
        r"Final previous after step time: ([\d.]+) seconds"
    )

    # Regex for total time taken
    total_time_pattern = re.compile(
        r"total time taken ([\d.]+) seconds"
    )

    results = []

    for file_path in file_paths:
        sums = {
            "Before Fwd Time": 0.0,
            "Fwd time": 0.0,
            "Bck time": 0.0,
            "After Bck Time": 0.0,
            "GPU Copy Time": 0.0,
            "Train Time": 0.0,
            "Previous After Step Time": 0.0,
            "Total Time Taken": 0.0,
        }

        first_after_step_skipped = False

        with open(file_path, 'r') as f:
            for line in f:
                if "- root - INFO - Step:" in line:
                    match = time_pattern.search(line)
                    if match:
                        values = list(map(float, match.groups()))
                        for i, (key, value) in enumerate(zip(list(sums.keys())[:-1], values)):
                            if key == "Previous After Step Time" and not first_after_step_skipped:
                                first_after_step_skipped = True
                                continue
                            sums[key] += value
                else:
                    final_match = final_after_step_pattern.search(line)
                    if final_match:
                        sums["Previous After Step Time"] += float(final_match.group(1))

                    total_time_match = total_time_pattern.search(line)
                    if total_time_match:
                        sums["Total Time Taken"] = float(total_time_match.group(1))

        row = {"File": os.path.basename(file_path)}
        row.update(sums)
        results.append(row)

    df = pd.DataFrame(results)
    return df

file_list = ["flash0_ckpt40_run1.log", "flash0_ckpt40_run2.log", "flash0_ckpt40_run3.log", "flash1_ckpt40_run1.log", "flash1_ckpt40_run2.log", "flash1_ckpt40_run3.log"]
df = parse_files(file_list)
print(df.to_string(index=False))
