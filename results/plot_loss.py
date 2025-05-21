import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dtypes = ["fp32", "bf16"]
sl_values = ["sl_2048", "sl_4096"]

# Plot the loss curves
for dtype in dtypes:
    for sl in sl_values:
        pattern = f"{dtype}/{sl}/loss_trace_*.csv"
        csv_files = glob.glob(pattern)

        if not csv_files:
            print(f"No CSV files found for pattern: {pattern}")
            continue

        # Plotting
        plt.figure(figsize=(8, 5))
        for fname in sorted(csv_files):
            df = pd.read_csv(fname)
            if {"step", "loss"} <= set(df.columns):
                label = os.path.splitext(os.path.basename(fname))[0]
                plt.plot(df["step"], df["loss"], label=label)
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Training Loss Across Runs")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"loss_curves_{dtype}_{sl}.png")

