import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pattern = "fp32/loss_trace_*.csv"
csv_files = glob.glob(pattern)

# Plot the loss curves

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
plt.savefig("loss_curves.png")