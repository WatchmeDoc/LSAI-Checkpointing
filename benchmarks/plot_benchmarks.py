#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Read benchmark CSV
    df = pd.read_csv('runtimes.csv')

    # Create a category column combining dtype and sequence_length
    df['category'] = df['dtype'] + '_' + df['sequence_length'].astype(str)

    # Define the ordered categories and series keys
    categories = ['bf16_1024', 'bf16_2048', 'fp32_1024', 'fp32_2048']
    series_keys = [
        ('nocheck', 1),
        ('checkp', 1),
        ('pccheck', 1),
        ('pccheck', 4),
    ]
    
    key_mapping = {
        "nocheck_t1": "No Checkpointing",
        "checkp_t1": "torch.save",
        "pccheck_t1": "PCCheck Single Thread",
        "pccheck_t4": "PCCheck 4 Threads"
    }

    # Extract times for each series across categories
    times = {key: [] for key in series_keys}
    for cat in categories:
        sub = df[df['category'] == cat]
        for key in series_keys:
            method, threads = key
            val = sub[(sub['checkpointing_name'] == method) & (sub['num_threads'] == threads)]['time']
            if not val.empty:
                times[key].append(val.iloc[0])
            else:
                times[key].append(np.nan)

    # Plotting
    x = np.arange(len(categories))
    width = 0.2

    fig, ax = plt.subplots()
    for i, key in enumerate(series_keys):
        ax.bar(x + i*width, times[key], width, label=key_mapping[f"{key[0]}_t{key[1]}"])

    ax.set_xticks(x + width * (len(series_keys) - 1) / 2)
    ax.set_xticklabels(categories)
    ax.set_xlabel('Configuration (dtype_sequence_length)')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Training Time by Configuration and Threads')
    ax.legend()
    plt.tight_layout()
    plt.savefig('plots/benchmark_plot.png')

if __name__ == '__main__':
    main()
