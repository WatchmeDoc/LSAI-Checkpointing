import pandas as pd
import io
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("checkpoint_times.csv")

cp_cols = [f'cp{i}' for i in range(1, 11)]
df['avg_time'] = df[cp_cols].mean(axis=1)

means = df.groupby(['seq_len', 'dtype', 'algo'])['avg_time'].mean().reset_index()

pivot = means.pivot_table(index=['seq_len', 'dtype'], columns='algo', values='avg_time').reset_index()

seq_lengths = sorted(pivot['seq_len'].unique())
dtypes = ['bf16', 'fp32']
bar_width = 0.3
gap = 0.15

cluster_centers = np.arange(len(seq_lengths)) * (len(dtypes) * (bar_width + gap) + 0.4)
algo_colors = {'pccheck': 'tab:blue', 'baseline': 'tab:orange'}

plt.figure(figsize=(7, 4))

for i, dtype in enumerate(dtypes):
    subset = pivot[pivot['dtype'] == dtype]
    positions = cluster_centers + i * (bar_width + gap)
    
    pccheck_vals = subset['pccheck'].values
    baseline_vals = subset['baseline'].values
    
    plt.bar(positions, pccheck_vals, width=bar_width,
            color=algo_colors['pccheck'],
            label='pccheck' if (i == 0) else None)
    
    plt.bar(positions, baseline_vals, width=bar_width,
            bottom=pccheck_vals,
            color=algo_colors['baseline'],
            label='baseline' if (i == 0) else None)
    
    for x_pos, total_height in zip(positions, pccheck_vals + baseline_vals):
        plt.text(x_pos, total_height + 0.5, dtype,
                 ha='center', va='bottom', fontsize=9, rotation=0)

plt.xticks(cluster_centers + (len(dtypes)-1)*(bar_width+gap)/2, seq_lengths)
plt.xlabel('Sequence Length')
plt.ylabel('Checkpoint Time (s)')
plt.title('Average Checkpoint Time by DType')
plt.legend(title='Algo')
plt.grid(axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig("benchmarks.png")
