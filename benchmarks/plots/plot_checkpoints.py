import pandas as pd
import io
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("checkpoints.csv")

checkpoint_cols = [f'cp{i}' for i in range(1, 11)]
df['avg_time'] = df[checkpoint_cols].mean(axis=1)

group_order = [
    (1024, 'bf16'),
    (1024, 'fp32'),
    (2048, 'bf16'),
    (2048, 'fp32'),
]

df['group_label'] = df.apply(lambda row: f"{row['seq_len']}\n{row['dtype']}", axis=1)

algos = ['baseline', 'pccheck_single', 'pccheck']

bar_width = 0.22
x_indices = np.arange(len(group_order))

plt.figure(figsize=(11, 6))

for i, algo in enumerate(algos):
    heights = []
    for (seq, dtype) in group_order:
        row = df[(df['seq_len'] == seq) &
                 (df['dtype'] == dtype) &
                 (df['algo'] == algo)]
        heights.append(row['avg_time'].values[0] if not row.empty else np.nan)

    plt.bar(
        x_indices + (i - (len(algos) - 1) / 2) * bar_width,
        heights,
        width=bar_width,
        label=algo
    )

plt.title('Average checkpoint time vs. sequence length and dtype')
plt.xlabel('Seq Len / Dtype')
plt.ylabel('Average Time (s)')
plt.xticks(x_indices, [f"{seq}\n{dtype}" for seq, dtype in group_order])
plt.legend(title='Algorithm')
plt.tight_layout()
plt.savefig(f"checkpoint_time.png")
