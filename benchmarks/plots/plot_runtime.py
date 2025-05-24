import pandas as pd
import io
import matplotlib.pyplot as plt
import numpy as np

df_runtime = pd.read_csv("runtimes.csv")

group_order = [
    (1024, 'bf16'),
    (1024, 'fp32'),
    (2048, 'bf16'),
    (2048, 'fp32'),
]

df_runtime['group_label'] = df_runtime.apply(lambda row: f"{row['seq_len']}\n{row['dtype']}", axis=1)

algos = ['baseline', 'pccheck_single', 'pccheck']

bar_width = 0.22
x_indices = np.arange(len(group_order))

plt.figure(figsize=(11, 6))

for i, algo in enumerate(algos):
    heights = []
    for (seq, dtype) in group_order:
        row = df_runtime[
            (df_runtime['seq_len'] == seq) &
            (df_runtime['dtype'] == dtype) &
            (df_runtime['algo'] == algo)
        ]
        heights.append(row['runtime'].values[0] if not row.empty else np.nan)

    plt.bar(
        x_indices + (i - (len(algos) - 1) / 2) * bar_width,
        heights,
        width=bar_width,
        label=algo
    )

plt.title('Overall runtime vs. sequence length & dtype (single figure, 3 algorithms)')
plt.xlabel('Seq Len / Dtype')
plt.ylabel('Runtime (s)')
plt.xticks(x_indices, [f"{seq}\n{dtype}" for seq, dtype in group_order])
plt.legend(title='Algorithm')
plt.tight_layout()
plt.savefig(f"runtime.png")
