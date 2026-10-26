M2 memory comparison: 197 train / 46 held-out episodes (12 held-out train scenes), 3 seeds; held-out positives: collision 18/781, offroad 19/781

| memory | collision AUROC | collision Brier | offroad AUROC | offroad Brier | progress-gain MAE | next-embed NLL | next-embed cosine | best epoch |
|---|---|---|---|---|---|---|---|---|
| frame | 0.696 ± 0.104 | 0.023 ± 0.001 | 0.778 ± 0.041 | 0.025 ± 0.002 | 0.015 ± 0.003 | -0.160 ± 0.002 | 0.735 ± 0.003 | 86.667 ± 18.856 |
| stack | 0.548 ± 0.098 | 0.023 ± 0.000 | 0.764 ± 0.046 | 0.024 ± 0.000 | 0.015 ± 0.002 | -0.138 ± 0.005 | 0.724 ± 0.005 | 66.667 ± 16.997 |
| gru | 0.691 ± 0.073 | 0.022 ± 0.000 | 0.758 ± 0.052 | 0.024 ± 0.001 | 0.018 ± 0.001 | -0.150 ± 0.003 | 0.741 ± 0.015 | 83.333 ± 16.997 |
