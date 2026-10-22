| encoder (DINOv2-base features, 3 seeds) | params | val regret (m) | val transition MSE |
|---|---|---|---|
| frame (memoryless) | 202656 | 0.695 ± 0.072 | — |
| stack-3 | 251936 | 0.660 ± 0.160 | — |
| GRU, task loss | 301728 | 0.759 ± 0.057 | — |
| AIS-GRU, λ=1 | 328032 | 0.796 ± 0.008 | 0.871 |
| AIS-GRU, λ=0.3 | 328032 | 0.750 ± 0.045 | 0.731 |
| AIS-GRU, λ=0.1 | 328032 | 0.772 ± 0.025 | 0.841 |

Always-FOLLOW: 0.801 m. Same data, objective, optimiser and epochs for all rows.
