Ambiguous = current-frame kNN (k=20) expert-label disagreement >= 0.6; 48% of val decisions.

| encoder (3 seeds) | val regret all (m) | ambiguous frames | unambiguous frames |
|---|---|---|---|
| frame | 0.695 ± 0.072 | 0.548 ± 0.063 | 0.834 ± 0.127 |
| stack | 0.660 ± 0.160 | 0.481 ± 0.122 | 0.829 ± 0.196 |
| gru | 0.759 ± 0.057 | 0.510 ± 0.077 | 0.992 ± 0.071 |
| ais (λ=0.3) | 0.750 ± 0.045 | 0.603 ± 0.020 | 0.887 ± 0.091 |
