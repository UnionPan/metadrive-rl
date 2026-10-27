# M3/M4 open-loop ablation: what does the adapter's gain come from?

Flow SFT of a gated adapter on VaVAM-B (muP base shapes), expert-replay windows of 8 queries, 8 epochs, 312 held-out windows from 12 train scenes. Each run reports its best held-out epoch; ± is the spread over 2 seeds. Commands: `runs/sft/ablation*.sh`; script: `scripts/sft_ablation_report.py`.

Stock VaVAM-B (zero gate): held-out flow 0.0858, ADE 3.34 m, FDE 5.69 m, longitudinal median -1.30 m

| variant | runs | held-out flow | ADE m | FDE m | lateral med. m | longitudinal med. m | best epoch |
|---|---|---|---|---|---|---|---|
| abl_motion | 2 | 0.041 ± 0.000 | 2.433 ± 0.001 | 3.681 ± 0.006 | -0.070 ± 0.000 | -0.324 ± 0.057 | 7.000 ± 0.000 |
| abl_frame | 2 | 0.037 ± 0.001 | 2.252 ± 0.106 | 4.235 ± 0.147 | -0.011 ± 0.003 | -0.245 ± 0.024 | 3.000 ± 0.000 |
| abl_stack | 2 | 0.037 ± 0.001 | 2.275 ± 0.035 | 4.262 ± 0.009 | -0.011 ± 0.009 | -0.213 ± 0.097 | 2.500 ± 0.500 |
| abl_gru | 2 | 0.069 ± 0.002 | 2.889 ± 0.054 | 5.194 ± 0.066 | -0.006 ± 0.012 | -1.099 ± 0.044 | 7.000 ± 1.000 |
| b4_gru | 2 | 0.073 ± 0.001 | 3.073 ± 0.031 | 5.392 ± 0.040 | -0.034 ± 0.012 | -0.539 ± 0.089 | 8.000 ± 0.000 |

| variant | guideline ID | adapter input |
|---|---|---|
| abl_motion | B1m (control) | executed motion u_{t-1} only (incl. speed) |
| abl_frame | B1 | DINOv3-S of the current frame + u_{t-1} |
| abl_stack | B2 | last 4 (frame, motion) pairs |
| abl_gru | B3 | GRU over the episode, trained by the flow loss |
| b4_gru | B4 | GRU trained with AIS losses on CATK rollouts (M2), frozen |

## Reading

- **The gain is the executed motion.** Motion alone takes held-out flow loss
  from 0.086 to 0.041, about 90% of the best reduction. Adding the current
  DINOv3 frame gets to 0.037 (better ADE, worse FDE). VaVAM-B sees neither its
  speed nor its past motion (only images and the command), so the adapter
  mainly supplies speed.
- **No memory benefit.** The 4-step stack equals frame. Both GRUs are worse
  within 8 epochs: 0.069 trained end-to-end, 0.073 for the frozen
  AIS-trained memory. The AIS objective does not help the action head (B4 ≥
  B3).
- **Caveat: budget.** Both GRUs were still improving at epoch 8, while
  frame/stack/motion peaked at epochs 2–7. A 16-epoch GRU run checks this
  (`docs/PROGRESS.md`).
- **Closed loop:** the frame and GRU adapters give no score gain over B0b\*
  (`reports/m3_closed_loop.md`). These open-loop differences do not
  translate to driving on val_cl.
