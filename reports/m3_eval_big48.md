# M3 confirmation on eval_big-48: frame adapter vs B0b*

48 eval_big scenes with artifacts on disk (`runs/ais/eval_big48/scenes.txt`), M1 protocol (2 rollouts, CATK, 4 s replay, route commands, 8 frames, muP base shapes). Logs and videos: `runs/ais/eval_big48/{b0b,frame}/sim/`.

| run | scenes | rollouts | aborted | score | pass rate | at-fault collision | offroad | progress | distance (m) |
|---|---|---|---|---|---|---|---|---|---|
| B0b* | 43 | 86 | 8 | 0.287 [0.181, 0.398] | 0.43 [0.291, 0.57] | 0.151 [0.0698, 0.244] | 0.43 [0.291, 0.57] | 0.43 [0.353, 0.509] | 67.1 [54.2, 79.9] |
| B1-frame | 43 | 84 | 12 | 0.199 [0.122, 0.287] | 0.326 [0.221, 0.442] | 0.221 [0.128, 0.314] | 0.453 [0.326, 0.581] | 0.429 [0.353, 0.509] | 71.6 [57.1, 86.5] |

Paired B1-frame − B0b* over 43 matched scenes:
- score: -0.0879 [-0.161, -0.023] (23 scenes differ)
- pass rate: -0.105 [-0.221, +0] (16 scenes differ)
- at-fault collision: +0.0698 [-0.0233, +0.163] (15 scenes differ)
- offroad: +0.0233 [-0.0581, +0.105] (14 scenes differ)
- progress: -0.000986 [-0.045, +0.0382] (39 scenes differ)
- distance (m): +4.5 [-2.47, +11.1] (42 scenes differ)

## Simulator aborts

| cause | B0b\* | B1-frame | kind |
|---|---|---|---|
| renderer exception on dynamic nodes (`478d6ec4`, `90a3d349`) | 4 | 4 | scene: our gsplat renderer cannot serve these scenes |
| route sanity check (`a07e81de` folds back, `bc9ffa8d` off map) | 4 | 4 | scene: broken route |
| "No usable map geometry within 100 m" (`70d535f2`, `95a8ccb7`, `fea0f66a`×2) | 0 | 4 | **policy**: the ego left the mapped area |

Counting the 4 policy-induced aborts as failures (score 0), the paired score
over 44 scenes is −0.086 [−0.156, −0.023]; excluding them it is −0.088
[−0.161, −0.023].

## Reading

- **The frame adapter makes closed-loop driving worse:** paired score −0.09
  (CI excludes 0), pass rate −0.10, at-fault collisions +0.07, and 4 rollouts
  that drive off the mapped area. Progress is unchanged.
- The val_cl null (+0.005 [−0.14, +0.13]) was underpowered; on 3× the scenes
  the effect is negative.
- Imitation of human replays with the executed-motion input (the adapter's
  main signal, `reports/m4_open_loop_ablation.md`) lowers open-loop error but
  does not produce a better closed-loop driver. This is consistent with the
  copycat/inertia risk of feeding past motion to a behaviour-cloned policy
  (human replays pair the human's speed with the human's next action).
- Four eval_big scenes are unusable for every policy (renderer or route); they
  should be excluded from future eval_big runs or fixed in the renderer.
