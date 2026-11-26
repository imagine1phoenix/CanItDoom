# Training Run Log (Vaibhav)

Track one row per training or benchmark run per `CONTRIBUTING.md`:
start time, machine, git commit, map set, config, best held-out score, checkpoint name.

| Start Time (IST) | Machine | Git Commit | Map Set | Config / Preset | Measured SPS | Best Held-Out Score | Checkpoint | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-11-26 14:54 | LOQ (16 logical CPUs, Windows 11) | `7a8e145` | `maps/two_rooms.wad:MAP01` | `benchmark_sps` (mp x16, render=False) | 9234.9 | — | — | 9h est ~299.2M steps (single=2698.7 SPS) |
| 2026-11-26 15:26 | LOQ (cpu) | `7a8e145` | `maps/two_rooms.wad:MAP01` | `smoke` (envs=1) | 1997.7 | 0.000 | `smoke/latest.ckpt` | infra smoke/training run |
| 2026-11-26 15:27 | LOQ (16 logical CPUs, Windows 11) | `7a8e145` | `freedoom2.wad:MAP01` | `benchmark_sps` (mp x16, render=False) | 755.6 | — | — | 9h est ~24.5M steps (>=500 SPS PASS; single=113.2 SPS) |
