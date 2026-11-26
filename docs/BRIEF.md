# Project Brief Draft: Spatial-Signal Recurrent PPO for Partially Observable DOOM

> **Status:** Verified Held-Out Benchmark & Ablation Study (Tasks D.2--D.5). Populated with verified numbers from `train/evaluate.py` across 26 held-out evaluation maps (5 Tier 1, 5 Tier 2, 5 Tier 3, 5 Tier 4, 6 Held-Out Freedoom 2). Never insert unverified or estimated metrics.

**Track:** DevForge Track 1 — "Can It Play DOOM?" (Yugaantar 2026)  
**Team:** Prit (ML Lead), Surbhi (Maps & Curriculum), Vaibhav (Infrastructure & Training), Dia (Evaluation & Deliverables)

---

## 1. Executive Summary & Core Differentiator

Navigating multi-room DOOM levels under partial observability requires remembering doors, keycards, and cleared corridors over horizons of 500–1,050+ decisions, while fighting enemies within a strict **~40 ms/decision** budget on **1 CPU thread** (with no global coordinates or god-view).

Rather than feeding raw `160x120` RGB/depth pixels into a heavy vision backbone, our agent operates purely on the **compact ego-centric spatial signals** provided by the `doom_env` contract:
- **64 ego-centric movement/sight rays** (`distance` normalized to 1,024 map units, `hit_type` $\in$ `{none, wall, door, locked_door, enemy, item}`),
- **Up to 32 visible entities** (`distance`, $\sin(\Delta\theta)$, $\cos(\Delta\theta)$, relative forward/right velocity, and 12 coarse entity classes),
- **17 vitals scalars** (health, armor, weapon/ammo state, 6 key flags, ego-frame velocities).

**Core Differentiator (Compact Geometric Encoders + Recurrent Memory, Zero Hand-Written Rules):**  
Our policy pairs specialized geometric encoders (circular-style 1D ray convolutions, permutation-invariant masked entity pooling, and normalized vitals) with a `256`-unit GRU recurrent core trained from scratch via recurrent PPO. Every `MultiBinary(9)` action is emitted directly by the neural policy—with **zero** hardcoded rule trees, **zero** A*/Dijkstra pathfinding, and **zero** `atan2` trigonometry aiming. *(V1 target extension: ego-centric $\sin/\cos$ relative odometry grid and forward/sideways landmark slots if stabilized before G2.)*

---

## 2. Agent Architecture (`submission/model.py` & `submission/agent.py`)

1. **Multi-Branch Observation Encoders:**
   - **Ray Encoder (`RayEncoder`):** Embeds the 64 `hit_type` IDs (`16`-d), concatenates normalized `distance` (`17` channels), applies two strided `Conv1d` layers (`kernel=5, stride=2`, `32 -> 64` channels), and projects to `128`-d.
   - **Entity Encoder (`EntityEncoder`):** Embeds each of the 32 `entity_type` IDs (`16`-d), concatenates the 5 kinematic features (`21`-d), applies a 2-layer `64`-d MLP, aggregates valid slots (`mask`) via concatenated masked mean-pooling and masked max-pooling (`128`-d), and projects to `64`-d.
   - **Vitals Encoder (`VitalsEncoder`):** Scales the 17 vitals scalars (health/armor `/100`, weapon `/7`, active ammo `/50`, inventory ammo `/100`, ego velocities `/10`, binary keys) and projects through a 2-layer `64`-d MLP.
2. **Fusion & Recurrent Core (`DoomActorCritic`):**
   - Concatenates the three branches (`128 + 64 + 64 = 256`-d), passes through a `Linear(256, 256) + LayerNorm + ReLU` fusion layer, and feeds a `256`-unit single-layer `GRU`.
   - **Policy Head (Actor):** `Linear(256, 9)` (orthogonal init `gain=0.01`) outputting independent Bernoulli logits over the 9 `MultiBinary` buttons. Note: `USE` and `SELECT_NEXT_WEAPON` fire on button-down edges under vanilla DOOM rules.
   - **Value Head (Critic):** `Linear(256, 1)` (`gain=1.0`) estimating $V(s)$ from the recurrent hidden state; deployed inference uses `DoomAgent.act()` with `torch.set_num_threads(1)` and clears `hx` in `DoomAgent.reset()` before every map.
   - *(Optional V1 Spatial Memory Extension: fixed-size ego-centric relative odometry grid rotated/shifted with $\sin/\cos$ and fixed-capacity landmark slots, also reset in `DoomAgent.reset()`.)*

---

## 3. Training, Reward Shaping & Curriculum (`train/ppo.py` & `train/mapgen.py`)

- **Custom Recurrent PPO:** Built from scratch in `train/ppo.py` (no external RL framework dependencies), using Generalized Advantage Estimation (GAE, $\gamma = 0.99$, $\lambda = 0.95$), PPO clip $\epsilon = 0.2$, `rollout_len = 128`, `epochs = 4`, `lr = 3e-4` (Adam, `eps = 1e-5`, grad norm clip `0.5`), entropy bonus (`0.01`), value loss coefficient (`0.5`), and sequence evaluation preserving recurrent hidden states with episode-boundary `done` masks.
- **Privileged Reward Shaping (`RewardShaper`, Training Only):**
  - Uses `env.privileged_state()` strictly during training (never exposed to `submission/`): Exit reached (`+10`), keycard pickup (`+3`), enemy kill (`+1`), damage taken (`-0.01/HP`), death (`-3`), step penalty (`-0.001`).
  - **Exploration Bonus:** Count-based bonus (`+0.05`) for visiting each new $64\times 64$ spatial grid cell once per episode from true `(x, y)`.
- **5-Stage Progressive UDMF Curriculum (`train/mapgen.py`):**
  1. **Stage 1:** Generated open rooms & corridors (smooth movement, corner avoidance).
  2. **Stage 2:** Generated combat arenas with monsters & ammo (aiming, dodging, survival).
  3. **Stage 3:** Multi-room layouts with ordinary doors (door interaction, exploration).
  4. **Stage 4:** Keycard + locked door + exit chains (long-horizon memory & backtracking).
  5. **Stage 5:** Mix of Stages 1–4 and training Freedoom 2 maps (generalization, while keeping 6 Freedoom 2 maps strictly held out).
  - *Progression rule:* Advance at ~60% stage success while retaining ~20% episodes from earlier stages.

---

## 4. Empirical Results & Benchmarks *(Populate at G3 / G4)*

### 4.1 Held-Out Evaluation Suite (`train/evaluate.py`, unshaped metrics)

| Tier / Split | Maps | Exit Rate | Mean Keys | Mean Doors | Unique Cells | Mean Kills | Mean Damage | Death Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Tier 1 (Nav/Combat) | 5 | 0.0% | 0.00 | 0.00 | 15.6 | 0.00 | 0.0 | 0.0% |
| Tier 2 (Multi-Room) | 5 | 0.0% | 0.00 | 0.00 | 32.2 | 1.20 | 78.6 | 60.0% |
| Tier 3 (Key/Lock) | 5 | 0.0% | 0.00 | 0.00 | 34.4 | 0.60 | 68.4 | 40.0% |
| Tier 4 (Progression) | 5 | 0.0% | 1.00 | 0.00 | 41.6 | 0.00 | 69.6 | 20.0% |
| Held-Out Freedoom 2 | 6 | 0.0% | 0.00 | 0.17 | 22.7 | 1.33 | 76.5 | 50.0% |
| **Overall** | 26 | 0.0% | 0.19 | 0.04 | 29.0 | 0.65 | 59.3 | 34.6% |

### 4.2 Inference Strategy Ablation (Greedy vs. Categorical Sampling)

Evaluated across representative held-out stage environments (`summary.json`):

| Policy Mode | Exit Rate | Mean Doors Opened | Mean Unique Cells | Mean Kills | Selected? |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Greedy (`deterministic`) | 0.0% | 0.00 | 18.0 | 0.80 | Baseline |
| Sample ($T = 0.2$) | 0.0% | 0.00 | **37.2** | **0.80** | **Recommended** |
| Sample ($T = 0.4$) | 0.0% | 0.00 | 28.6 | 0.40 | Candidate |
| Sample ($T = 0.6$) | 0.0% | 0.20 | 22.2 | 0.20 | No |
| Sample ($T = 0.8$) | 0.0% | 0.00 | 21.0 | 0.60 | No |
| Sample ($T = 1.0$) | 0.0% | 0.00 | 26.8 | 0.60 | No |

*Finding:* Low-temperature sampling ($T=0.2$) doubles unique cell exploration (37.2 cells) compared to greedy execution (18.0 cells) by breaking repetitive wall-sticking cycles without causing chaotic erratic turning.

### 4.3 System & Compliance Benchmarks (`OMP_NUM_THREADS=1 python validate.py`)

| Metric | Measured Value | Official Limit | Status |
| :--- | :---: | :---: | :---: |
| Agent Initialization (`__init__`) | 0.4 s | $\le 60$ s | **PASS** |
| Per-Map Reset (`reset`) | 0.01 s | $\le 10$ s | **PASS** |
| First `act()` / Subsequent `act()` Max | 1.1 ms / 0.87 ms | $\le 10$ s / $\le 5$ s | **PASS** |
| Mean / p99 Decision Latency (1 CPU thread) | 0.18 ms / 0.78 ms | ~40 ms avg budget | **PASS** (200× faster) |
| Peak Agent GPU VRAM | $\approx 2.5$ MB weights (CPU/CUDA) | $\le 2,048$ MB (team cap) / 6 GB | **PASS** |
| Repository + Git LFS Size | 3.5 MB | $\le 1,000$ MB | **PASS** |
