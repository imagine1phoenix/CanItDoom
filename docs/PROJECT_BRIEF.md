# Project Brief: Neural DOOM Agent

**Competition:** Yugaantar 2026 · DevForge — Track 1: "Can It Play DOOM?"  
**Team:** ImaginePhoenix (@imaginephoenix) · Prit (ML Lead), Surbhi (Curriculum), Vaibhav (Infra), Dia (Evaluation)  
**Repository:** [https://github.com/imagine1phoenix/CanItDoom](https://github.com/imagine1phoenix/CanItDoom)

---

## 1. Executive Summary & Core Innovation

For decades, *"Can it run DOOM?"* benchmarked hardware capability. The modern question is: **can a neural network truly learn to play DOOM zero-shot across unseen, complex maps?**

### The Core Differentiator: Ego-Centric Spatial Memory Without Raw Pixels
Rather than consuming heavy RGB image matrices, our agent operates entirely on **high-frequency ego-centric spatial sensors** provided by the official contract:
- **64 Circular Raycasts:** Continuous distances and semantic surface types (`wall`, `door`, `locked_door`, `enemy`, `item`).
- **32 Visible Entities:** Line-of-sight relative polar vectors, approach velocities, and entity categories.
- **17-D Vitals & Odometry:** Health, armor, inventory ammos, keycard flags, and instantaneous local velocities ($v_{\text{forward}}, v_{\text{right}}, v_{\text{up}}$).

By processing spatial vectors directly, the model achieves **0.35 ms per decision** on a single CPU thread (100× faster than the 40 ms evaluation budget), allowing millions of training transitions in hours while maintaining deep spatial awareness.

---

## 2. Neural Architecture

```
                  ┌───────────────────────────────────────────────┐
                  │                 OBSERVATIONS                  │
                  └───────────────────────┬───────────────────────┘
                                          │
            ┌─────────────────────────────┼─────────────────────────────┐
            ▼                             ▼                             ▼
   ┌─────────────────┐           ┌─────────────────┐           ┌─────────────────┐
   │   Ray Encoder   │           │ Entity Encoder  │           │ Vitals Encoder  │
   │ 64 Rays x 17-D  │           │ 32 Slots x 21-D │           │   17 Scalars    │
   │  1D Conv + MLP  │           │  Masked Pooling │           │  2-Layer MLP    │
   └────────┬────────┘           └────────┬────────┘           └────────┬────────┘
            │ 128-d                       │ 64-d                        │ 64-d
            └───────────────────────┬─────┴─────────────────────────────┘
                                    ▼
                         ┌──────────────────────┐
                         │ Feature Fusion Trunk │  (256-d + LayerNorm)
                         └──────────┬───────────┘
                                    ▼
                         ┌──────────────────────┐
                         │   GRU Recurrent Core │  (256-d Hidden State)
                         └──────────┬───────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
       ┌────────────────────────┐      ┌────────────────────────┐
       │   Actor Policy Head    │      │   Critic Value Head    │
       │ MultiBinary(9) Logits  │      │ Scalar State Value V(s)│
       └────────────────────────┘      └────────────────────────┘
```

1. **Ray Encoder (128-d):** Embeds categorical hit types into a 16-d representation, concatenates normalized distances, and applies circular 1D convolutions with stride downsampling.
2. **Entity Encoder (64-d):** Embeds entity classes, passes feature vectors through a linear projection, applies valid-slot masking, and concatenates masked mean-pooling and max-pooling for permutation invariance.
3. **Vitals Encoder (64-d):** Normalizes player status and local velocity vectors through a 2-layer MLP.
4. **Recurrent Trunk & Heads:** Merges encoder representations into a 256-d LayerNorm feature vector feeding a 256-d GRU recurrent core. The actor head outputs independent Bernoulli logits for the 9 actions; the critic outputs scalar value $V(s)$.

---

## 3. Training Pipeline & Exploration Strategy

### Algorithm: Recurrent PPO with GAE
- **Optimization:** Recurrent Proximal Policy Optimization (PPO) with Generalized Advantage Estimation (GAE, $\lambda=0.95, \gamma=0.99$).
- **Sequence Updates:** Mini-batches sampled across continuous rollout chunks with recurrent state resets across episode boundaries.

### Dense Privileged Reward Shaping (Training Only)
Using `env.privileged_state()` exclusively in the training critic loop:
- **Level Exit:** $+10.0$
- **Keycard Pickup / Lock Opening:** $+3.0$
- **Door Activation:** $+0.5$
- **Enemy Neutralization:** $+1.0$
- **Damage Taken:** $-0.01$ per health point lost
- **Player Death:** $-3.0$
- **Decision Step Penalty:** $-0.001$ per tick (efficiency incentive)
- **Count-Based Spatial Exploration Bonus:** $+0.05$ awarded upon entering any previously unvisited $64\times64$ world grid cell.

### 5-Stage Curriculum Progression
1. **Stage 1 (Locomotion & Corridors):** Open geometry teaching basic navigation and unsticking from walls.
2. **Stage 2 (Combat & Resources):** Zombieman and Imp encounters with shotgun and ammo pickups.
3. **Stage 3 (Multi-Room Navigation):** Multiple chambers connected via standard doors.
4. **Stage 4 (Long-Horizon Progression):** Blue/Red keycard retrieval, locked doors, and exit routes.
5. **Stage 5 (Generalization):** Mixed procedural stages plus Freedoom 2 held-out maps.

---

## 4. Empirical Evaluation & Rule Compliance

### Verified Performance Benchmarks
- **Average Inference Latency:** $0.33\text{ ms}$ (p99: $0.52\text{ ms}$, max: $22.5\text{ ms}$ on first warm-up step).
- **Initialization Time:** $0.5\text{ s}$ (Budget: $60.0\text{ s}$).
- **Repository Footprint:** $\approx 2.8\text{ MB}$ (Limit: $1,000\text{ MB}$ / $1\text{ GB}$).
- **Thread Concurrency:** Strictly 1 CPU thread via `torch.set_num_threads(1)`.

### Strict Rule Compliance Guarantee
- **Pure Neural Policy:** 100% of actions are sampled or chosen greedily from policy logits. Zero `if/else` action overrides, zero stuck-detection scripts, zero $A^*$ or graph pathfinding, zero trigonometric aimbots.
- **Engine Isolation:** Zero calls to `vizdoom` or privileged state inside `submission/`.
