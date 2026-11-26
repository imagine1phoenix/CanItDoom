# Task Specification: Prit (ML Lead)

**Owner:** Prit (@imaginephoenix)  
**Role:** ML Lead & Repository Maintainer  
**Primary Ownership:** `submission/model.py`, `submission/agent.py`, `train/ppo.py`, `train/mapgen.py`, `RULES.md`, `AGENTS.md`, `tools/`

---

## 1. Responsibilities & Objectives

Prit owns all code and systems that directly decide the competition score:
1. **Model Architecture (`submission/model.py`):**
   - V0: Multi-branch observation encoder (Rays: 64 distances + types, Entities: 32 objects × 5 features, Vitals: 17 scalar features), fused into a GRU recurrent core ($\le 256$ hidden units), with actor & critic heads.
   - V1: Ego-centric relative odometry map grid ($\sin$/$\cos$ dead reckoning) + forward/lateral landmark memory slots.
2. **PPO Training Engine (`train/ppo.py`):**
   - Recurrent PPO with Generalized Advantage Estimation (GAE).
   - Sequence minibatch sampling handling hidden state packing.
   - Vectorized environment runner supporting fast headless execution (`render_buffers=False`).
3. **Reward Shaping Engine:**
   - Privileged training reward computation: exit reach (+10), keycard pick / unlock (+3), door open (+0.5), enemy kill (+1), new grid cell explored (+0.05), damage taken (−0.01/hp), death (−3), step penalty (−0.001).
4. **Procedural Map Generator (`train/mapgen.py`):**
   - Pure Python UDMF generator for Stages 1–4 (rooms, corridors, doors, keys, monsters, exits).
5. **Final Agent Interface (`submission/agent.py`):**
   - Load checkpoint once in `__init__`, clean state reset in `reset()`, policy inference in `act()` $\le 40\text{ ms}$.
6. **PR Reviewer:** Sole merger into `main`.

---

## 2. Granular Task Checklist

- [ ] **Task P.1 — Observation Encoders:** Build PyTorch modules in `submission/model.py` for `rays`, `visible_entities`, and `vitals`.
  - *Done when:* Encoders output clean embeddings and pass shape tests with batch and time dimensions.
- [ ] **Task P.2 — Recurrent Core & Actor-Critic Heads:** Implement GRU recurrent trunk + categorical/multi-binary actor head and value critic head.
  - *Done when:* Forward pass operates on `(batch, seq_len, ...)` and hidden state shape is verified.
- [ ] **Task P.3 — Vectorized PPO Algorithm:** Implement recurrent PPO in `train/ppo.py` with GAE ($\lambda=0.95, \gamma=0.99$) and orthogonal weight initialization.
  - *Done when:* Training on `maps/two_rooms.wad` yields monotonically increasing episode returns by 19:00 Thu.
- [ ] **Task P.4 — Procedural Map Generator:** Write `train/mapgen.py` capable of writing UDMF map files with variable room counts, doors, and monster spawns.
  - *Done when:* Surbhi can generate map batches with CLI options by 19:30 Thu.
- [ ] **Task P.5 — Relative Odometry & Landmark Memory (V1):** Integrate relative position tracking from ego velocities ($v_x, v_y, \text{turn}$) using $\sin$/$\cos$.
  - *Done when:* Replay trajectories match internal map representation by 22:00 Thu (fallback: freeze V0).
- [ ] **Task P.6 — Production `DoomAgent` Implementation:** Wire trained weights into `submission/agent.py`, ensure device portability (`cpu`/`cuda`), and verify average inference latency $\le 40\text{ ms}$.
  - *Done when:* `validate.py` passes with zero warnings.

---

## 3. Step-by-Step AI Agent Commands

When an AI agent is working on Prit's tasks, follow this exact command protocol:

### Step 1: Branch Setup
```bash
git switch main
git pull
git switch -c prit/ml-architecture
```

### Step 2: Implementation & Local Unit Verification
```bash
# Verify Python syntax and static AST rules
python3 tools/check_rules.py

# Test model shapes or training dry-run
uv run python -c "from submission.agent import DoomAgent; print('Agent imports cleanly')"
```

### Step 3: Full Scorer Compliance Check
```bash
# Must pass with 0 errors
python3 tools/check_rules.py
uv run python validate.py
```

### Step 4: Staging, Committing, and Pushing
```bash
# Add files explicitly by name (NEVER git add .)
git status
git add submission/model.py submission/agent.py train/ppo.py train/mapgen.py
git commit -m "feat(ml): implement recurrent PPO and observation encoders"
git push -u origin prit/ml-architecture
```
*(Open PR on GitHub and review before merging to `main`)*
