# Judge Defense Q&A Preparation (10 Technical Questions)

**Owner:** Dia (Evaluation & Deliverables Lead)  
**Rehearsal Plan:** Coordinate one 15-minute mock defense with Prit, Surbhi, and Vaibhav before final submission (11:00–13:00 Fri). Every team member answers questions for their own lane and must be able to explain every file they touched.

---

## 1. Architecture & Policy Purity (Prit)

**Q1. How do you guarantee that no privileged engine state leaks into the deployed policy?**  
- During evaluation, `submission/agent.py` runs in an isolated process over a unix domain socket (`doom_env/remote.py`) and receives only the ego-centric `observation` dict (`rgb`, `depth`, `vitals`, `rays`, `visible_entities`).
- During training, `env.privileged_state()` is accessed exclusively inside `train/ppo.py` (`RewardShaper`) for dense reward computation. `DoomActorCritic` in `submission/model.py` takes only the contract observation tensors (`rays`, `visible_entities`, `vitals`).

**Q2. Did you use any hardcoded rule trees, auto-USE, stuck-escape logic, A*, or trigonometry aimbots?**  
- No. Every button in the 9-dimensional `MultiBinary` action vector is emitted directly by `DoomActorCritic.step()` in `submission/model.py` via independent `Bernoulli` logits.
- Static AST checks (`tools/check_rules.py`) verify there is no `atan2`/`arctan2`, no `vizdoom` or `train/` imports in `submission/`, and no forbidden helpers.

**Q3. Why skip RGB/depth pixels and rely on rays, visible entities, and vitals?**  
- Under the 1-CPU-thread (`torch.set_num_threads(1)`), ~40 ms/decision latency budget, and 24-hour training window, compact geometric inputs (`64` rays, up to `32` visible entities, `17` vitals) give direct distance/bearing/type signal without spending compute on a vision CNN.
- That budget goes into our `128`-d Conv1d `RayEncoder`, `64`-d masked mean+max `EntityEncoder`, `64`-d `VitalsEncoder`, `LayerNorm` fusion, and `256`-unit `GRU`.

---

## 2. Spatial Memory & State Isolation (Prit)

**Q4. How does the agent remember unseen rooms/objects without global coordinates or `atan2`?**  
- Our V0 agent integrates temporal context across decisions inside its `256`-unit `GRU` hidden state (`self.hx`), driven purely by ego-centric rays, visible entities, and ego-frame velocities (`vel_forward`, `vel_right` in `vitals`).
- *(If V1 is merged before G2:)* V1 adds a local ego-centric occupancy map rotated/shifted with $\sin/\cos$ odometry plus forward/sideways landmark slots for doors and items, avoiding `atan2` and avoiding any classical path planner (A*/Dijkstra).

**Q5. How do you ensure state from one map never leaks into the next map?**  
- `DoomAgent.reset()` is called before every map and explicitly resets `self.hx = None` (so the next `act()` initializes a fresh zero GRU hidden state via `model.initial_state(batch_size=1)`—plus clearing any V1 odometry/landmark buffers if present).
- All observation tensors and hidden buffers have fixed sizes (`64` rays, `32` entities, `17` vitals, `256`-d GRU state), so long episodes cannot grow RAM or VRAM.

---

## 3. Training, Reward Shaping & Curriculum (Prit & Surbhi)

**Q6. Why did you write your own recurrent PPO instead of using Stable-Baselines3 or Sample Factory?**  
- We wanted full control over `evaluate_sequence()` with `done_mask_seq` hidden-state resets, our multi-branch `DoomActorCritic`, and privileged `RewardShaper` without pulling in unapproved wrapper abstractions.
- Keeping all training code self-contained in `train/ppo.py` makes every gradient step and reward term inspectable by the judges.

**Q7. How did you teach long-horizon keycard and door behavior without overfitting to evaluation maps?**  
- Surbhi built a 5-stage UDMF curriculum using Prit's procedural generator (`train/mapgen.py`), progressing from open rooms (Stage 1) -> combat (Stage 2) -> multi-room doors (Stage 3) -> keycard/locked-door chains (Stage 4) -> mixed + Freedoom 2 training maps (Stage 5).
- 6 Freedoom 2 maps were strictly held out from training (`train/maps.json`) and never used for gradient updates.

**Q8. How did you handle DOOM's edge-triggered `USE` rule on doors?**  
- Under vanilla DOOM rules (`docs/CONTRACT.md` §3), `USE` triggers only on the rising edge (`0 -> 1`), and pressing `USE` on a moving door reverses it.
- Instead of scripting a cooldown in `submission/`, the recurrent policy learns from door/key rewards in training when to press `USE` cleanly and wait for the sector to open.

---

## 4. Infrastructure, Validation & Reproducibility (Vaibhav & Dia)

**Q9. How did you verify the agent meets the strict hardware, size, and single-thread latency limits?**  
- Vaibhav runs `python tools/check_rules.py` and `OMP_NUM_THREADS=1 python validate.py` on a fresh clone with `git lfs pull`, checking repo size ($\le 1\text{ GB}$), VRAM ($\le 2\text{ GB}$ team cap), init/reset timeouts, and per-step latency ($\approx 40\text{ ms}$ average budget).
- All weights are tracked via Git LFS and training runs strictly on Google Colab (never Kaggle) + local machines.

**Q10. How did you evaluate checkpoints objectively without reward-shaping bias?**  
- Dia's offline harness (`train/evaluate.py`) evaluates checkpoints on held-out maps with **zero** reward shaping, logging the competition's true unshaped metrics from `env.privileged_state()` (`exit_reached`, `keys_found`, `doors_opened`, `unique_cells_visited`, `kills`, `damage_taken`, `deaths`, `agent_time_ms`).
- We also benchmarked greedy vs. temperature-sampled action selection (`--sweep-temperatures`) to select the highest-performing inference setting for `submission/agent.py`.
