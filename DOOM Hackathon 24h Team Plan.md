# DOOM Hackathon 24h Team Plan

Nov 26, 2026 · @imaginephoenix

## Strategy

Submit a valid agent by 16:00 today, then train one compact custom model overnight on the spatial signals, with no RGB pixels. Prit owns the parts that decide the score; Surbhi, Vaibhav and Dia own tasks with a clear pass test. This plan assumes the 24 hours run from 13:00 Thu 26 Nov to 13:00 Fri 27 Nov (IST).

### Organizer answers: yes to all five (reported Thu 26 Nov)

- Cloud GPUs are approved for training. We use Google Colab only, not Kaggle or any other cloud service.
- Training rewards and the value network may use true position and game variables. The policy never sees them.
- The agent may estimate relative position from velocity and turns, using sine and cosine.
- Common ML utilities: answered yes, but the exact list is not recorded. Check the base image Dockerfile before using TensorBoard, matplotlib or tqdm.
- AI coding assistants may be used.

**Libraries:** the plan needs nothing beyond vizdoom 1.3.3, gymnasium, torch, numpy, scipy, opencv and the Python standard library, plus Git, Git LFS and GitHub for submission. We write our own PPO, parallel environment runner and map generator. Stable-Baselines3, Sample Factory, EnvPool, Omgifol and Obsidian are not used.

## Agent design

&#91;embedded content: agent network · 4 inputs, 1 recurrent core\]

The selling point is the highlighted branch: the agent builds its own local map from what it has seen and how it has moved. The map follows the player's heading. Slots remember where doors, locked doors and items were sighted, and it plans no paths. Field names, ray angles, velocity frame and action format must be checked against docs/CONTRACT.md first.

| Version | Contents | Rule |
| --- | --- | --- |
| V0 (must have) | Ray, entity and vitals encoders, GRU, our own recurrent PPO | Training by 19:00 |
| V1 (should have) | V0 plus odometry map and landmark slots | In the long run only if stable by 22:00 |
| Skipped | RGB pixels, curiosity networks, hierarchical policies | No time in 24 hours |

**Training:** parallel environments through Python multiprocessing, PPO with GAE and sequence minibatches, and exploration from a count-based bonus for newly visited cells. Model and memory code live in submission/, and train/ imports them.

| Reward event (starting values, tune after first runs) | Value |
| --- | --- |
| Reach exit | +10 |
| Pick up keycard, or unlock a locked door | +3 |
| Open a new door | +0.5 |
| Kill an enemy | +1 |
| Visit a new cell (once per map) | +0.05 |
| Damage taken, per health point | −0.01 |
| Death | −3 |
| Each decision | −0.001 |

| Curriculum stage | Maps | What it teaches |
| --- | --- | --- |
| 1 | Generated open rooms and corridors | Moving, not sticking in corners |
| 2 | Generated rooms with enemies and ammo | Combat and survival |
| 3 | Generated multi-room layouts with doors | Exploration |
| 4 | Generated keycard, locked door, exit | Long-horizon progression |
| 5 | Mix of stages 1–4 and Freedoom 2 maps | Generalization |

Move up a stage at about 60% success, and keep about 20% of episodes on earlier stages. The generator writes UDMF map text into a WAD with plain Python; ViZDoom [builds map nodes itself](https://vizdoom.farama.org/environments/creating_custom/).

## Work division

Prit takes everything that directly decides the score. The other three each own one lane that saves Prit hours and has a clear pass test.

| Person | Role | Owns | Main files (suggested) |
| --- | --- | --- | --- |
| Prit | ML lead | Model, memory, PPO, rewards, map generator, final agent | `submission/model.py`, `train/ppo.py`, `train/mapgen.py`, `submission/agent.py` |
| Surbhi | Maps and curriculum | Training and held-out maps, stage settings, watching behaviour | `train/maps.json`, `train/check_maps.py` |
| Vaibhav | Infrastructure and training runs | Repo, setup, speed tests, runs, backups, validator | `train/run.py` settings, `.gitattributes`, run log |
| Dia | Evaluation and submission | Organizer records, scoring script, videos, 2-page PDF, Q&A prep | `train/evaluate.py`, project brief |

By 14:00 all four agree three formats: the `maps.json` entry (file, map name, stage, train or held-out), the metrics CSV columns, and checkpoint file names.

### Prit: ML lead

- [ ] Read docs/CONTRACT.md and share the observation and action format. Done when: written in the repo README by 14:00.
- [ ] Build the V0 encoders, GRU, PPO and parallel environments. Done when: reward rises on one map by 19:00.
- [ ] Define the reward events and the training-only signals they need. Done when: each event is logged and fires correctly.
- [ ] Write the map generator for stages 1–4. Done when: Surbhi can create maps by changing settings, by 19:30.
- [ ] Add the odometry map and landmark slots. Done when: a replay shows the map matching the route, by 22:00.
- [ ] Write the final `DoomAgent` (load once, clear memory in `reset`, act) and choose the final checkpoint.
- [ ] Review all code the others commit.

### Surbhi: maps and curriculum

- [ ] Run the template's test map. Done when: one full episode runs on her laptop.
- [ ] List Freedoom 2 maps (bundled with ViZDoom) and hold out 6 that are never trained on. Done when: `maps.json` is committed by 16:00.
- [ ] Write a script that loads every map and runs 100 random actions. Done when: it prints pass or fail per map.
- [ ] Generate 50 or more maps per stage with Prit's generator, plus separate held-out ones. Done when: all pass the script, by 21:30.
- [ ] Watch recorded episodes and note failures (stuck in corners, ignores doors, wastes ammo). Done when: a short list goes to Prit at 21:00 and 05:30.

### Vaibhav: infrastructure and training runs

- [ ] Create the public repo from the starter template and set up Git LFS for weights. Add the team rule files in the first commit. Done when: everyone can clone, push and has read RULES.md.
- [ ] Set up Python 3.13 and vizdoom 1.3.3 on every machine. Use Linux or WSL on Windows for training, as [ViZDoom advises for heavy experiments](https://github.com/Farama-Foundation/ViZDoom).
- [ ] Push a do-nothing agent that passes `python validate.py`. Done when: passing by 16:00.
- [ ] Measure decisions per second on each laptop and on Colab (also note its CPU core count). Done when: the main training machine is chosen by 19:00.
- [ ] Start and monitor runs: main run on a fresh Colab runtime with the newest model, backup V0 run on a laptop. Save a checkpoint every 15 minutes and download a copy.
- [ ] Test resuming from a checkpoint before the overnight run. Done when: reward continues from where it stopped.

### Dia: evaluation and submission

- [ ] Record who gave each organizer answer and save proof in docs/ORGANIZER\_CLARIFICATIONS.md. Done when: complete by 14:00.
- [ ] Write the scoring script with Prit: exit, keys, doors, new areas, kills, damage and death per held-out map. Done when: one command writes a CSV by 21:00.
- [ ] Compare sampled and greedy actions during evaluation. Done when: the better setting is reported to Prit at 08:00.
- [ ] Record gameplay videos with OpenCV for the demo.
- [ ] Write the 2-page PDF (selling point, architecture, spatial memory, training and exploration). Done when: draft by 00:00, real results added by 10:00.
- [ ] Prepare 10 likely judge questions; each person answers those about their own lane.

## 24-hour timeline

The long training run starts at 22:00, and the final push is at 11:00 on Friday, leaving two hours of buffer. All times are IST.

| Time | Prit | Surbhi | Vaibhav | Dia |
| --- | --- | --- | --- | --- |
| Thu 13:00–15:00 | Contract, state format, encoders | Setup, test map | Repo, LFS, installs | Answer proof, scoring spec |
| 15:00–16:00 | PPO and environments | Map list, held-out split | Dummy agent passes validator | PDF outline |
| 16:00–19:00 | PPO smoke test, rewards | Map check script | Speed tests, pick machine | Scoring script |
| 19:00–22:00 | Generator by 19:30, then memory | Generate and verify maps | First runs, resume test | Finish scoring, test on first checkpoint |
| 22:00–00:00 | Start long run, check curves | Sleep from 22:00 | Monitor runs | PDF draft |
| Fri 00:00–02:30 | Sleep | Sleep | On watch | Sleep |
| 02:30–05:00 | Sleep | On watch | Sleep | Sleep |
| 05:00–08:00 | Fine-tune from best checkpoint | Review replays | Sleep until 07:30, then backups | Evaluate checkpoints |
| 08:00–11:00 | Final agent and checkpoint | Clean map files and notes | Validator, fresh-clone test, push | Finish PDF and videos |
| 11:00–13:00 | Buffer, Q&A practice | Buffer, Q&A practice | Buffer, Q&A practice | Submit PDF, lead practice |

**Watch duty:** every 30 minutes, check the run is alive, losses are numbers, and the reward trend. If it crashed, restart from the latest checkpoint and change no code. Wake Prit if a restart fails twice. Keep the Colab tab open: a free runtime lasts at most 12 hours and can end sooner (Colab FAQ: research.google.com/colaboratory/faq.html).

| Gate | Time | Pass test | If it fails |
| --- | --- | --- | --- |
| G0 | Thu 16:00 | Dummy agent passes `validate.py` and is pushed | All four fix setup first |
| G1 | Thu 19:00 | PPO runs end to end and reward rises | Pause memory work until PPO is fixed |
| G2 | Thu 22:00 | Final architecture frozen, long run started | Run V0 only |
| G3 | Fri 08:00 | Best checkpoint chosen on held-out maps | Take the most stable earlier one |
| G4 | Fri 11:00 | Fresh clone passes `validate.py`, push done | Submit the last passing version |

As an illustration, 1,000 decisions per second for the 9 hours from 22:00 to 07:00 gives about 32 million decisions. If Vaibhav measures under 500 per second, train stages 1–3 only and collect partial-progress points.

## Submission checklist

Vaibhav ticks these with Prit before the 11:00 push. The fuller version, with commands, is SUBMISSION\_CHECKLIST.md in the rule files.

- [ ] Public repo, created during the hackathon from the official starter template, with training code included
- [ ] Total size 1 GB or less including Git LFS; weights stored through Git LFS
- [ ] `submission/agent.py` defines `DoomAgent` with `__init__(observation_space, action_space, device)`, `reset()` and `act(observation)`
- [ ] `reset()` clears the GRU state, map and landmark slots
- [ ] No Dockerfile; `requirements.txt` only for extra packages the organizers approve, with pinned versions
- [ ] Runs on Python 3.13 with 1 CPU thread, within 2 GB VRAM and 8 GB RAM
- [ ] Within time limits: 60 s start, 10 s reset, 10 s first action, 5 s other actions, about 40 ms average per decision
- [ ] No network access, no file writes into the repo, and only the given observations used
- [ ] Every `act` call returns a valid action; no hardcoded rule trees, A\*, Dijkstra or atan2 aiming
- [ ] Final agent runs on the unmodified engine and vanilla rules
- [ ] 2-page PDF brief attached
- [ ] `python validate.py` passes on a fresh clone
