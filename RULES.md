# RULES

Read this before your first commit. If you are unsure whether something is allowed, **stop and ask Prit**.

Order of authority: the organizers' official statement and the template (`docs/CONTRACT.md`, `env/doom_env/limits.py`) come first. If this file disagrees with them, they win. Tell Prit so we can fix this file.

## The 10 rules to memorize

1. The agent in `submission/` uses **only the observation it is given**.
2. `submission/` never touches the game engine, network, host machine or evaluation maps.
3. The **trained network chooses every action**. No hand-written behaviour.
4. Engine tricks and true game state are for **training only** (`train/`).
5. The final agent is judged on the **unmodified engine with vanilla rules**.
6. Our core model and training code are **written during this hackathon**.
7. The repo is public, comes from the official starter template, and stays **1 GB or less including Git LFS**.
8. Model weights go through **Git LFS**.
9. **No Dockerfile submitted.** No new package without Prit's approval.
10. Nothing reaches `main` unless `python tools/check_rules.py` and `python validate.py` both pass.

## 1. Disqualification (organizer fair-play rule)

| Never do this | Examples |
| --- | --- |
| Access the game engine from the agent | `import vizdoom` in `submission/`, reaching into wrapper internals such as private attributes of the environment |
| Access evaluation maps | Searching the file system for maps, reading WAD files from the agent |
| Access the network | `requests`, `urllib`, `socket`, downloading weights when the agent starts |
| Access the host | `subprocess`, `os.system`, reading environment variables or files outside our repo |
| Deliberately exhaust shared resources | Grabbing GPU memory we do not need, filling the disk, spawning extra processes or threads to use more CPU |

## 2. How the agent may make decisions

The statement requires decisions from trained neural parameters. It calls hardcoded rule trees, classical pathfinders and trigonometry aimbots "discouraged". **Our team treats all of them as banned.**

| Not allowed in `submission/` | Allowed |
| --- | --- |
| `if enemy_ahead: shoot` | The policy network outputs the shoot action |
| Auto-pressing USE when facing a door | The policy learns when to press USE |
| "If stuck for N steps, turn around" | A stuck signal given to the network as an input (ask Prit first) |
| Overriding or filtering the network's action with hand-written logic | Choosing greedily or by sampling from the network's output |
| A*, Dijkstra or any path planner | A learned memory the network reads |
| `atan2` or other trigonometry to aim | Sine and cosine **only** to update our position estimate and rotate the memory map (approved, see `docs/ORGANIZER_CLARIFICATIONS.md`) |

Store landmark directions as forward and sideways offsets, so the code never needs `atan2`.

## 3. Training versus submission

| Item | `train/` | `submission/` |
| --- | --- | --- |
| Engine changes, ACS scripts, infinite ammo, auto doors | Yes | **No** |
| True position and game variables | Reward and value network only | **No** |
| `import vizdoom` | Yes | **No** |
| Importing code from `train/` | Yes | **No** |
| Google Colab (our only cloud service, see below) | Yes | Not applicable (evaluation is offline) |
| TensorBoard, matplotlib, tqdm | Yes, after checking the base image | **No** (not needed to act) |

- **Never edit `env/` or `validate.py`.** Training changes go in our own wrappers inside `train/`. The observation must stay identical to evaluation.
- Everything the agent needs to run (model, memory, preprocessing) lives in `submission/`. `train/` imports from `submission/`, never the other way round.
- The **policy** (the part that picks actions) must never receive privileged information, even in training. Only the reward and value network may.
- The saved agent must load and run without the value network's privileged inputs.
- Before choosing a final checkpoint, evaluate it on the **unmodified** environment.

### Cloud training: Google Colab only

- **No Kaggle**: no notebooks, GPUs, datasets or tools from it.
- Do not use any other cloud or rented GPU. Laptops are fine.
- Never use extra accounts to get around Colab's limits. [Colab forbids it](https://research.google.com/colaboratory/faq.html).
- Run training in notebook cells. On free runtimes, Colab can terminate SSH shells, remote desktops, separate web interfaces and distributed workers.
- Do not run scripts that fake activity to avoid disconnects (team rule).

## 4. Original work and libraries

- Create the repo from the official starter template during the hackathon.
- Do not copy agent, model or training code from other repositories, earlier projects or blog posts.
- Do not use weights trained before the event or by someone else (team rule).
- Open-source libraries are permitted. We stick to: vizdoom 1.3.3, gymnasium, torch, numpy, scipy, opencv, the Python standard library and the approved training utilities.
- Not in our plan, so ask Prit first: Stable-Baselines3, Sample Factory, EnvPool, Omgifol, Obsidian, or anything else.
- AI coding assistants are approved. **You are responsible for what you commit.** Read `AGENTS.md`, and be able to explain your code to the judges.
- Keep all training code in the repo. Reviewers and judges read it.
- If you add a community map, note its source and licence in `train/MAP_SOURCES.md`.

## 5. Agent interface

File: `submission/agent.py`, stateful class `DoomAgent`.

| Method | Must do |
| --- | --- |
| `__init__(self, observation_space, action_space, device)` | Load weights **once** |
| `reset(self)` | Clear **all** memory before every map: recurrent state, memory map, landmark slots, last action, counters |
| `act(self, observation) -> action` | Return one valid action (one decision every 4 game ticks) |

The observation format is fixed by the official wrapper. It has no global map coordinates.

## 6. Hardware and time limits

Check `env/doom_env/limits.py` for the exact values.

| Limit | Value |
| --- | --- |
| Start-up (load weights) | 60 s |
| `reset()` | 10 s |
| First `act()` of each map | 10 s |
| Every other `act()` | 5 s |
| Average compute per decision, per map | about 40 ms |
| CPU | 1 thread |
| GPU memory | 2 GB (RTX 3090) |
| RAM | 8 GB |
| Network | None |
| Repository mount | Read-only |
| Python | 3.13 |

A crash, timeout or invalid action makes the player **stand still for the rest of that map**. Points already earned stay, and penalties such as dying still apply.

So: never write files from the agent, call `torch.set_num_threads(1)`, load weights with `map_location=device`, and keep memory buffers a fixed size.

## 7. Repository

- Total size **1 GB or less, including Git LFS objects**. Old versions of LFS files also take up space, so commit only final or milestone weights.
- Do not commit videos, logs, datasets, replay files or every checkpoint.
- Never commit passwords, API keys or tokens (including GitHub tokens). The repo is public.
- Do not submit a Dockerfile of our own. Leave the template's base image files as they are.
- `submission/requirements.txt` is only for approved extra packages, with pinned versions (`name==x.y.z`).

## If you think a rule was broken

1. Do not push anything else.
2. Tell Prit immediately, with the file and line.
3. Fix it in a new commit and run `SUBMISSION_CHECKLIST.md` again.

Reporting your own mistake early is always the right move.
