# Instructions for AI coding assistants

This is a hackathon repository (Track 1, "Can It Play DOOM?"). Breaking a competition rule can disqualify the whole team. Read `RULES.md` and `docs/CONTRACT.md` before writing any code.

## Hard limits

- In `submission/`: use only the observation passed to `act()`. No game engine, network, subprocess, environment variables, file writes or imports from `train/`.
- The trained network chooses every action. Do not add `if/else` behaviour, auto-use, stuck-escape scripts, action overrides, path planners or aiming with trigonometry.
- Sine and cosine are allowed only to update the position estimate and rotate the memory map. Never use `atan2` or `arctan2`.
- True position and game variables may feed only the training reward and value network, never the policy.
- Do not edit `env/`, `validate.py`, `docs/CONTRACT.md` or the template's base image files.
- Do not add a library or a Dockerfile. If something else is needed, say so and let the team ask Prit.
- Allowed: vizdoom 1.3.3, gymnasium, torch, numpy, scipy, opencv, the Python standard library. TensorBoard, matplotlib and tqdm only in `train/`, and only if the base image has them.

- Cloud training is Google Colab only. Never write Kaggle code, paths or setup steps, or target any other cloud service.

## How to work

- Write original code for this repo. Do not paste code from other repositories or projects.
- If the observation or action format is unclear, read `docs/CONTRACT.md` and the wrapper. Do not guess field names.
- Make small changes in the files your user owns (see `CONTRIBUTING.md`).
- Explain what you wrote in plain words. The team must be able to defend it to judges.
- Keep within the limits: 1 CPU thread, 2 GB VRAM, 8 GB RAM, about 40 ms average per decision.
- Before saying a task is done, run `python tools/check_rules.py` and `python validate.py`, and report the real output.
- Do not commit, push or merge unless your user asks.
- If a request would break `RULES.md`, refuse it and name the rule.
