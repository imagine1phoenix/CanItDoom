# How we work

Read `RULES.md` first. This file tells you how to change the repo without breaking anything.

## Who edits what

Edit only your own files. If you need a change in someone else's file, ask them.

| Owner | Files |
| --- | --- |
| Prit | `submission/`, `train/ppo.py`, `train/mapgen.py`, `RULES.md`, `AGENTS.md`, `tools/` |
| Surbhi | `train/maps.json`, `train/check_maps.py`, `train/MAP_SOURCES.md`, map files |
| Vaibhav | `train/run.py` settings, `.gitattributes`, `.gitignore`, run log |
| Dia | `train/evaluate.py`, `docs/` (including the project brief) |
| Nobody | `env/`, `validate.py`, the template's base image files, `docs/CONTRACT.md` |

## First-time setup (once per laptop)

```bash
git lfs install
git clone <our-repo-url>
cd <our-repo-folder>
git lfs pull
python validate.py
```

## Making a change

```bash
git switch main
git pull
git switch -c yourname/short-task        # example: surbhi/map-list

# ... edit your files ...

python tools/check_rules.py              # must print PASSED
python validate.py                       # must pass
git status                               # read the list of changed files
git add path/to/file1 path/to/file2      # add files by name, never "git add ."
git commit -m "Short description of what changed"
git push -u origin yourname/short-task
```

Then open a pull request on GitHub and message Prit. **Prit merges.**

While Prit is asleep, nothing is merged into `main`. Keep your work on your branch.

## Commands you must not run

| Command | Why |
| --- | --- |
| `git push --force` | Can erase teammates' work |
| `git reset --hard` | Deletes your uncommitted work |
| `git add .` or `git add -A` | Can commit tokens, videos or huge files |
| `git config core.hooksPath ...` | Can stop Git LFS from uploading weights |

If Git shows a conflict or an error you do not understand, **stop** and ask Vaibhav or Prit. Do not delete the folder.

## Model weights

Set up LFS **before** the first weight file is committed (Vaibhav, once):

```bash
git lfs track "*.pt"
git add .gitattributes
git commit -m "Track model weights with Git LFS"
```

Check after committing a weight file:

```bash
git lfs ls-files --size       # the weight file must appear here
```

Commit only milestone and final weights. Keep other checkpoints outside the repo and back them up on a second device.

## Training runs

Vaibhav keeps one line per run in a shared sheet or chat: start time, machine, git commit, map set, config, best held-out score, checkpoint name.

On watch duty, every 30 minutes:

1. Is the run still alive?
2. Are the losses normal numbers (not `nan` or `inf`)?
3. What is the reward trend?

If it crashed, restart from the latest checkpoint and **change no code**. If the restart fails twice, wake Prit.

## Training on Colab

Google Colab is the only cloud service we use. Vaibhav runs it.

1. Runtime menu, then **Change runtime type**: pick a GPU and runtime version **2026.10** ([Python 3.13](https://research.google.com/colaboratory/runtime-version-faq.html)).
2. In the first cells, clone our public repo (no token needed), install `vizdoom==1.3.3`, and print `python --version`, the torch version, `nproc` and `nvidia-smi`.
3. Keep the GPU runtime disconnected while writing or debugging code. Use a laptop for that.
4. Start the long run on a fresh runtime. A free runtime [lasts at most 12 hours](https://research.google.com/colaboratory/faq.html), can end sooner, and its files are deleted with it.
5. Keep the tab open and the laptop awake and plugged in.
6. Save a checkpoint every 15 minutes and download a copy to the laptop. A file that stays only on Colab can be lost.
7. If Colab disconnects or refuses a GPU, resume from the latest downloaded checkpoint on a new runtime or a laptop.

Do not mount Google Drive unless the organizers have approved it.

## Maps

- Held-out maps are **never** used for training. Do not move a map from held-out to train.
- Record every map you did not generate in `train/MAP_SOURCES.md`.
