# Task Specification: Vaibhav (Infrastructure & Training Lead)

**Owner:** Vaibhav  
**Role:** Infrastructure, Performance & Training Run Lead  
**Primary Ownership:** `train/run.py` configs, `.gitattributes`, `.gitignore`, run logs & checkpoint management

---

## 1. Responsibilities & Objectives

Vaibhav manages runtime infrastructure, compute allocation, model weight checkpoints, and validation pipelines:
1. **Git LFS & Repository Size Governance:**
   - Enforce Git LFS tracking for all weight formats (`*.pt`, `*.pth`, `*.safetensors`, `*.ckpt`, `*.onnx`).
   - Monitor total repo size to strictly keep it $\le 1\text{ GB}$ (including Git LFS objects).
2. **Environment & Dependency Maintenance:**
   - Keep `.venv` (Python 3.13, `vizdoom 1.3.3`, `gymnasium 1.4.0`) locked and clean via `uv`.
   - Prevent forbidden packages or Dockerfile additions.
3. **Throughput & Hardware Profiling:**
   - Measure decision steps per second (SPS) on local CPUs and Colab T4/A100 instances.
   - Target $\ge 500\text{ SPS}$ (ideal: $\ge 1000\text{ SPS}$) to guarantee $\sim 30\text{M}$ transitions overnight.
4. **Google Colab Training Orchestration:**
   - Manage the long training run (starts 22:00 Thu) on a fresh Google Colab GPU runtime.
   - Prevent idle disconnects using legitimate visible dashboard notebooks (no banned activity faking scripts).
   - Ensure checkpoint saving every 15 minutes and maintain local mirrored backups.
   - Verify checkpoint resumption logic prior to overnight execution.
5. **Watch Duty:**
   - Active monitoring from 00:00–02:30 Fri and resume backup checks at 07:30 Fri.
   - Check every 30 min: run alive? Loss values finite (no `nan`/`inf`)? Reward slope positive?
6. **Fresh-Clone Validation (Gate 4):**
   - Conduct clean checkout test in `/tmp/fresh-check` with `OMP_NUM_THREADS=1` prior to final push.

---

## 2. Granular Task Checklist

- [x] **Task V.1 — Repository & Remote Setup (G0):** Setup Git LFS, configure public repository, push initial commit. *(Done)*
- [ ] **Task V.2 — Benchmark Hardware SPS:** Write/run speed test script measuring steps per second with headless ViZDoom.
  - *Done when:* Machine selection finalized by 19:00 Thu.
- [ ] **Task V.3 — Colab Notebook Setup:** Prepare official Colab runner notebook with Python 3.13 runtime, cloning public repo, installing requirements.
  - *Done when:* Colab executes 1,000 steps and logs metrics without errors by 20:00 Thu.
- [ ] **Task V.4 — Checkpoint Resumption Test:** Save a checkpoint, kill process, reload, and verify training resumes from identical step.
  - *Done when:* Verified by 21:30 Thu.
- [ ] **Task V.5 — Launch Overnight Run (G2):** Launch main run on Colab at 22:00 Thu; set up local secondary run if needed.
  - *Done when:* Training curve live and saving snapshots every 15 min.
- [ ] **Task V.6 — Fresh-Clone Final Gate (G4):** Clone repo into empty directory, pull LFS weights, and verify `validate.py` passes under 1 thread.
  - *Done when:* Passes clean at 11:00 Fri.

---

## 3. Step-by-Step AI Agent Commands

When an AI agent is working on Vaibhav's tasks, follow this exact command protocol:

### Step 1: Branch Setup
```bash
git switch main
git pull
git switch -c vaibhav/training-infra
```

### Step 2: Verification of LFS & Tracked Weights
```bash
# Confirm Git LFS is tracking weights
git lfs ls-files --size

# Check repository size budget
python3 tools/check_rules.py
```

### Step 3: Run Validation in Constrained Single-Thread Environment
```bash
# Simulates the competition scorer environment (1 CPU thread)
OMP_NUM_THREADS=1 uv run python validate.py
```

### Step 4: Staging, Committing, and Pushing
```bash
# Commit infrastructure configs and run scripts
git status
git add train/run.py .gitattributes .gitignore
git commit -m "chore(infra): configure training run loop and LFS checkpoint rotation"
git push -u origin vaibhav/training-infra
```

### Step 5: Fresh Clone Verification Procedure
```bash
# Run before merging or declaring Gate 4 complete
git clone https://github.com/imagine1phoenix/CanItDoom.git /tmp/fresh-check
cd /tmp/fresh-check
git lfs pull
git lfs ls-files --size
OMP_NUM_THREADS=1 uv run python validate.py
git status --porcelain # Must be completely empty
cd /Users/pritthacker/CanItDoom && rm -rf /tmp/fresh-check
```
