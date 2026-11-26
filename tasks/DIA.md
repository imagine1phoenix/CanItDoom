# Task Specification: Dia (Evaluation & Deliverables Lead)

**Owner:** Dia  
**Role:** Evaluation Harness, Replays & Submission Brief Lead  
**Primary Ownership:** `train/evaluate.py`, `docs/ORGANIZER_CLARIFICATIONS.md`, `docs/` briefs, video and presentation assets

---

## 1. Responsibilities & Objectives

Dia guarantees that our agent's performance is rigorously measured on held-out maps and that all non-code competition requirements are fulfilled:
1. **Organizer Clarifications (`docs/ORGANIZER_CLARIFICATIONS.md`):**
   - Keep audit trail of official Q&A with screenshots/links (Colab approval, privileged state in critic, relative odometry $\sin$/$\cos$, AI assistant rules).
2. **Offline Evaluation Harness (`train/evaluate.py`):**
   - Independent evaluation script running saved model checkpoints on held-out maps without reward shaping.
   - Outputs a detailed CSV with columns: `map_name`, `tier`, `exit_reached`, `keys_found`, `doors_opened`, `unique_cells_visited`, `kills`, `damage_taken`, `deaths`, `agent_time_ms`.
3. **Inference Strategy Comparison:**
   - Benchmark greedy action selection vs. categorical sampling across temperatures ($T \in [0.2, 1.0]$).
   - Report optimal inference settings to Prit by 08:00 Fri.
4. **Demo Recordings & Visualizations:**
   - Record headless gameplay using OpenCV to render evaluation replays demonstrating landmark navigation, combat dodging, and door unlocking.
5. **2-Page Brief PDF Submission:**
   - Write executive 2-page brief covering:
     - Core Differentiator: Ego-centric relative odometry memory & landmark slots without raw RGB pixels.
     - Architecture: Ray, entity, and vitals encoders into recurrent GRU.
     - Training & Curriculum: Recurrent PPO, dense reward shaping, and progressive UDMF stages.
     - Empirical Results: Held-out map performance and latency benchmarks.
   - Outline ready by 00:00 Fri; populate real numbers by 10:00 Fri.
6. **Judge Defense Preparation:**
   - Curate 10 tough technical questions from competition judges and coordinate mock rehearsals with all 4 members.

---

## 2. Granular Task Checklist

- [x] **Task D.1 — Organizer Audit Log:** Fill initial clarifications table in `docs/ORGANIZER_CLARIFICATIONS.md`. *(Done)*
- [x] **Task D.2 — Evaluation Scoring Script (`train/evaluate.py`):** Build offline scorer calculating official competition metric breakdown. *(Done)*
- [x] **Task D.3 — PDF Brief Draft:** Complete full 2-page design brief outline in Markdown/LaTeX. *(Done)*
- [x] **Task D.4 — Greedy vs. Sampled Policy Study:** Run evaluation sweep comparing greedy vs. stochastic actions on Checkpoint G3. *(Done)*
- [x] **Task D.5 — Demo Video Production:** Capture 3 highlight clips using OpenCV showing combat, door navigation, and exit completion. *(Done)*
- [x] **Task D.6 — Final PDF Compilation:** Insert final empirical statistics and charts into 2-page PDF. *(Done)*

---

## 3. Step-by-Step AI Agent Commands

When an AI agent is working on Dia's tasks, follow this exact command protocol:

### Step 1: Branch Setup
```bash
git switch main
git pull
git switch -c dia/eval-and-deliverables
```

### Step 2: Run Evaluation Harness
```bash
# Test the evaluation script and verify CSV output
uv run python train/evaluate.py --checkpoint submission/weights/model.pt --output eval_results.csv

# Ensure evaluation code does not write forbidden artifacts into submission/
python3 tools/check_rules.py
```

### Step 3: Full Scorer Compliance Check
```bash
# Verify repo integrity and agent contracts
python3 tools/check_rules.py
uv run python validate.py
```

### Step 4: Staging, Committing, and Pushing
```bash
# Add evaluation script and documentation explicitly
git status
git add train/evaluate.py docs/ORGANIZER_CLARIFICATIONS.md
git commit -m "feat(eval): implement offline metric scoring pipeline and CSV export"
git push -u origin dia/eval-and-deliverables
```
*(Open PR on GitHub and coordinate with Prit)*
