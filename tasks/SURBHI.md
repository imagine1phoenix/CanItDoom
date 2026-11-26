# Task Specification: Surbhi (Maps & Curriculum Lead)

**Owner:** Surbhi  
**Role:** Maps, Datasets & Curriculum Specialist  
**Primary Ownership:** `train/maps.json`, `train/check_maps.py`, `train/MAP_SOURCES.md`, `maps/`

---

## 1. Responsibilities & Objectives

Surbhi ensures the agent learns generalizable behaviors across progressively challenging environments and prevents overfitting:
1. **Map Catalog & Metadata (`train/maps.json`):**
   - Schema: `[{"file": "...", "map": "MAP01", "stage": 1, "split": "train"|"held_out"}]`.
   - Separate training environments from unseen evaluation holdouts.
2. **Freedoom 2 Held-Out Selection:**
   - Reserve 6 maps from `freedoom2.wad` that are strictly excluded from training.
   - Record author attribution and BSD-3-Clause licensing in [`train/MAP_SOURCES.md`](file:///Users/pritthacker/CanItDoom/train/MAP_SOURCES.md).
3. **Map Verification Harness (`train/check_maps.py`):**
   - Automated script to load every cataloged WAD, step 100 random actions, verify sector limits ($\le 2,048$ sectors), and report integrity.
4. **Curriculum Dataset Generation:**
   - Use Prit's procedural generator (`train/mapgen.py`) to build $\ge 50$ maps per curriculum stage:
     - Stage 1: Open rooms and corridors (navigation, unsticking).
     - Stage 2: Combat arenas with ammo and enemy variants.
     - Stage 3: Multi-room layouts with standard doors.
     - Stage 4: Keycard locked door chains to exits.
5. **Replay & Behavioral Analysis:**
   - Inspect video recordings from training checkpoints at 21:00 Thu and 05:30 Fri.
   - Report behavioral bugs (corner spinning, ignoring doors, excessive ammo waste) to Prit.
6. **Watch Duty:** Active monitoring of overnight Colab runs from 02:30–05:00 Fri.

---

## 2. Granular Task Checklist

- [ ] **Task S.1 — Baseline Map Verification:** Run `maps/two_rooms.wad` through `doom_env` and confirm zero errors.
  - *Done when:* Episode runs locally without crashes.
- [ ] **Task S.2 — Held-Out Selection & Catalog:** Select 6 Freedoom 2 maps, update `train/MAP_SOURCES.md`, and generate `train/maps.json`.
  - *Done when:* `train/maps.json` is committed by 16:00 Thu.
- [ ] **Task S.3 — Map Integrity Test Script (`train/check_maps.py`):** Build script testing all maps for compatibility and sector bounds.
  - *Done when:* Script prints tabular pass/fail summary across all registered maps.
- [ ] **Task S.4 — Batch Map Generation:** Run procedural generator to create 50+ WADs per stage for Stages 1–4.
  - *Done when:* All generated maps pass `train/check_maps.py` by 21:30 Thu.
- [ ] **Task S.5 — Failure Audit 1:** Review checkpoint replays at 21:00 Thu and deliver issue list to Prit.
- [ ] **Task S.6 — Failure Audit 2:** Review overnight training replays at 05:30 Fri and report progression rates.

---

## 3. Step-by-Step AI Agent Commands

When an AI agent is working on Surbhi's tasks, follow this exact command protocol:

### Step 1: Branch Setup
```bash
git switch main
git pull
git switch -c surbhi/curriculum-maps
```

### Step 2: Implementation & Map Checks
```bash
# Check map catalog validity and run random action smoke tests
uv run python train/check_maps.py

# Verify that no forbidden files or secrets were created
python3 tools/check_rules.py
```

### Step 3: Full Scorer Compliance Check
```bash
# Must pass before staging commits
python3 tools/check_rules.py
uv run python validate.py
```

### Step 4: Staging, Committing, and Pushing
```bash
# Stage map files and catalog by name (DO NOT use git add .)
git status
git add train/maps.json train/check_maps.py train/MAP_SOURCES.md
git commit -m "feat(curriculum): catalog held-out maps and verification harness"
git push -u origin surbhi/curriculum-maps
```
*(Open PR on GitHub and notify Prit)*
