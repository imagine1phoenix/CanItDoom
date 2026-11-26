# Team Tasks & Coordination Index

This directory contains individual task specifications, file ownership, acceptance criteria, and AI agent execution commands for each member of the DOOM Hackathon team (Yugaantar 2026 · DevForge Track 1: "Can It Play DOOM?").

---

## Member Task Files

| Member | Role | Task File | Primary File Ownership |
| :--- | :--- | :--- | :--- |
| **Prit** | ML Lead | [tasks/PRIT.md](file:///Users/pritthacker/CanItDoom/tasks/PRIT.md) | `submission/model.py`, `submission/agent.py`, `train/ppo.py`, `train/mapgen.py`, `RULES.md`, `AGENTS.md`, `tools/` |
| **Surbhi** | Maps & Curriculum | [tasks/SURBHI.md](file:///Users/pritthacker/CanItDoom/tasks/SURBHI.md) | `train/maps.json`, `train/check_maps.py`, `train/MAP_SOURCES.md`, `maps/` |
| **Vaibhav** | Infrastructure & Training | [tasks/VAIBHAV.md](file:///Users/pritthacker/CanItDoom/tasks/VAIBHAV.md) | `train/run.py`, `.gitattributes`, `.gitignore`, run logs & checkpoint configs |
| **Dia** | Evaluation & Deliverables | [tasks/DIA.md](file:///Users/pritthacker/CanItDoom/tasks/DIA.md) | `train/evaluate.py`, `docs/ORGANIZER_CLARIFICATIONS.md`, `docs/` briefs, video/PDF assets |

---

## Universal Rules for AI Agents Working on Any Task

1. **Strict File Ownership:** Only edit files assigned to your persona. Never touch `env/`, `validate.py`, `Dockerfile`, `docker/requirements.lock`, or `docs/CONTRACT.md`.
2. **Pre-Push Validation:** Never say a task is complete without running both compliance scripts:
   ```bash
   python3 tools/check_rules.py
   uv run python validate.py
   ```
3. **No Unsanctioned Logic in `submission/`:**
   - No `import vizdoom` or `import train` / `import training`.
   - No rule trees, auto-use, stuck-recovery scripts, path planners ($A^*$, Dijkstra), or `atan2` aimbots.
   - All actions must be emitted by trained neural parameters.
4. **Git Discipline:**
   - Never run `git push --force`, `git reset --hard`, or `git add .`.
   - Always commit on a task branch (`git switch -c <name>/<task-slug>`).
   - Add files explicitly by name (`git add path/to/file`).

---

## Milestone Gates & Timeline (IST)

- **G0 (Thu 16:00):** Baseline agent passes `validate.py` and repository created on GitHub. *(PASSED)*
- **G1 (Thu 19:00):** PPO runs end-to-end; reward rises on test map.
- **G2 (Thu 22:00):** Architecture frozen (V1 if stable, otherwise V0); overnight Colab run started.
- **G3 (Fri 08:00):** Best checkpoint evaluated and selected on held-out maps.
- **G4 (Fri 11:00):** Fresh clone verification passes; final push to `origin/main`.
