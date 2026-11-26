# Failure Audit 1: Curriculum & Behavioral Analysis (Task S.5)

**Auditor:** Surbhi (Maps & Curriculum Lead)  
**Date:** Thu 26 Nov 2026 (21:00 IST Target)  
**Evaluated Artifacts:** Checkpoint evaluation replays in `replays/` across Stages 2–4 (`s2_seed050.wad`, `s3_seed050.wad`, `s4_seed050.wad`) and 26 held-out maps (`held_out_summary.json`).

---

## 1. Key Behavioral Observations

### Issue 1: Greedy Policy Corner & Wall-Sticking Loops
- **Observation:** In greedy inference mode, the agent frequently rotates into a wall or acute corner and loops indefinitely in a forward/turning cycle without making forward progress (mean unique cells = 18.0).
- **Root Cause:** Symmetrical ray observations in corners produce near-constant GRU hidden state updates. Without exploration noise, the greedy argmax chooses the same wall-colliding action sequence.
- **Recommendation for Prit (ML Lead):** Set inference temperature to $T=0.2$ (validated by Dia in Task D.4 to double exploration to 37.2 cells) or add an exploration curiosity bonus during training.

### Issue 2: Door Interaction & Alignment Deficiencies
- **Observation:** In Stage 3 multi-room maps (`s3_seed050.wad`), the agent navigates adjacent to standard manual doors (linedef type 1/26) but fails to reliably open them (mean doors opened = 0.00).
- **Root Cause:** Standard Doom doors require the agent to be within 64 map units, facing within $\pm 45^\circ$ of the linedef normal, and pressing `USE`. In wide corridor intersections, the agent often clips past the door at an oblique angle.
- **Recommendation for Curriculum:** For Stage 3 procedural map generation (`train/mapgen.py`), narrow corridors leading into doors to $96\text{ units}$ wide, forcing the agent to face the door linedef directly as it navigates.

### Issue 3: Combat Disengagement & High Monster Mortality
- **Observation:** In Stage 2 combat arenas (`s2_seed050.wad`), the agent suffers 60.0% death rates and takes an average of 78.6 damage per episode.
- **Root Cause:** The agent shoots continuously when enemies are visible but lacks defensive strafing/backpedaling to dodge projectile attacks (e.g. Imp fireballs).
- **Recommendation for Prit (ML Lead):** 
  - Ensure reward shaping incorporates health preservation ($-0.01\text{/HP}$ damage penalty is already in place, but consider $+0.2$ for dodging projectiles).
  - Verify `EntityEncoder` masked pooling fix so zero-entity observations do not pollute the recurrent trunk.

### Issue 4: Keycard Acquisition vs. Exit Progression Chaining
- **Observation:** In Stage 4 progression maps (`s4_seed050.wad`), the agent achieves 100% keycard collection (mean keys = 1.00), demonstrating that ray/entity guidance for items works well. However, exit rate remains 0% because the agent does not return to unlock the matching door.
- **Root Cause:** Finding the keycard changes inventory state, but the agent lacks persistent spatial memory to remember where the locked door was located earlier in the trajectory.
- **Recommendation for Prit (ML Lead):** Prioritize V1 architecture (ego-centric relative odometry memory grid & landmark slots) so the agent maintains a persistent spatial representation of locked doors visited before picking up the keycard.

---

## 2. Curriculum Dataset Status

- **Catalog Total:** 253 maps registered in `train/maps.json`.
- **Procedural Maps:** 220 WADs generated (55 per stage across Stages 1–4).
- **Freedoom 2 Reference:** 6 held-out maps strictly excluded from training; 26 training maps.
- **Integrity Status:** Verified with `train/check_maps.py`.
