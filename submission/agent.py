"""Evaluation entry point. The scorer instantiates DoomAgent from this file.

Contract: docs/CONTRACT.md §6.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from submission.model import DoomActorCritic

SUBMISSION_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = SUBMISSION_DIR / "weights"


class DoomAgent:
    def __init__(self, observation_space, action_space, device: str):
        """Called once. Load weights here onto `device` ('cuda' or 'cpu')."""
        # Rule compliance: 1 CPU thread limit
        torch.set_num_threads(1)

        self.observation_space = observation_space
        self.action_space = action_space
        self.device = torch.device(device)

        self.model = DoomActorCritic(num_actions=action_space.shape[0])

        # Attempt loading checkpoint if present
        checkpoint_path = WEIGHTS_DIR / "model.pt"
        if checkpoint_path.is_file():
            is_lfs_pointer = False
            try:
                with open(checkpoint_path, "rb") as f:
                    header = f.read(100)
                    if header.startswith(b"version https://git-lfs"):
                        is_lfs_pointer = True
            except OSError:
                pass

            if not is_lfs_pointer:
                try:
                    state = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
                except Exception:
                    state = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

                sd = state.get("model_state_dict", state) if isinstance(state, dict) else None
                if isinstance(sd, dict):
                    curr = self.model.state_dict()
                    compatible = {k: v for k, v in sd.items() if k in curr and curr[k].shape == v.shape}
                    if len(compatible) == len(curr):
                        self.model.load_state_dict(compatible)
                    else:
                        print(f"[DoomAgent] Partial weight match ({len(compatible)}/{len(curr)} tensors); using compatible subset.")
                        self.model.load_state_dict(compatible, strict=False)

        self.model.to(self.device)
        self.model.eval()

        self.hx: torch.Tensor | None = None
        self.mem_state = None
        self.prev_action: torch.Tensor | None = None
        self.deterministic: bool = False
        self.temperature: float = 0.2

    def set_eval_mode(self, deterministic: bool = False, temperature: float = 0.2) -> None:
        """Configures inference mode (stochastic sampling or greedy deterministic)."""
        self.deterministic = bool(deterministic)
        self.temperature = max(1e-4, float(temperature))

    def reset(self) -> None:
        """Called before every map. Clear recurrent state, memory map, landmark slots, and last action."""
        self.hx = None
        self.mem_state = None
        self.prev_action = None

    def act(self, observation: dict) -> np.ndarray:
        """Return a MultiBinary action: int array of shape (9,) with values in {0, 1}."""
        if self.hx is None:
            self.hx = self.model.initial_state(batch_size=1, device=self.device)
        if self.mem_state is None:
            self.mem_state = self.model.initial_memory(batch_size=1, device=self.device)

        # Prepare ego-centric observations as batch-1 tensors on the target device
        obs_tensor = {
            "rays": {
                "distance": torch.from_numpy(observation["rays"]["distance"]).unsqueeze(0).to(self.device),
                "hit_type": torch.from_numpy(observation["rays"]["hit_type"]).unsqueeze(0).to(self.device),
            },
            "visible_entities": {
                "features": torch.from_numpy(observation["visible_entities"]["features"]).unsqueeze(0).to(self.device),
                "entity_type": torch.from_numpy(observation["visible_entities"]["entity_type"]).unsqueeze(0).to(self.device),
                "mask": torch.from_numpy(observation["visible_entities"]["mask"]).unsqueeze(0).to(self.device),
            },
            "vitals": torch.from_numpy(observation["vitals"]).unsqueeze(0).to(self.device),
        }

        with torch.no_grad():
            action, _, _, _, self.hx, self.mem_state = self.model.step_with_memory(
                obs_tensor,
                self.hx,
                self.mem_state,
                prev_action=self.prev_action,
                deterministic=self.deterministic,
            )
            self.prev_action = action

        return action.squeeze(0).cpu().numpy().astype(np.int8)
