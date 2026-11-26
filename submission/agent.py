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
            state = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            if isinstance(state, dict) and "model_state_dict" in state:
                self.model.load_state_dict(state["model_state_dict"])
            elif isinstance(state, dict):
                self.model.load_state_dict(state)

        self.model.to(self.device)
        self.model.eval()

        self.hx: torch.Tensor | None = None

    def reset(self) -> None:
        """Called before every map. Clear recurrent state and trackers."""
        self.hx = None

    def act(self, observation: dict) -> np.ndarray:
        """Return a MultiBinary action: int array of shape (9,) with values in {0, 1}."""
        if self.hx is None:
            self.hx = self.model.initial_state(batch_size=1, device=self.device)

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
            action, _, _, _, self.hx = self.model.step(obs_tensor, self.hx, deterministic=True)

        return action.squeeze(0).cpu().numpy().astype(np.int8)
