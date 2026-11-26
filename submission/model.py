"""Neural network architecture for DOOM agent.

V0 Architecture:
- Multi-branch encoder:
  1. RayEncoder: 64 circular rays (distances + 6 hit types) -> Conv1d + MLP (128-d)
  2. EntityEncoder: Up to 32 visible entities (5 features + 12 types) with masked pooling -> MLP (64-d)
  3. VitalsEncoder: 17 scalar features (health, armor, ammo, keys, ego-velocity) -> MLP (64-d)
- Recurrent Core: 256-d GRU trunk
- Policy Head: MultiBinary(9) action logits
- Value Head: Scalar state-value critic V(s)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Bernoulli


class RayEncoder(nn.Module):
    """Encodes 64 ego-centric raycasts (distances and hit types)."""

    def __init__(self, num_rays: int = 64, num_hit_types: int = 6, out_dim: int = 128):
        super().__init__()
        self.num_rays = num_rays
        self.hit_type_emb = nn.Embedding(num_hit_types, 16)
        
        # Input channels: 1 (distance) + 16 (hit type embedding) = 17
        self.conv = nn.Sequential(
            nn.Conv1d(17, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Flatten(),
        )
        # 64 rays / 4 downsampling = 16 positions * 64 channels = 1024
        self.proj = nn.Sequential(
            nn.Linear(64 * (num_rays // 4), out_dim),
            nn.ReLU(),
        )

    def forward(self, distance: torch.Tensor, hit_type: torch.Tensor) -> torch.Tensor:
        # distance: (..., 64), hit_type: (..., 64)
        orig_shape = distance.shape[:-1]
        dist = distance.view(-1, self.num_rays, 1)
        ht = hit_type.long().view(-1, self.num_rays)
        
        emb = self.hit_type_emb(ht)  # (B, 64, 16)
        x = torch.cat([dist, emb], dim=-1)  # (B, 64, 17)
        x = x.transpose(1, 2)  # (B, 17, 64) for Conv1d
        
        feat = self.conv(x)
        out = self.proj(feat)
        return out.view(*orig_shape, -1)


class EntityEncoder(nn.Module):
    """Encodes visible entities in line-of-sight using permutation-invariant masked pooling."""

    def __init__(self, max_entities: int = 32, num_types: int = 12, out_dim: int = 64):
        super().__init__()
        self.max_entities = max_entities
        self.type_emb = nn.Embedding(num_types, 16)
        
        # 5 continuous features + 16-d type embedding = 21
        self.fc = nn.Sequential(
            nn.Linear(5 + 16, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.out_proj = nn.Sequential(
            nn.Linear(128, out_dim),  # concatenation of mean-pool and max-pool
            nn.ReLU(),
        )

    def forward(self, features: torch.Tensor, entity_type: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # features: (..., 32, 5), entity_type: (..., 32), mask: (..., 32)
        orig_shape = features.shape[:-2]
        feat = features.view(-1, self.max_entities, 5)
        etype = entity_type.long().view(-1, self.max_entities)
        m = mask.float().view(-1, self.max_entities, 1)
        
        emb = self.type_emb(etype)
        x = torch.cat([feat, emb], dim=-1)
        h = self.fc(x)  # (B, 32, 64)
        h = h * m       # zero-out invalid slots
        
        # Masked mean pooling
        counts = torch.clamp(m.sum(dim=1), min=1.0)
        mean_pool = h.sum(dim=1) / counts  # (B, 64)
        
        # Masked max pooling (replace masked elements with large negative value)
        large_neg = -1e9 * (1.0 - m)
        max_pool = (h + large_neg).max(dim=1)[0]
        max_pool = torch.where(counts > 0, max_pool, torch.zeros_like(max_pool))  # (B, 64)
        
        pooled = torch.cat([mean_pool, max_pool], dim=-1)  # (B, 128)
        out = self.out_proj(pooled)
        return out.view(*orig_shape, -1)


class VitalsEncoder(nn.Module):
    """Encodes health, armor, ammo, keys, and ego-centric velocities."""

    def __init__(self, in_features: int = 17, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim),
            nn.ReLU(),
        )

    def forward(self, vitals: torch.Tensor) -> torch.Tensor:
        # Normalize continuous values for stable gradients
        # vitals: [health, armor, weapon, ammo_curr, bullets, shells, rockets, cells, 6 keys, 3 vels]
        v = vitals.clone()
        v[..., 0:2] = v[..., 0:2] / 100.0      # health, armor
        v[..., 2] = v[..., 2] / 7.0            # weapon id
        v[..., 3] = v[..., 3] / 50.0           # current weapon ammo
        v[..., 4:8] = v[..., 4:8] / 100.0      # inventory ammos
        # keys [8:14] are binary (0/1)
        v[..., 14:17] = v[..., 14:17] / 10.0   # velocities
        return self.net(v)


class DoomActorCritic(nn.Module):
    """Recurrent Actor-Critic network for DOOM navigation and combat."""

    def __init__(self, num_actions: int = 9, hidden_dim: int = 256):
        super().__init__()
        self.num_actions = num_actions
        self.hidden_dim = hidden_dim

        self.ray_encoder = RayEncoder(out_dim=128)
        self.entity_encoder = EntityEncoder(out_dim=64)
        self.vitals_encoder = VitalsEncoder(out_dim=64)

        # Total fused feature dim: 128 + 64 + 64 = 256
        self.fusion = nn.Sequential(
            nn.Linear(256, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

        self.actor_head = nn.Linear(hidden_dim, num_actions)
        self.critic_head = nn.Linear(hidden_dim, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        """Orthogonal weight initialization for stable PPO training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.414)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.orthogonal_(m.weight, gain=1.414)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Actor head: smaller gain for near-uniform initial exploration
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
        nn.init.zeros_(self.actor_head.bias)

        # Critic head: standard gain 1.0
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)
        nn.init.zeros_(self.critic_head.bias)

    def extract_features(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract and fuse spatial and scalar observations."""
        ray_feat = self.ray_encoder(obs["rays"]["distance"], obs["rays"]["hit_type"])
        ent_feat = self.entity_encoder(
            obs["visible_entities"]["features"],
            obs["visible_entities"]["entity_type"],
            obs["visible_entities"]["mask"],
        )
        vit_feat = self.vitals_encoder(obs["vitals"])
        
        fused = torch.cat([ray_feat, ent_feat, vit_feat], dim=-1)
        return self.fusion(fused)

    def initial_state(self, batch_size: int = 1, device: torch.device | str = "cpu") -> torch.Tensor:
        """Returns zero hidden state for the GRU."""
        return torch.zeros(1, batch_size, self.hidden_dim, device=device)

    def step(
        self,
        obs: dict[str, torch.Tensor],
        hx: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single decision step for agent acting.
        
        Returns:
            action: (B, 9) binary action tensor
            log_prob: (B,) log probability of the sampled action
            entropy: (B,) entropy of the action distribution
            value: (B,) critic state-value estimate
            next_hx: (1, B, hidden_dim) updated recurrent state
        """
        # Ensure batch dimension
        features = self.extract_features(obs)  # (B, hidden_dim)
        features = features.unsqueeze(1)       # (B, 1, hidden_dim) for GRU

        out, next_hx = self.gru(features, hx)
        out = out.squeeze(1)  # (B, hidden_dim)

        logits = self.actor_head(out)
        value = self.critic_head(out).squeeze(-1)

        dist = Bernoulli(logits=logits)
        if deterministic:
            action = (logits > 0.0).float()
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        return action, log_prob, entropy, value, next_hx

    def evaluate_sequence(
        self,
        obs_seq: dict[str, torch.Tensor],
        init_hx: torch.Tensor,
        actions_seq: torch.Tensor,
        done_mask_seq: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sequence forward pass for PPO training updates.
        
        Args:
            obs_seq: dict of batched sequence observations (B, T, ...)
            init_hx: initial hidden state (1, B, hidden_dim)
            actions_seq: (B, T, 9) actions taken
            done_mask_seq: (B, T) 0.0 when episode done, 1.0 otherwise (optional for sequence resets)
            
        Returns:
            log_probs: (B, T)
            entropy: (B, T)
            values: (B, T)
        """
        features = self.extract_features(obs_seq)  # (B, T, hidden_dim)
        
        if done_mask_seq is not None:
            # Step-by-step through sequence with done mask resets
            B, T, _ = features.shape
            outputs = []
            hx = init_hx
            for t in range(T):
                step_feat = features[:, t : t + 1, :]
                if t > 0:
                    hx = hx * done_mask_seq[:, t - 1].view(1, B, 1)
                out, hx = self.gru(step_feat, hx)
                outputs.append(out)
            gru_out = torch.cat(outputs, dim=1)
        else:
            gru_out, _ = self.gru(features, init_hx)

        logits = self.actor_head(gru_out)          # (B, T, 9)
        values = self.critic_head(gru_out).squeeze(-1)  # (B, T)

        dist = Bernoulli(logits=logits)
        log_probs = dist.log_prob(actions_seq).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        return log_probs, entropy, values
