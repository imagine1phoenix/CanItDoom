"""Modular neural network cluster architecture for DOOM agent.

Purely neural Mixture-of-Experts (MoE) policy architecture:
- Shared Sensory Encoders:
  1. RayEncoder: 64 circular rays (distances + 6 hit types) -> Conv1d + MLP (128-d)
  2. FilteredEntityEncoder (all): Up to 32 visible entities -> masked pooling (64-d)
  3. VitalsEncoder: 17 scalar vitals/velocities -> MLP (64-d)
  4. LandmarkOdometryEncoder: sin/cos nearest-ray geometry for key landmarks (32-d)
  5. PrevActionEncoder: previous MultiBinary(9) action embedding (16-d)
- Shared Recurrent Core: LayerNorm fusion (304 -> 256) + 256-d GRU trunk
- Specialist Neural Cluster (4 experts + learned neural router):
  0. NavExpert: corridor steering, wall clearances, ego velocity, exit-seeking
  1. EnemyExpert: hostile/projectile attention, combat vitals, aiming & dodging
  2. DoorExpert: door/locked-door front cone geometry, key state, edge-triggered USE
  3. ProgressionExpert: item/key entity attention, locked-door backtracking
  - ClusterRouter: learned softmax gating over the 4 experts (no if/else rules)
- Critic Head: Scalar state-value critic V(s)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.distributions import Bernoulli

EXPERT_NAMES = ("nav", "enemy", "door", "progression")
NUM_EXPERTS = len(EXPERT_NAMES)


def normalize_vitals(vitals: torch.Tensor) -> torch.Tensor:
    """Normalizes the 17-element vitals tensor into bounded ranges."""
    v = vitals.clone()
    v[..., 0:2] = v[..., 0:2] / 100.0     # health, armor
    v[..., 2] = v[..., 2] / 7.0           # selected_weapon
    v[..., 3] = v[..., 3] / 50.0          # selected_weapon_ammo
    v[..., 4:8] = v[..., 4:8] / 100.0     # inventory ammos
    # keys [8:14] are binary (0/1)
    v[..., 14:17] = v[..., 14:17] / 10.0  # velocities
    return v


def nearest_ray_features(
    distance: torch.Tensor,
    hit_type: torch.Tensor,
    target_types: int | tuple[int, ...],
    ray_cos: torch.Tensor,
    ray_sin: torch.Tensor,
) -> torch.Tensor:
    """Extracts [seen, min_dist, cos, sin] for the nearest ray matching target_types."""
    if isinstance(target_types, int):
        mask = hit_type == target_types
    else:
        mask = torch.zeros_like(hit_type, dtype=torch.bool)
        for t in target_types:
            mask = mask | (hit_type == t)

    has_hit = mask.any(dim=-1, keepdim=True).float()
    masked_dist = torch.where(mask, distance, torch.full_like(distance, 2.0))
    min_dist, idx = masked_dist.min(dim=-1, keepdim=True)
    min_dist = torch.where(has_hit > 0, min_dist, torch.ones_like(min_dist))
    cos_a = torch.where(has_hit > 0, ray_cos[idx], torch.zeros_like(min_dist))
    sin_a = torch.where(has_hit > 0, ray_sin[idx], torch.zeros_like(min_dist))
    return torch.cat([has_hit, min_dist, cos_a, sin_a], dim=-1)


class LandmarkOdometryEncoder(nn.Module):
    """Ego-centric landmark and odometry feature extractor (cos/sin only, no atan2)."""

    def __init__(self, num_rays: int = 64, out_dim: int = 32):
        super().__init__()
        angles = torch.arange(num_rays, dtype=torch.float32) * (2.0 * math.pi / num_rays)
        self.register_buffer("ray_cos", torch.cos(angles))
        self.register_buffer("ray_sin", torch.sin(angles))

        # 3 landmark types * 4 features (seen, dist, cos, sin) + 2 odometry/key = 14
        self.net = nn.Sequential(
            nn.Linear(14, out_dim),
            nn.ReLU(),
        )

    def forward(self, distance: torch.Tensor, hit_type: torch.Tensor, vitals: torch.Tensor) -> torch.Tensor:
        locked_door = nearest_ray_features(distance, hit_type, 3, self.ray_cos, self.ray_sin)
        std_door = nearest_ray_features(distance, hit_type, 2, self.ray_cos, self.ray_sin)
        enemy_ray = nearest_ray_features(distance, hit_type, 4, self.ray_cos, self.ray_sin)

        has_key = vitals[..., 8:14].max(dim=-1, keepdim=True)[0]
        speed = torch.sqrt(vitals[..., 14:15] ** 2 + vitals[..., 15:16] ** 2 + 1e-6) / 10.0

        feats = torch.cat([locked_door, std_door, enemy_ray, has_key, speed], dim=-1)
        return self.net(feats)


class RayEncoder(nn.Module):
    """Encodes 64 ego-centric raycasts (distances and hit types)."""

    def __init__(self, num_rays: int = 64, num_hit_types: int = 6, out_dim: int = 128):
        super().__init__()
        self.num_rays = num_rays
        self.num_hit_types = num_hit_types
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
        orig_shape = distance.shape[:-1]
        dist = distance.view(-1, self.num_rays, 1)
        ht = hit_type.long().clamp(0, self.num_hit_types - 1).view(-1, self.num_rays)

        emb = self.hit_type_emb(ht)
        x = torch.cat([dist, emb], dim=-1).transpose(1, 2)
        out = self.proj(self.conv(x))
        return out.view(*orig_shape, -1)


class FilteredEntityEncoder(nn.Module):
    """Encodes visible entities via permutation-invariant masked mean+max pooling."""

    def __init__(
        self,
        max_entities: int = 32,
        num_types: int = 12,
        out_dim: int = 64,
        emb_dim: int = 16,
        mlp_dim: int = 64,
        target_types: tuple[int, ...] | None = None,
    ):
        super().__init__()
        self.max_entities = max_entities
        self.num_types = num_types
        self.target_types = target_types
        self.type_emb = nn.Embedding(num_types, emb_dim)

        self.fc = nn.Sequential(
            nn.Linear(5 + emb_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, mlp_dim),
            nn.ReLU(),
        )
        self.out_proj = nn.Sequential(
            nn.Linear(mlp_dim * 2, out_dim),
            nn.ReLU(),
        )

    def forward(self, features: torch.Tensor, entity_type: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        orig_shape = features.shape[:-2]
        feat = features.view(-1, self.max_entities, 5)
        etype = entity_type.long().clamp(0, self.num_types - 1).view(-1, self.max_entities)
        m = mask.float().view(-1, self.max_entities)

        if self.target_types is not None:
            type_match = torch.zeros_like(m, dtype=torch.bool)
            for t in self.target_types:
                type_match = type_match | (etype == t)
            m = m * type_match.float()

        m = m.unsqueeze(-1)
        emb = self.type_emb(etype)
        h = self.fc(torch.cat([feat, emb], dim=-1)) * m

        valid_counts = m.sum(dim=1)
        has_entities = valid_counts > 0
        counts = torch.clamp(valid_counts, min=1.0)

        mean_pool = torch.where(has_entities, h.sum(dim=1) / counts, torch.zeros_like(h[:, 0, :]))
        large_neg = -1e9 * (1.0 - m)
        raw_max = (h + large_neg).max(dim=1)[0]
        max_pool = torch.where(has_entities, raw_max, torch.zeros_like(raw_max))

        out = self.out_proj(torch.cat([mean_pool, max_pool], dim=-1))
        return out.view(*orig_shape, -1)


# Backward-compatible alias
EntityEncoder = FilteredEntityEncoder


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
        return self.net(normalize_vitals(vitals))


class SpecialistExpert(nn.Module):
    """Single specialist neural policy head mapping domain features + GRU state to action logits."""

    def __init__(self, domain_dim: int, gru_dim: int, num_actions: int = 9, hidden: int = 64):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(domain_dim + gru_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden, num_actions)

    def forward(self, domain_feat: torch.Tensor, gru_out: torch.Tensor) -> torch.Tensor:
        h = self.trunk(torch.cat([domain_feat, gru_out], dim=-1))
        return self.head(h)


class ClusterRouter(nn.Module):
    """Learned neural gating network blending specialist engines end-to-end."""

    def __init__(self, feat_dim: int, gru_dim: int, num_experts: int = NUM_EXPERTS, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim + gru_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_experts),
        )

    def forward(self, fused_feat: torch.Tensor, gru_out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gate_logits = self.net(torch.cat([fused_feat, gru_out], dim=-1))
        gate_probs = torch.softmax(gate_logits, dim=-1)
        return gate_logits, gate_probs


class DoomActorCritic(nn.Module):
    """Recurrent Mixture-of-Experts Actor-Critic cluster for DOOM."""

    EXPERT_NAMES = EXPERT_NAMES

    def __init__(self, num_actions: int = 9, hidden_dim: int = 256, num_rays: int = 64):
        super().__init__()
        self.num_actions = num_actions
        self.hidden_dim = hidden_dim
        self.num_rays = num_rays
        self.last_gate_probs: torch.Tensor | None = None

        angles = torch.arange(num_rays, dtype=torch.float32) * (2.0 * math.pi / num_rays)
        self.register_buffer("ray_cos", torch.cos(angles))
        self.register_buffer("ray_sin", torch.sin(angles))

        # Shared sensory encoders
        self.ray_encoder = RayEncoder(num_rays=num_rays, out_dim=128)
        self.entity_encoder = FilteredEntityEncoder(out_dim=64)
        self.vitals_encoder = VitalsEncoder(out_dim=64)
        self.landmark_encoder = LandmarkOdometryEncoder(num_rays=num_rays, out_dim=32)
        self.prev_action_encoder = nn.Sequential(
            nn.Linear(num_actions, 16),
            nn.ReLU(),
        )

        # Total shared fused feature dim: 128 + 64 + 64 + 32 + 16 = 304
        self.fusion = nn.Sequential(
            nn.Linear(304, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

        # Domain-specific entity encoders for Enemy and Progression engines
        self.enemy_entity_encoder = FilteredEntityEncoder(
            out_dim=32,
            emb_dim=8,
            mlp_dim=32,
            target_types=(1, 2, 3, 4),
        )
        self.item_entity_encoder = FilteredEntityEncoder(
            out_dim=32,
            emb_dim=8,
            mlp_dim=32,
            target_types=(5, 6, 7, 8, 9),
        )

        # Specialist neural cluster: nav (0), enemy (1), door (2), progression (3)
        self.experts = nn.ModuleDict(
            {
                "nav": SpecialistExpert(domain_dim=24 + 3 + num_actions, gru_dim=hidden_dim, num_actions=num_actions),
                "enemy": SpecialistExpert(domain_dim=32 + 4 + 8 + num_actions, gru_dim=hidden_dim, num_actions=num_actions),
                "door": SpecialistExpert(domain_dim=4 + 4 + 18 + 8 + num_actions, gru_dim=hidden_dim, num_actions=num_actions),
                "progression": SpecialistExpert(domain_dim=32 + 4 + 4 + 9 + num_actions, gru_dim=hidden_dim, num_actions=num_actions),
            }
        )
        self.router = ClusterRouter(feat_dim=hidden_dim, gru_dim=hidden_dim, num_experts=NUM_EXPERTS)
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

        for expert in self.experts.values():
            nn.init.orthogonal_(expert.head.weight, gain=0.01)
            nn.init.zeros_(expert.head.bias)

        router_out = self.router.net[-1]
        nn.init.orthogonal_(router_out.weight, gain=0.01)
        nn.init.zeros_(router_out.bias)

        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)
        nn.init.zeros_(self.critic_head.bias)

    def _resolve_prev_action(self, obs: dict[str, torch.Tensor], prev_action: torch.Tensor | None) -> torch.Tensor:
        if prev_action is not None:
            return prev_action.float()
        vitals = obs["vitals"]
        return torch.zeros(*vitals.shape[:-1], self.num_actions, device=vitals.device, dtype=vitals.dtype)

    def extract_features(self, obs: dict[str, torch.Tensor], prev_action: torch.Tensor | None = None) -> torch.Tensor:
        """Extracts and fuses shared spatial, scalar, landmark, and previous-action observations."""
        pa = self._resolve_prev_action(obs, prev_action)
        ray_feat = self.ray_encoder(obs["rays"]["distance"], obs["rays"]["hit_type"])
        ent_feat = self.entity_encoder(
            obs["visible_entities"]["features"],
            obs["visible_entities"]["entity_type"],
            obs["visible_entities"]["mask"],
        )
        vit_feat = self.vitals_encoder(obs["vitals"])
        lm_feat = self.landmark_encoder(obs["rays"]["distance"], obs["rays"]["hit_type"], obs["vitals"])
        pa_feat = self.prev_action_encoder(pa)

        fused = torch.cat([ray_feat, ent_feat, vit_feat, lm_feat, pa_feat], dim=-1)
        return self.fusion(fused)

    def _extract_domain_inputs(
        self,
        obs: dict[str, torch.Tensor],
        prev_action: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Builds domain-specific feature vectors for each specialist neural engine."""
        dist = obs["rays"]["distance"]
        ht = obs["rays"]["hit_type"]
        norm_v = normalize_vitals(obs["vitals"])
        lead_shape = dist.shape[:-1]

        # 1. Nav domain: 8-sector min/mean clearances + wall density + ego vel + prev_action
        dist_sec = dist.view(*lead_shape, 8, 8)
        wall_sec = (ht == 1).float().view(*lead_shape, 8, 8)
        nav_in = torch.cat(
            [
                dist_sec.min(dim=-1)[0],
                dist_sec.mean(dim=-1),
                wall_sec.mean(dim=-1),
                norm_v[..., 14:17],
                prev_action,
            ],
            dim=-1,
        )

        # 2. Enemy domain: hostile/projectile entities + nearest enemy ray + combat vitals + prev_action
        enemy_ent = self.enemy_entity_encoder(
            obs["visible_entities"]["features"],
            obs["visible_entities"]["entity_type"],
            obs["visible_entities"]["mask"],
        )
        enemy_ray = nearest_ray_features(dist, ht, 4, self.ray_cos, self.ray_sin)
        enemy_in = torch.cat([enemy_ent, enemy_ray, norm_v[..., 0:8], prev_action], dim=-1)

        # 3. Door domain: std/locked door rays + 9-ray front cone + keys + speed + prev_action
        std_door = nearest_ray_features(dist, ht, 2, self.ray_cos, self.ray_sin)
        locked_door = nearest_ray_features(dist, ht, 3, self.ray_cos, self.ray_sin)
        front_dist = torch.cat([dist[..., -4:], dist[..., :5]], dim=-1)
        front_ht = torch.cat([ht[..., -4:], ht[..., :5]], dim=-1)
        front_is_door = ((front_ht == 2) | (front_ht == 3)).float()
        has_key = norm_v[..., 8:14].max(dim=-1, keepdim=True)[0]
        speed = torch.sqrt(norm_v[..., 14:15] ** 2 + norm_v[..., 15:16] ** 2 + 1e-6)
        door_in = torch.cat(
            [std_door, locked_door, front_dist, front_is_door, norm_v[..., 8:14], has_key, speed, prev_action],
            dim=-1,
        )

        # 4. Progression domain: item/key entities + item/locked-door rays + keys + survival vitals + prev_action
        item_ent = self.item_entity_encoder(
            obs["visible_entities"]["features"],
            obs["visible_entities"]["entity_type"],
            obs["visible_entities"]["mask"],
        )
        item_ray = nearest_ray_features(dist, ht, 5, self.ray_cos, self.ray_sin)
        prog_vitals = torch.cat([norm_v[..., 8:14], norm_v[..., 0:2], norm_v[..., 3:4]], dim=-1)
        prog_in = torch.cat([item_ent, item_ray, locked_door, prog_vitals, prev_action], dim=-1)

        return {
            "nav": nav_in,
            "enemy": enemy_in,
            "door": door_in,
            "progression": prog_in,
        }

    def _cluster_forward(
        self,
        obs: dict[str, torch.Tensor],
        fused_feat: torch.Tensor,
        gru_out: torch.Tensor,
        prev_action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluates all 4 specialist experts and blends their logits via the neural router."""
        domain_inputs = self._extract_domain_inputs(obs, prev_action)
        expert_logits = torch.stack(
            [self.experts[name](domain_inputs[name], gru_out) for name in self.EXPERT_NAMES],
            dim=-2,
        )
        gate_logits, gate_probs = self.router(fused_feat, gru_out)
        logits = (gate_probs.unsqueeze(-1) * expert_logits).sum(dim=-2)
        values = self.critic_head(gru_out).squeeze(-1)
        return logits, values, expert_logits, gate_logits, gate_probs

    def initial_state(self, batch_size: int = 1, device: torch.device | str = "cpu") -> torch.Tensor:
        """Returns zero hidden state for the GRU."""
        return torch.zeros(1, batch_size, self.hidden_dim, device=device)

    def initial_prev_action(self, batch_size: int = 1, device: torch.device | str = "cpu") -> torch.Tensor:
        """Returns zero previous-action buffer."""
        return torch.zeros(batch_size, self.num_actions, device=device)

    def forward_step(
        self,
        obs: dict[str, torch.Tensor],
        hx: torch.Tensor,
        prev_action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single decision forward pass returning logits, value, next_hx, and routing details."""
        pa = self._resolve_prev_action(obs, prev_action)
        fused = self.extract_features(obs, prev_action=pa)
        out, next_hx = self.gru(fused.unsqueeze(1), hx)
        gru_out = out.squeeze(1)
        logits, value, expert_logits, gate_logits, gate_probs = self._cluster_forward(
            obs, fused, gru_out, pa
        )
        return logits, value, next_hx, expert_logits, gate_logits, gate_probs

    def step(
        self,
        obs: dict[str, torch.Tensor],
        hx: torch.Tensor,
        prev_action: torch.Tensor | None = None,
        deterministic: bool = False,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single decision step for agent acting and PPO rollouts."""
        logits, value, next_hx, _, _, gate_probs = self.forward_step(obs, hx, prev_action=prev_action)
        self.last_gate_probs = gate_probs.detach()

        scaled_logits = logits / max(1e-4, float(temperature))
        dist = Bernoulli(logits=scaled_logits)
        action = (logits > 0.0).float() if deterministic else dist.sample()

        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return action, log_prob, entropy, value, next_hx

    def evaluate_sequence(
        self,
        obs_seq: dict[str, torch.Tensor],
        init_hx: torch.Tensor,
        actions_seq: torch.Tensor,
        done_mask_seq: torch.Tensor | None = None,
        prev_actions_seq: torch.Tensor | None = None,
        return_aux: bool = False,
    ) -> (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]
    ):
        """Sequence forward pass for PPO and Behavioral Cloning training updates."""
        if prev_actions_seq is None:
            prev_actions_seq = torch.zeros_like(actions_seq)
            if actions_seq.shape[1] > 1:
                shifted = actions_seq[:, :-1, :]
                if done_mask_seq is not None:
                    shifted = shifted * done_mask_seq[:, :-1].unsqueeze(-1)
                prev_actions_seq[:, 1:, :] = shifted

        features = self.extract_features(obs_seq, prev_action=prev_actions_seq)

        if done_mask_seq is not None:
            B, T, _ = features.shape
            outputs = []
            hx = init_hx
            for t in range(T):
                if t > 0:
                    hx = hx * done_mask_seq[:, t - 1].view(1, B, 1)
                out, hx = self.gru(features[:, t : t + 1, :], hx)
                outputs.append(out)
            gru_out = torch.cat(outputs, dim=1)
        else:
            gru_out, _ = self.gru(features, init_hx)

        logits, values, expert_logits, gate_logits, gate_probs = self._cluster_forward(
            obs_seq, features, gru_out, prev_actions_seq
        )

        dist = Bernoulli(logits=logits)
        log_probs = dist.log_prob(actions_seq).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        if not return_aux:
            return log_probs, entropy, values

        gate_entropy = -(gate_probs * torch.log(gate_probs.clamp(min=1e-8))).sum(dim=-1)
        return (
            log_probs,
            entropy,
            values,
            {
                "logits": logits,
                "expert_logits": expert_logits,
                "gate_logits": gate_logits,
                "gate_probs": gate_probs,
                "gate_entropy": gate_entropy,
            },
        )
