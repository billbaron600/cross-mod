import numpy as np
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class PerKeyMLPExtractor(BaseFeaturesExtractor):
    """
    Dict obs  ──► stem[k] ⟶ per-key latent  ┐
                                            │ (concat)
                 … one per obs key …        ├─► fusion MLP ─► features_dim
                                            │
    """

    def __init__(
        self,
        observation_space,
        per_key_dim: int = 32,
        stem_hidden = (64,),
        fusion_hidden  = (256, 256),
    ):
        # The final fusion output defines `features_dim` for the policy
        super().__init__(observation_space, features_dim=fusion_hidden[-1])

        # --- build a small MLP for each dict key --------------------------------
        self.stems = nn.ModuleDict()
        for key, space in observation_space.spaces.items():
            in_dim = int(np.prod(space.shape))
            layers = []
            prev = in_dim
            for h in stem_hidden:
                layers += [nn.Linear(prev, h), nn.ReLU()]
                prev = h
            layers += [nn.Linear(prev, per_key_dim), nn.ReLU()]
            self.stems[key] = nn.Sequential(*layers)

        # --- fusion network (operates on concatenated stem outputs) -------------
        concat_dim = per_key_dim * len(observation_space.spaces)
        fusion_layers = []
        prev = concat_dim
        for h in fusion_hidden:
            fusion_layers += [
                nn.Linear(prev, h),
                nn.LayerNorm(h),          # optional but usually helps
                nn.ReLU(),
            ]
            prev = h
        self.fusion = nn.Sequential(*fusion_layers)

    def forward(self, obs_dict):
        # obs_dict is a Mapping[str, Tensor]; SB3 guarantees consistent key order
        stem_outputs = []
        for k, x in obs_dict.items():           # iterate in insertion order
            x = x.view(x.size(0), -1)           # flatten per key
            stem_outputs.append(self.stems[k](x))
        fused = th.cat(stem_outputs, dim=1)
        return self.fusion(fused)


class PerKeyAngleMLPExtractor(BaseFeaturesExtractor):
    """
    One mini-MLP (“stem”) per dict key, but for any key listed in
    `angle_keys` we first replace each angle θ with [sin θ, cos θ].
    """

    def __init__(
        self,
        observation_space,
        angle_keys=("joint_positions",),     # dict keys to expand as sin/cos
        per_key_dim: int = 32,
        stem_hidden  = (64,),
        fusion_hidden  = (256, 256),
    ):
        super().__init__(observation_space, features_dim=fusion_hidden[-1])

        self.angle_keys = set(angle_keys)
        self.stems = nn.ModuleDict()

        for key, space in observation_space.spaces.items():
            orig_dim = int(np.prod(space.shape))
            in_dim = 2 * orig_dim if key in self.angle_keys else orig_dim

            layers = []
            prev = in_dim
            for h in stem_hidden:
                layers += [nn.Linear(prev, h), nn.ReLU()]
                prev = h
            layers += [nn.Linear(prev, per_key_dim), nn.ReLU()]
            self.stems[key] = nn.Sequential(*layers)

        concat_dim = per_key_dim * len(observation_space.spaces)
        fusion_layers = []
        prev = concat_dim
        for h in fusion_hidden:
            fusion_layers += [
                nn.Linear(prev, h),
                nn.LayerNorm(h),
                nn.ReLU(),
            ]
            prev = h
        self.fusion = nn.Sequential(*fusion_layers)

    def forward(self, obs_dict):
        stem_outputs = []
        for k, x in obs_dict.items():            # keeps dict insertion order
            x = x.view(x.size(0), -1)            # flatten  ✓
            if k in self.angle_keys:
                # replace θ with [sin θ, cos θ] element-wise
                x = th.cat([th.sin(x), th.cos(x)], dim=1)
            stem_outputs.append(self.stems[k](x))
        fused = th.cat(stem_outputs, dim=1)
        return self.fusion(fused)

class NormMLPExtractor(BaseFeaturesExtractor):
    """
    (1) Flatten each sub-observation
    (2) Concatenate
    (3) 2-layer MLP with LayerNorm
    """
    def __init__(self, observation_space, latent_dim=256):
        super().__init__(observation_space, features_dim=latent_dim)
        self._concat_dim = sum(space.shape[0] for space in observation_space.spaces.values())

        self.net = nn.Sequential(
            nn.Linear(self._concat_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(),
        )

    def forward(self, obs_dict):
        flat = th.cat([obs_dict[k].view(obs_dict[k].size(0), -1)
                       for k in obs_dict.keys()], dim=1)
        return self.net(flat)