import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from network.components.visual_feature_extractor import VisionBackbone, NestedTensor


def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-torch.arange(half, dtype=t.dtype, device=t.device) * (math.log(10000) / half))
    args = t * freqs
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class FlowAgent_DiT(nn.Module):
    """
    Diffusion Transformer-based Flow Matching policy for visuomotor imitation.
    Replaces MLP with DiT-style Transformer encoder.
    """
    def __init__(self, config):
        super().__init__()
        self.num_steps = config["num_steps"]
        self.chunk_size = config["chunk_size"]
        self.action_dim = config["action_dim"]
        self.d_model = config["d_model"]

        # ---- Vision encoder ----
        self.backbone = VisionBackbone(config["d_model"], config["resnet_name"], config["return_interm_layers"], config["include_depth"])
        self.img_dim = self.backbone.num_channels
        self.img_proj = nn.Linear(self.img_dim, self.d_model)

        # ---- State + time projection ----
        self.state_proj = nn.Linear(config["d_proprioception"], self.d_model)
        self.time_proj = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, self.d_model),
        )

        # ---- Action token projection ----
        self.action_proj = nn.Linear(self.action_dim, self.d_model)
        self.action_out = nn.Linear(self.d_model, self.action_dim)

        # ---- Transformer ----
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=8,
            dim_feedforward=self.d_model * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)

        self.layernorm = nn.LayerNorm(self.d_model)

    # ----------------------------- #
    #           Training
    # ----------------------------- #
    def forward(self, obs, action):
        """
        obs: dict { "rgb": (B, 1, 3, H, W), "state": (B, state_dim) }
        action: (B, chunk, act_dim)
        """
        B = obs["rgb"].shape[0]
        device = obs["rgb"].device

        # ---- Sample time and noise ----
        t = torch.rand(B, 1, device=device)
        t_emb = timestep_embedding(t, self.d_model)
        t_emb = self.time_proj(t_emb)

        noise = torch.randn_like(action)
        x_t = action * t.view(B, 1, 1) + noise * (1 - t).view(B, 1, 1)
        u_t = action - noise

        # ---- Encode vision ----
        features, _ = self.backbone(NestedTensor(obs["rgb"].squeeze(1), None))
        img_features = features[0].tensors.mean(dim=[2, 3])  # (B, C)
        img_token = self.img_proj(img_features).unsqueeze(1)

        # ---- Encode state and time ----
        state_token = self.state_proj(obs["state"]).unsqueeze(1)
        time_token = t_emb.unsqueeze(1)

        # ---- Project actions into tokens ----
        act_tokens = self.action_proj(x_t)  # (B, chunk, d_model)

        # ---- Combine all tokens ----
        tokens = torch.cat([time_token, img_token, state_token, act_tokens], dim=1)
        tokens = self.layernorm(tokens)

        # ---- Run transformer ----
        encoded = self.transformer(tokens)

        # ---- Take only action token outputs ----
        action_output_tokens = encoded[:, -self.chunk_size:, :]
        v_t = self.action_out(action_output_tokens)

        # ---- Compute Flow Matching loss ----
        loss = F.mse_loss(v_t, u_t)
        return loss, v_t

    # ----------------------------- #
    #           Inference
    # ----------------------------- #
    def get_action(self, obs):
        """Euler integration through the learned vector field."""
        device = obs["rgb"].device
        B = obs["rgb"].shape[0]
        dt = 1.0 / self.num_steps
        x_t = torch.randn(B, self.chunk_size, self.action_dim, device=device)

        # static encodings
        features, _ = self.backbone(NestedTensor(obs["rgb"].squeeze(1), None))
        img_features = features[0].tensors.mean(dim=[2, 3])
        img_token = self.img_proj(img_features).unsqueeze(1)
        state_token = self.state_proj(obs["state"]).unsqueeze(1)

        for step in range(self.num_steps):
            t = torch.ones(B, 1, device=device) * (step / self.num_steps)
            t_emb = timestep_embedding(t, self.d_model)
            t_emb = self.time_proj(t_emb).unsqueeze(1)

            act_tokens = self.action_proj(x_t)
            tokens = torch.cat([t_emb, img_token, state_token, act_tokens], dim=1)
            tokens = self.layernorm(tokens)
            encoded = self.transformer(tokens)
            v_t = self.action_out(encoded[:, -self.chunk_size:, :])
            x_t = x_t + v_t * dt  # Euler update

        return x_t