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


# -------------------- DiT Block --------------------
class DiTBlock(nn.Module):
    def __init__(self, dim, nhead, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=nhead, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Self-attention + MLP (pre-norm)
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        x_norm = self.norm2(x)
        x = x + self.mlp(x_norm)
        return x


# -------------------- FlowAgent_DiT --------------------
class FlowAgent_DiT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config["d_model"]
        self.chunk_size = config["chunk_size"]
        self.action_dim = config["action_dim"]
        self.num_steps = config["num_steps"]

        # ---- Backbone ----
        self.backbone = VisionBackbone(config["d_model"], config["resnet_name"], False, False)

        # ---- Projections ----
        self.img_proj = nn.Linear(self.backbone.num_channels, self.d_model)
        self.state_proj = nn.Linear(config["d_proprioception"], self.d_model)
        self.action_proj = nn.Linear(self.action_dim, self.d_model)
        self.time_proj = nn.Linear(self.d_model, self.d_model)

        # ---- Transformer Blocks ----
        self.blocks = nn.ModuleList([
            DiTBlock(dim=self.d_model, nhead=8, mlp_ratio=4)
            for _ in range(6)
        ])
        self.cond_cat_proj = nn.Linear(self.d_model * 4, self.d_model)
        self.out = nn.Linear(self.d_model, self.action_dim)

    def forward(self, obs, action):
        """
        obs: dict { "rgb": (B, 1, 3, H, W), "state": (B, state_dim) }
        action: (B, chunk, action_dim)
        """
        B = obs["rgb"].shape[0]
        device = obs["rgb"].device

        # ---- Diffusion noise ----
        t = torch.rand(B, 1, device=device)
        noise = torch.randn_like(action)
        x_t = action * t.view(B, 1, 1) + noise * (1 - t).view(B, 1, 1)
        u_t = action - noise  # target (velocity)

        # ---- Embeddings ----
        t_emb = self.time_proj(timestep_embedding(t, self.d_model))   # (B, d_model)
        features, _ = self.backbone(NestedTensor(obs["rgb"].squeeze(1), None))
        img_emb = self.img_proj(features[0].tensors.mean(dim=[2, 3])) # (B, d_model)
        state_emb = self.state_proj(obs["state"])                     # (B, d_model)

        # ---- 组合成 token 序列 ----
        # broadcast 每个条件到时间序列长度
        img_token = img_emb.unsqueeze(1).expand(B, self.chunk_size, self.d_model)
        state_token = state_emb.unsqueeze(1).expand(B, self.chunk_size, self.d_model)
        t_token = t_emb.unsqueeze(1).expand(B, self.chunk_size, self.d_model)
        act_token = self.action_proj(x_t)                             # (B, chunk, d_model)

        # concat: [noisy_action | state | image | time]
        tokens = torch.cat([act_token, state_token, img_token, t_token], dim=-1)
        # 经过线性层统一映射到 d_model
        # tokens = nn.Linear(tokens.shape[-1], self.d_model, device=device)(tokens)
        tokens = self.cond_cat_proj(tokens)

        # ---- Transformer ----
        for blk in self.blocks:
            tokens = blk(tokens)

        # ---- 输出动作速度场 ----
        v_t = self.out(tokens)
        loss = F.mse_loss(v_t, u_t)
        return loss, v_t

    @torch.no_grad()
    def get_action(self, obs):
        """Euler integration through the learned flow field."""
        self.eval()
        device = obs["rgb"].device
        B = obs["rgb"].shape[0]
        x_t = torch.randn(B, self.chunk_size, self.action_dim, device=device)
        dt = 1.0 / self.num_steps

        # ---- Static embeddings ----
        features, _ = self.backbone(NestedTensor(obs["rgb"].squeeze(1), None))
        img_emb = self.img_proj(features[0].tensors.mean(dim=[2, 3]))   # (B, d_model)
        state_emb = self.state_proj(obs["state"])                       # (B, d_model)

        # ---- Euler integration ----
        for step in range(self.num_steps):
            t = torch.ones(B, 1, device=device) * (step / self.num_steps)
            t_emb = self.time_proj(timestep_embedding(t, self.d_model))

            img_token = img_emb.unsqueeze(1).expand(B, self.chunk_size, self.d_model)
            state_token = state_emb.unsqueeze(1).expand(B, self.chunk_size, self.d_model)
            t_token = t_emb.unsqueeze(1).expand(B, self.chunk_size, self.d_model)
            act_token = self.action_proj(x_t)

            tokens = torch.cat([act_token, state_token, img_token, t_token], dim=-1)
            # tokens = nn.Linear(tokens.shape[-1], self.d_model, device=device)(tokens)
            tokens = self.cond_cat_proj(tokens)

            for blk in self.blocks:
                tokens = blk(tokens)

            v_t = self.out(tokens)
            x_t = x_t + v_t * dt

        self.train()
        return x_t