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

        # -------- FiLM 调制层 --------
        self.film_scale1 = nn.Linear(dim, dim)
        self.film_shift1 = nn.Linear(dim, dim)
        self.film_scale2 = nn.Linear(dim, dim)
        self.film_shift2 = nn.Linear(dim, dim)

        # --- 初始化为恒等映射 ---
        for m in [self.film_scale1, self.film_scale2]:
            nn.init.zeros_(m.weight)
            nn.init.ones_(m.bias)
        for m in [self.film_shift1, self.film_shift2]:
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x, cond):
        # cond: (B, d_model)
        # 使用 tanh 限幅控制数值范围
        scale1 = 1 + 0.1 * torch.tanh(self.film_scale1(cond)).unsqueeze(1)
        shift1 = 0.1 * torch.tanh(self.film_shift1(cond)).unsqueeze(1)
        scale2 = 1 + 0.1 * torch.tanh(self.film_scale2(cond)).unsqueeze(1)
        shift2 = 0.1 * torch.tanh(self.film_shift2(cond)).unsqueeze(1)

        # --- Self-Attention ---
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + (scale1 * attn_out + shift1)

        # --- Feed Forward ---
        x_norm = self.norm2(x)
        mlp_out = self.mlp(x_norm)
        x = x + (scale2 * mlp_out + shift2)
        return x


class FlowAgent_DiT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config["d_model"]
        self.chunk_size = config["chunk_size"]
        self.action_dim = config["action_dim"]
        self.num_steps = config["num_steps"]

        # ---- Backbone & projections ----
        self.backbone = VisionBackbone(config["d_model"], config["resnet_name"], False, False)
        self.img_proj = nn.Linear(self.backbone.num_channels, self.d_model)
        self.state_proj = nn.Linear(config["d_proprioception"], self.d_model)
        self.action_proj = nn.Linear(self.action_dim, self.d_model)
        self.time_embed = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, self.d_model),
        )

        # ---- Transformer blocks ----
        self.blocks = nn.ModuleList([
            DiTBlock(dim=self.d_model, nhead=8, mlp_ratio=4)
            for _ in range(6)
        ])
        self.out = nn.Linear(self.d_model, self.action_dim)

    def forward(self, obs, action):
        B = obs["rgb"].shape[0]
        t = torch.rand(B, 1, device=obs["rgb"].device)
        noise = torch.randn_like(action)
        x_t = action * t.view(B, 1, 1) + noise * (1 - t).view(B, 1, 1)
        u_t = action - noise

        # ---- Embeddings ----
        t_emb = self.time_embed(timestep_embedding(t, self.d_model))          # (B, d_model)
        features, _ = self.backbone(NestedTensor(obs["rgb"].squeeze(1), None))
        img_emb = self.img_proj(features[0].tensors.mean(dim=[2, 3]))         # (B, d_model)
        cond = img_emb + t_emb                                                # 全局条件

        # ---- Token sequence: [state + noisy_action_seq] ----
        state_token = self.state_proj(obs["state"]).unsqueeze(1)              # (B, 1, d_model)
        act_tokens = self.action_proj(x_t)                                    # (B, chunk, d_model)
        tokens = torch.cat([state_token, act_tokens], dim=1)                  # (B, 1+chunk, d_model)

        # ---- Transformer ----
        for blk in self.blocks:
            tokens = blk(tokens, cond)

        # ---- Predict velocity ----
        v_t = self.out(tokens[:, 1:, :])                                      # 去掉 state token
        loss = F.mse_loss(v_t, u_t)
        return loss, v_t

    @torch.no_grad()
    def get_action(self, obs):
        self.eval()
        device = obs["rgb"].device
        B = obs["rgb"].shape[0]

        x_t = torch.randn(B, self.chunk_size, self.action_dim, device=device)
        dt = 1.0 / self.num_steps

        # ---- Static embeddings ----
        features, _ = self.backbone(NestedTensor(obs["rgb"].squeeze(1), None))
        img_emb = self.img_proj(features[0].tensors.mean(dim=[2, 3]))          # (B, d_model)
        state_token = self.state_proj(obs["state"]).unsqueeze(1)

        # ---- Euler integration ----
        for step in range(self.num_steps):
            t = torch.ones(B, 1, device=device) * (step / self.num_steps)
            t_emb = self.time_embed(timestep_embedding(t, self.d_model))
            cond = img_emb + t_emb

            act_tokens = self.action_proj(x_t)
            tokens = torch.cat([state_token, act_tokens], dim=1)

            for blk in self.blocks:
                tokens = blk(tokens, cond)

            v_t = self.out(tokens[:, 1:, :])
            x_t = x_t + v_t * dt

        self.train()
        return x_t

    
    @torch.no_grad()
    def get_action(self, obs):
        """
        Euler integration through the learned vector field to sample actions.

        Args:
            obs: dict {
                "rgb": [B, 1, 3, H, W],
                "state": [B, state_dim]
            }

        Returns:
            x_0: [B, chunk_size, action_dim] — predicted clean action sequence
        """
        self.eval()
        device = obs["rgb"].device
        B = obs["rgb"].shape[0]

        # 初始化 x_T 为高斯噪声
        x_t = torch.randn(B, self.chunk_size, self.action_dim, device=device)
        dt = 1.0 / self.num_steps

        # ---- 提取静态嵌入 ----
        # 图像
        features, _ = self.backbone(NestedTensor(obs["rgb"].squeeze(1), None))
        img_emb = self.img_proj(features[0].tensors.mean(dim=[2, 3]))  # (B, d_model)
        # 状态
        state_token = self.state_proj(obs["state"]).unsqueeze(1)       # (B, 1, d_model)

        # ---- 时间积分 ----
        for step in range(self.num_steps):
            # 当前时间步
            t = torch.ones(B, 1, device=device) * (step / self.num_steps)
            t_emb = self.time_embed(timestep_embedding(t, self.d_model))
            cond = img_emb + t_emb                                     # (B, d_model)

            # 动作 token 序列
            act_tokens = self.action_proj(x_t)                         # (B, chunk, d_model)
            tokens = torch.cat([state_token, act_tokens], dim=1)       # (B, 1+chunk, d_model)

            # Transformer
            for blk in self.blocks:
                tokens = blk(tokens, cond)

            # 预测速度场
            v_t = self.out(tokens[:, 1:, :])                           # (B, chunk, action_dim)
            x_t = x_t + v_t * dt                                       # 欧拉积分一步

        self.train()
        return x_t