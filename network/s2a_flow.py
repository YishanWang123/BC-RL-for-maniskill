import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from network.components.visual_feature_extractor import VisionBackbone, NestedTensor


def timestep_embedding(t, dim):
    """
    t: (B, 1)
    return: (B, dim)
    """
    half = dim // 2
    freqs = torch.exp(
        -torch.arange(half, device=t.device, dtype=t.dtype)
        * (math.log(10000) / half)
    )
    args = t * freqs
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    return emb


class FlowAgentRectified(nn.Module):
    """
    Noise-free Rectified Flow:
        x_t = (1 - t) * state + t * action
        u_t = action - state
        v_theta(x_t, t | image)
    """

    def __init__(self, config):
        super().__init__()

        self.num_steps = config["num_steps"]
        self.chunk_size = config["chunk_size"]
        self.action_dim = config["action_dim"]
        self.action_total_dim = self.chunk_size * self.action_dim

        self.state_dim = config["d_proprioception"]
        self.time_dim = config["d_model"]

        # vision backbone
        self.backbone = VisionBackbone(
            config["d_model"],
            config["resnet_name"],
            config["return_interm_layers"],
            config["include_depth"],
        )
        self.img_dim = self.backbone.num_channels

        # ---------- Flow MLP (input: x_t only) ----------
        self.fc1 = nn.Linear(self.action_total_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, self.action_total_dim)

        self.act = nn.SiLU()
        self.norm1 = nn.LayerNorm(256)
        self.norm2 = nn.LayerNorm(256)
        self.norm3 = nn.LayerNorm(256)
        self.convert_proj = nn.Linear(self.state_dim, self.action_total_dim)

        # ---------- FiLM: cond = image + time ----------
        self.cond_encoder = nn.Sequential(
            nn.Linear(self.img_dim + self.time_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 256 * 2 * 3),  # (gamma, beta) × 3 layers
        )

    # -------------------------------------------------
    # training
    # -------------------------------------------------
    def forward(self, obs, action):
        """
        obs:
            obs["rgb"]   : (B, C, H, W)
            obs["state"] : (B, action_total_dim)
        action:
            (B, action_total_dim)
        """
        B = action.shape[0]
        device = action.device

        # sample t ~ U(0,1)
        t = torch.rand(B, 1, device=device)
        

        # construct x_t (noise-free)
        state = obs["state"]
        state_a = self.convert_proj(state)       # [64, 40]
        action_flat = action.view(B, -1)
        # state = state_a.reshape(B, self.chunk_size, self.action_dim)
        # print("state_a shape:", state_a.shape)
        # print("state shape:", state.shape)
        # print("t shape:", t.shape)
        # print("state shape:", state.shape)
        # print("action shape:", action.shape)
        # import pdb; pdb.set_trace()
        x_t = (1.0 - t) * state_a + t * action_flat
        x_t = x_t.view(B, -1)

        # target vector field
        u_t = action_flat - state_a
        u_t = u_t.view(B, self.chunk_size, self.action_dim)

        # image feature
        feats, _ = self.backbone(NestedTensor(obs["rgb"], None))
        img_feat = feats[0].tensors.mean(dim=[2, 3])  # (B, C)

        # time embedding
        t_emb = timestep_embedding(t, self.time_dim)

        # FiLM parameters
        cond = torch.cat([img_feat, t_emb], dim=-1)
        film = self.cond_encoder(cond).view(B, 3, 2, 256)
        # (g1, b1), (g2, b2), (g3, b3) = film[:, 0], film[:, 1], film[:, 2]
        # film shape: [B, 3, 2, 512]  => 3 layers, 2 = gamma/beta, 512 hidden
        g1, b1 = film[:, 0, 0], film[:, 0, 1]  # shape [B,512]
        g2, b2 = film[:, 1, 0], film[:, 1, 1]
        g3, b3 = film[:, 2, 0], film[:, 2, 1]

        # ---------- MLP with FiLM ----------
        x = self.fc1(x_t)
        x = self.act(x)
        x = g1 * x + b1
        x = self.norm1(x)

        x = self.fc2(x)
        x = self.act(x)
        x = g2 * x + b2
        x = self.norm2(x)

        x = self.fc3(x)
        x = self.act(x)
        x = g3 * x + b3
        x = self.norm3(x)

        v_t = self.fc4(x)
        v_t = v_t.view(B, self.chunk_size, self.action_dim)

        loss = F.mse_loss(v_t, u_t)
        return loss, v_t

    # -------------------------------------------------
    # inference: Euler ODE solve
    # -------------------------------------------------
    @torch.no_grad()
    def get_action(self, obs):
        """
        Integrate dx/dt = v_theta(x_t, t | image)
        """
        device = obs["rgb"].device
        B = obs["rgb"].shape[0]

        dt = 1.0 / self.num_steps

        # initial x_0 = state (noise-free source)
        state = obs["state"].view(B, -1)
        state_a = self.convert_proj(state)       # [64, 40]
        state = state_a.reshape(B, self.chunk_size, self.action_dim)
        x_t = state.view(B, -1)

        # image feature (fixed during rollout)
        feats, _ = self.backbone(NestedTensor(obs["rgb"].squeeze(1), None))
        img_feat = feats[0].tensors.mean(dim=[2, 3])

        for i in range(self.num_steps):
            t = torch.full((B, 1), i / self.num_steps, device=device)
            t_emb = timestep_embedding(t, self.time_dim)

            cond = torch.cat([img_feat, t_emb], dim=-1)
            film = self.cond_encoder(cond).view(B, 3, 2, 256)
            # (g1, b1), (g2, b2), (g3, b3) = film[:, 0], film[:, 1], film[:, 2]
            g1, b1 = film[:, 0, 0], film[:, 0, 1]  # shape [B,512]
            g2, b2 = film[:, 1, 0], film[:, 1, 1]
            g3, b3 = film[:, 2, 0], film[:, 2, 1]

            x = self.fc1(x_t)
            x = self.act(x)
            x = g1 * x + b1
            x = self.norm1(x)

            x = self.fc2(x)
            x = self.act(x)
            x = g2 * x + b2
            x = self.norm2(x)

            x = self.fc3(x)
            x = self.act(x)
            x = g3 * x + b3
            x = self.norm3(x)

            v = self.fc4(x)
            x_t = x_t + dt * v

        return x_t.view(B, self.chunk_size, self.action_dim)