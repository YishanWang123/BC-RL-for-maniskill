import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from network.components.visual_feature_extractor import VisionBackbone, NestedTensor


def timestep_embedding(t, dim):
    # 强制把 t 规整成 (B,1)
    if t.dim() > 2:
        t = t.view(t.shape[0], -1)[:, :1]
    elif t.dim() == 1:
        t = t[:, None]

    half = dim // 2
    freqs = torch.exp(
        -torch.arange(half, dtype=t.dtype, device=t.device) * (math.log(10000) / half)
    )
    args = t * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if emb.shape[-1] < dim:
        emb = F.pad(emb, (0, dim - emb.shape[-1]))
    return emb  # (B, dim)


class MIP2StepAgent(nn.Module):
    """
    MIP (two-step) policy:
      - step0: input I0 = 0, t=0 -> predict action
      - step2: input It = t* a + (1-t*) z, t=t* -> predict action
    Inference is deterministic (z=0):
      a0 = pi(o, 0, 0)
      a  = pi(o, t* a0, t*)
    """
    def __init__(self, config):
        super().__init__()
        self.chunk_size = config["chunk_size"]
        self.action_dim = config["action_dim"]
        self.d_model = config["d_model"]
        self.num_steps = config.get("num_steps", 20)
        self.t_star = float(config.get("t_fixed", 0.9))  # t*

        # vision
        self.backbone = VisionBackbone(
            config["d_model"],
            config["resnet_name"],
            config["return_interm_layers"],
            config["include_depth"],
        )
        self.img_dim = self.backbone.num_channels
        self.img_proj = nn.Linear(self.img_dim, self.d_model)

        # state + time
        self.state_proj = nn.Linear(config["d_proprioception"], self.d_model)
        self.time_proj = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, self.d_model),
        )

        # action token projection
        self.xt_proj = nn.Linear(self.action_dim, self.d_model)
        self.action_out = nn.Linear(self.d_model, self.action_dim)

        # encoder-only transformer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=8,
            dim_feedforward=self.d_model * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=6)

        self.mem_ln = nn.LayerNorm(self.d_model)
        self.tok_ln = nn.LayerNorm(self.d_model)

    def encode_tokens(self, obs, t_scalar):
        """
        t_scalar: (B,1)
        x_in:     (B,chunk,act)  -> projected to (B,chunk,d)
        returns tokens: (B, 3+chunk, d)
        """
        # img token
        feats, _ = self.backbone(NestedTensor(obs["rgb"].squeeze(1), None))
        img = feats[0].tensors.mean(dim=[2, 3])              # (B,C)
        img_tok = self.img_proj(img).unsqueeze(1)            # (B,1,d)

        # state token
        st_tok = self.state_proj(obs["state"]).unsqueeze(1)  # (B,1,d)

        # time token (建议放大尺度)
        t_emb = timestep_embedding(t_scalar, self.d_model)  # (B,d)
        t_tok = self.time_proj(t_emb).unsqueeze(1)                           # (B,1,d)

        # action/noise tokens
        # x_tok = self.xt_proj(x_in)                                           # (B,chunk,d)

        tokens = torch.cat([img_tok, st_tok, t_tok], dim=1)           # (B,3+chunk,d)
        return self.mem_ln(tokens)

    def predict_action(self, obs, x_in, t_scalar):
        """
        x_in: (B,chunk,act)   (I0 or It*)
        t_scalar: (B,1)
        """
        assert x_in.dim() == 3 and x_in.shape[1] == self.chunk_size and x_in.shape[2] == self.d_model
        assert t_scalar.dim() == 2 and t_scalar.shape[1] == 1

        tokens = self.encode_tokens(obs, t_scalar)
        tgt = x_in
        h = self.transformer(tgt, memory = tokens)                         # (B,3+chunk,d)
        h_x = h[:, -self.chunk_size:, :]                     # 只取 action tokens
        h_x = self.tok_ln(h_x)
        return self.action_out(h_x)                           # (B,chunk,act)

    def forward(self, obs, action):
        """
        MIP training:
          loss = ||pi(o, 0, 0) - a||^2 + ||pi(o, I_t*, t*) - a||^2
          where I_t* = t* a + (1-t*) z, z ~ N(0,1)
        """
        B = action.shape[0]
        device = action.device

        # step0: I0 = 0, t=0
        I0 = torch.zeros_like(action)
        t0 = torch.zeros((B, 1), device=device)
        I0 = self.xt_proj(I0)
        a0_pred = self.predict_action(obs, I0, t0)
        a0_target = action * self.t_star

        # step2: It* = t* a + (1-t*) z, t=t*
        z = torch.randn_like(action)
        t_star_scalar = torch.full((B, 1), self.t_star, device=device)
        t_star_bcast = t_star_scalar[:, :, None]            # (B,1,1) for mixing
        It = t_star_bcast * action + (1.0 - t_star_bcast) * z
        It = self.xt_proj(It)
        a_star_pred = self.predict_action(obs, It, t_star_scalar)

        loss = F.mse_loss(a0_pred, a0_target) + F.mse_loss(a_star_pred, action)
        return loss, a_star_pred

    @torch.no_grad()
    def get_action(self, obs):
        """
        Deterministic MIP inference (z=0):
          a0 = pi(o, 0, 0)
          a  = pi(o, t* a0, t*)
        """
        device = obs["rgb"].device
        B = obs["rgb"].shape[0]

        I0 = torch.zeros((B, self.chunk_size, self.action_dim), device=device)
        I0 = self.xt_proj(I0)
        t0 = torch.zeros((B, 1), device=device)
        a0 = self.predict_action(obs, I0, t0)
        a0 = self.xt_proj(a0)

        t_star_scalar = torch.full((B, 1), self.t_star, device=device)
        # It = self.t_star * a0                               # 因为 z=0, I0=0 -> It = t* a0
        a = self.predict_action(obs, a0, t_star_scalar)
        return a