import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from network.components.visual_feature_extractor import VisionBackbone, NestedTensor


class Critic(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config['hidden_dim']
        # ---- Vision encoder ----
        self.backbone = VisionBackbone(config["d_model"], config["resnet_name"], config["return_interm_layers"], config["include_depth"])
        self.img_dim = self.backbone.num_channels
        self.img_proj = nn.Linear(self.img_dim, self.hidden_dim)

        self.state_proj = nn.Linear(config["d_proprioception"], self.hidden_dim)
        self.total_action_dim  = config["action_dim"] * config["chunk_size"]
        self.action_proj = nn.Linear(self.total_action_dim, self.hidden_dim)
        self.value_head = nn.Sequential(
            nn.Linear(self.hidden_dim * 3, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1)
        )
    def forward(self, obs, action):
        """
        obs: dict { "rgb": (B, 1, 3, H, W), "state": (B, state_dim) }
        action: (B, chunk, act_dim)
        """
        B = obs["rgb"].shape[0]
        device = obs["rgb"].device

        # ---- Encode vision ----
        features, _ = self.backbone(NestedTensor(obs["rgb"].squeeze(1), None))
        img_features = features[0].tensors.mean(dim=[2, 3])  # (B, C)
        img_token = self.img_proj(img_features)

        # ---- Encode state ----
        state_token = self.state_proj(obs["state"])

        # ---- Encode action ----
        action_flat = action.view(B, -1)
        action_token = self.action_proj(action_flat)

        # ---- Combine all tokens ----
        combined = torch.cat([img_token, state_token, action_token], dim=-1)

        # ---- Value head ----
        value = self.value_head(combined)
        return value
        

class BCAgent(nn.Module):
    """
    Diffusion Transformer-based BC policy for visuomotor imitation.
    Replaces MLP with Transformer encoder.
    """
    def __init__(self, config):
        super().__init__()
        self.chunk_size = config["chunk_size"]
        self.action_dim = config["action_dim"]
        self.d_model = config["d_model"]

        # ---- Vision encoder ----
        self.backbone = VisionBackbone(config["d_model"], config["resnet_name"], config["return_interm_layers"], config["include_depth"])
        self.img_dim = self.backbone.num_channels
        self.img_proj = nn.Linear(self.img_dim, self.d_model)

        # ---- State + time projection ----
        self.state_proj = nn.Linear(config["d_proprioception"], self.d_model)

        # ---- Action token projection ----
        self.fc = nn.Linear(2, self.chunk_size)
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

    def action_projection(self, encoded):
        encoded = encoded.permute(0, 2, 1)
        x = self.fc(encoded)
        x = x.permute(0, 2, 1)
        return x

    # ----------------------------- #
    #           Training
    # ----------------------------- #
    def forward(self, obs, action=None):
        """
        obs: dict { "rgb": (B, 1, 3, H, W), "state": (B, state_dim) }
        action: (B, chunk, act_dim)
        """
        B = obs["rgb"].shape[0]
        device = obs["rgb"].device

        # ---- Encode vision ----
        features, _ = self.backbone(NestedTensor(obs["rgb"].squeeze(1), None))
        img_features = features[0].tensors.mean(dim=[2, 3])  # (B, C)
        img_token = self.img_proj(img_features).unsqueeze(1)

        # ---- Encode state and time ----
        state_token = self.state_proj(obs["state"]).unsqueeze(1)
        # print("state_token shape:", state_token.shape)
        # print("img_token shape:", img_token.shape)

        # ---- Combine all tokens ----
        tokens = torch.cat([img_token, state_token], dim=1)
        tokens = self.layernorm(tokens)
        # print("tokens shape:", tokens.shape)

        # ---- Run transformer ----
        encoded = self.transformer(tokens)
        # print("encoded shape:", encoded.shape)
        encoded = self.action_projection(encoded)

        # ---- Take only action token outputs ----
        action_output_tokens = encoded[:, -self.chunk_size:, :]
        action_pred = self.action_out(action_output_tokens)
        # print("action_pred shape:", action_pred.shape)
        # import pdb; pdb.set_trace()
        if action is None:
            return _, action_pred
        # ---- Compute Flow Matching loss ----
        loss = F.mse_loss(action_pred, action) 
        return loss, action_pred

    # ----------------------------- #
    #           Inference
    # ----------------------------- #
    @torch.no_grad()
    def get_action(self, obs):

        device = obs["rgb"].device
        B = obs["rgb"].shape[0]
        # dt = 1.0 / self.num_steps
        # x_t = torch.randn(B, self.chunk_size, self.action_dim, device=device)

        # static encodings
        features, _ = self.backbone(NestedTensor(obs["rgb"].squeeze(1), None))
        img_features = features[0].tensors.mean(dim=[2, 3])
        img_token = self.img_proj(img_features).unsqueeze(1)
        state_token = self.state_proj(obs["state"]).unsqueeze(1)

        tokens = torch.cat([img_token, state_token], dim=1)
        tokens = self.layernorm(tokens)
        encoded = self.transformer(tokens)
        encoded = self.action_projection(encoded)
        action_pred = self.action_out(encoded[:, -self.chunk_size:, :])

        return action_pred