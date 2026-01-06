import torch
import torch.nn as nn
import torch.nn.functional as F

from network.components.visual_feature_extractor import VisionBackbone, NestedTensor


class MLP_L1_Agent(nn.Module):
    """Simple MLP agent conditioned on image + proprioceptive state.

    Expects `obs` dict with keys:
      - "rgb": tensor shaped (B, C, H, W) or (B,1,C,H,W) as used elsewhere (we use .squeeze if needed)
      - "state": tensor shaped (B, state_dim)

    Methods:
      - forward(obs, action): returns (loss, pred_action)
      - get_action(obs): returns pred_action
    Uses L1 loss (MAE) for training.
    """

    def __init__(self, config):
        super().__init__()
        self.state_dim = config["d_proprioception"]
        self.d_model = config["d_model"]
        self.chunk_size = config["chunk_size"]
        self.action_dim = config["action_dim"]
        self.action_total_dim = self.action_dim * self.chunk_size

        # visual backbone to match existing observation preprocessing
        self.backbone = VisionBackbone(self.d_model, config["resnet_name"], config.get("return_interm_layers", False), config.get("include_depth", False))
        self.img_dim = self.backbone.num_channels

        # project proprioceptive state into model dim
        self.state_projection = nn.Linear(self.state_dim, self.d_model)

        # condition dimension: image features + projected state
        self.cond_dim = self.img_dim + self.d_model

        # MLP hidden sizes (small, configurable)
        h1 = config.get("mlp_h1", 1024)
        h2 = config.get("mlp_h2", 512)

        self.fc1 = nn.Linear(self.cond_dim, h1)
        self.norm1 = nn.LayerNorm(h1)
        self.fc2 = nn.Linear(h1, h2)
        self.norm2 = nn.LayerNorm(h2)
        self.fc3 = nn.Linear(h2, self.action_total_dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=config.get("dropout", 0.0))

        self.loss_fn = nn.L1Loss()

    def forward(self, obs, action=None):
        """If `action` provided, compute L1 loss against prediction and return (loss, pred).
        If `action` is None, returns (None, pred).
        """
        # ensure rgb shape is (B, C, H, W)
        rgb = obs["rgb"]
        if rgb.dim() == 5 and rgb.shape[1] == 1:
            # some pipelines use (B,1,C,H,W)
            rgb = rgb.squeeze(1)

        features, _ = self.backbone(NestedTensor(rgb, None))
        img_features = features[0].tensors.mean(dim=[2, 3])

        state_features = self.state_projection(obs["state"])

        cond = torch.cat([img_features, state_features], dim=-1)

        x = self.act(self.fc1(cond))
        x = self.dropout(self.norm1(x))
        x = self.act(self.fc2(x))
        x = self.dropout(self.norm2(x))
        out = self.fc3(x)

        pred = out.view(-1, self.chunk_size, self.action_dim)

        if action is not None:
            loss = self.loss_fn(pred, action)
            return loss, pred
        else:
            return None, pred

    @torch.no_grad()
    def get_action(self, obs):
        """Return predicted action tensor shaped (B, chunk_size, action_dim)."""
        _, pred = self.forward(obs, action=None)
        return pred


def build_mlp_agent(config):
    """Factory helper: build MLP agent from config dict."""
    return MLP_L1_Agent(config)
