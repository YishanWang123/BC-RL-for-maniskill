import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from network.components.visual_feature_extractor import VisionBackbone, NestedTensor


def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-torch.arange(half, dtype=t.dtype, device=t.device) * (math.log(10000) / half))
    args = t * freqs            # (B,1)*(half) -> (B,half)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, dim)


class FlowAgent(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_steps = config["num_steps"]
        # self.img_dim = self.backbone.num_channels     # e.g., 512 for resnet18, 2048 for resnet50
        self.state_dim = config["d_proprioception"]
        self.time_dim = config["d_model"]
        self.chunk_size = config["chunk_size"]
        self.action_dim = config["action_dim"]
        self.action_total_dim = self.action_dim * self.chunk_size

        self.backbone = VisionBackbone(config["d_model"], config["resnet_name"], config["return_interm_layers"], config["include_depth"])
        self.img_dim = self.backbone.num_channels
        self.state_projection = nn.Linear(self.state_dim, config["d_model"])
        self.d_model = config["d_model"]
        
        # total cond dimension (img + state + t)
        self.cond_dim = self.img_dim + self.d_model + self.time_dim

        # input to the MLP: cond + x_t
        self.fc_input_dim = self.cond_dim + self.action_total_dim
        # print("FlowAgent fc_input_dim:", self.fc_input_dim)
        # print("img_dim", self.img_dim)
        # print("state_dim", self.state_dim)
        # print("cond_dim", self.cond_dim)
        # print("action_total_dim", self.action_total_dim)
        # import pdb; pdb.set_trace()


        self.fc1 = nn.Linear(self.fc_input_dim, 2048)
        self.fc2 = nn.Linear(2048, 2048)
        self.fc3 = nn.Linear(2048, 1024)
        self.fc4 = nn.Linear(1024, self.action_total_dim)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(p=0.1)  # 0.05–0.3
        self.norm1 = nn.LayerNorm(2048)
        self.norm2 = nn.LayerNorm(2048)
        self.norm3 = nn.LayerNorm(1024)


    def forward(self, obs, action):
        assert action is not None, "training 时必须传入 ground-truth action"
        #time emb
        # t = torch.rand(obs["rgb"].shape[0], 1, device = obs["rgb"].device)
        t = torch.rand(obs["rgb"].shape[0], 1, 1, device=obs["rgb"].device)  # [B, 1, 1]
        t_emb = timestep_embedding(t.squeeze(-1), self.d_model)

        # noise = torch.randn(self.num_steps, obs["RGB"].shape[0], self.d_model).to(obs["RGB"].device)
        noise = torch.randn_like(action).to(obs["rgb"].device)
        
        #image & state 
        features, pos = self.backbone(NestedTensor(obs["rgb"], None))
        # features 是 list[NestedTensor]
        img_features = features[0].tensors.mean(dim=[2,3])     # (B, C, H, W)
        # img_features = x.flatten(2).transpose(1,2)  # (B, HW, C)
        
        state_features = self.state_projection(obs["state"])


        x_t = action * t + noise * (1 - t)
        x_t_flat = x_t.reshape(obs["rgb"].shape[0],-1)
        cond = torch.cat([img_features, state_features, t_emb], dim=-1)
        input = torch.cat([x_t_flat, cond], dim=-1)
        # print(input.shape)
        # print("img_features", img_features.shape)
        # print("state_features", state_features.shape)
        # print("t_emb", t_emb.shape)
        # print("x_t_flat", x_t_flat.shape)
        # import pdb; pdb.set_trace()
        u_t = action - noise

        x = self.act(self.fc1(input))
        x = self.dropout(self.norm1(x))
        x = self.act(self.fc2(x))
        x = self.dropout(self.norm2(x))
        x = self.act(self.fc3(x))
        x = self.dropout(self.norm3(x))
        v_t = self.fc4(x)

        v_t = v_t.view(-1, self.chunk_size, self.action_dim)
        loss = F.mse_loss(v_t, u_t)
        v_t = v_t.view(-1, self.chunk_size, self.action_dim)

        return loss, v_t
    
    def get_action(self, obs):
        """Euler method to get action from cond dist"""
        device = obs["rgb"].device
        B = obs["rgb"].shape[0]
        dt = 1.0 / self.num_steps
        x_t = torch.randn(B, self.action_total_dim).to(device)
        features, _ = self.backbone(NestedTensor(obs["rgb"].squeeze(1), None))
        img = features[0].tensors.mean(dim=[2,3])
        s = self.state_projection(obs["state"])
        for t in range(self.num_steps):
            t = torch.ones(B, 1, device = device) * (t / self.num_steps)
            t_emb = timestep_embedding(t, self.d_model)
            cond = torch.cat([img, s, t_emb], dim=-1)
            input = torch.cat([x_t, cond], dim=-1)
            # v_pred = self.fc2(self.relu(self.fc1(input)))
            v_pred = self.fc4(self.act(self.fc3(self.act(self.fc2(self.act(self.fc1(input)))))))
            x_t = x_t + v_pred * dt
            
        
        return x_t.reshape(B, self.chunk_size, self.action_dim)



