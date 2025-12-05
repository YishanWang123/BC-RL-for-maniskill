# utils/replay_buffer.py
import torch

class ReplayBuffer:
    def __init__(self, capacity, obs_shape, state_dim, device,
                 act_shape=None, act_dim=None):
        assert (act_shape is None) ^ (act_dim is None), "act_shape 和 act_dim 二选一"
        self.capacity = int(capacity)
        self.device = device

        self.obs_shape = tuple(obs_shape)          # (3,H,W)
        self.state_dim = int(state_dim)
        self.act_shape = tuple(act_shape) if act_shape is not None else (int(act_dim),)

        self.ptr = 0
        self.size = 0

        # —— 全部放 CPU；图像用 uint8（压缩 4 倍），状态/动作/奖励用半精度 —— #
        C,H,W = self.obs_shape
        self.obs       = torch.empty((self.capacity, C, H, W), dtype=torch.uint8, device='cpu').pin_memory()
        self.obs_next  = torch.empty((self.capacity, C, H, W), dtype=torch.uint8, device='cpu').pin_memory()
        self.state     = torch.zeros((self.capacity, self.state_dim), dtype=torch.float16, device='cpu').pin_memory()
        self.state_next= torch.zeros((self.capacity, self.state_dim), dtype=torch.float16, device='cpu').pin_memory()
        self.action    = torch.zeros((self.capacity,) + self.act_shape, dtype=torch.float16, device='cpu').pin_memory()
        self.reward    = torch.zeros((self.capacity, 1), dtype=torch.float16, device='cpu').pin_memory()
        self.done      = torch.zeros((self.capacity, 1), dtype=torch.float16, device='cpu').pin_memory()

    def add(self, obs, state, action, reward, obs_next, state_next, done):
        i = self.ptr

        # 图像先夹在 [0,1] or [0,255] 的张量上来，统一存 uint8
        # 你 dataset 里是 float32 [0,1] 的话：乘 255
        self.obs[i].copy_((obs.detach().clamp(0,1)*255).to(torch.uint8, copy=True).cpu())
        self.obs_next[i].copy_((obs_next.detach().clamp(0,1)*255).to(torch.uint8, copy=True).cpu())

        self.state[i].copy_(state.detach().to(torch.float16).cpu())
        self.state_next[i].copy_(state_next.detach().to(torch.float16).cpu())

        a = action.detach().to(torch.float16).cpu()
        if a.shape != self.action[i].shape: a = a.view_as(self.action[i])
        self.action[i].copy_(a)

        self.reward[i].fill_(float(reward))
        self.done[i].fill_(float(done))

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def __len__(self): return self.size

    def sample(self, batch_size):
        idx = torch.randint(0, self.size, (batch_size,), device='cpu')

        # —— 搬到 GPU，并做图像归一化（与 eval 的 Normalize 保持一致）—— #
        obs_u8  = self.obs[idx]
        obsn_u8 = self.obs_next[idx]
        obs  = obs_u8.to(self.device, non_blocking=True).to(torch.float32) / 255.0
        obsn = obsn_u8.to(self.device, non_blocking=True).to(torch.float32) / 255.0

        # Normalize: (x-mean)/std
        mean = torch.tensor([0.485,0.456,0.406], device=self.device).view(1,3,1,1)
        std  = torch.tensor([0.229,0.224,0.225], device=self.device).view(1,3,1,1)
        obs  = (obs - mean)/std
        obsn = (obsn - mean)/std

        state      = self.state[idx].to(self.device, non_blocking=True).to(torch.float32)
        state_next = self.state_next[idx].to(self.device, non_blocking=True).to(torch.float32)
        action     = self.action[idx].to(self.device, non_blocking=True).to(torch.float32)
        reward     = self.reward[idx].to(self.device, non_blocking=True).to(torch.float32)
        done       = self.done[idx].to(self.device, non_blocking=True).to(torch.float32)

        return obs, state, action, reward, obsn, state_next, done