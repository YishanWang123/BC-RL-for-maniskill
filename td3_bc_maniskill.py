from collections import defaultdict
from datetime import datetime
import time
from dataclasses import dataclass
import random
from tqdm import tqdm
import copy

import torchvision.transforms
import tyro
import yaml
import numpy as np
import torch
from torch.utils.data.sampler import RandomSampler, BatchSampler
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from env.make_maniskill_envs_rl import make_eval_envs, evaluate
from datasets.maniskill_datasets import ManiskillDataset
from datasets.maniskill_datasets_tools import IterationBasedBatchSampler, worker_init_fn
from network.td3_bc import BCAgent, Critic
from utils.replay_buffer import ReplayBuffer
from util import kl_divergence, save_ckpt

@dataclass
class Args:
    config_file_path: str = "configs/flow_maniskill_config.yaml"

# 因为各种原因, 可能配置文件不在 configs/ 文件夹内, 因此用终端传入轨迹
args = tyro.cli(Args)
if __name__ == "__main__":
    # ==========> 载入参数
    with open(args.config_file_path, 'r', encoding='utf-8') as file:
        configs = yaml.safe_load(file)  # 使用 safe_load 避免潜在的安全风险

    # ==========> 为每一次运行设置独一无二的名字
    run_name = f"{configs['exp_name']}-" + \
               f"{configs['envs']['env_id']}-" + \
               f"{configs['seed']}-" + \
               f"{datetime.today().strftime('%Y%m%d')}-" + \
               f"{int(time.time())}"
    print("当前的运行名是: ", run_name)

    # ==========> 设置随机数种子
    random.seed(configs["seed"])
    np.random.seed(configs["seed"])
    torch.manual_seed(configs["seed"])
    torch.backends.cudnn.deterministic = configs["torch_deterministic"]

    # ==========> 设置运行设备
    device = torch.device("cuda" if torch.cuda.is_available() and configs["cuda"] else "cpu")
    dtype = torch.float32

    # ==========> 初始化仿真器测试环境
    env_kwargs = dict(
        control_mode=configs["dataset"]["control_mode"],  # 注意: Mani-Skill 的控制模式要和数据集一致!!!
        reward_mode="sparse",
        obs_mode="rgb",
        render_mode="rgb_array"  # 这里不设可调参数, 不是高频调参项, 必要时直接在这里改动即可
    )
    # 自定义每条 episode 的最大步数, 这是因为模仿学习到的技能, 完成任务所需的步数会超过环境最大步数
    if configs["envs"]["max_episode_steps"] is not None:
        env_kwargs["max_episode_steps"] = configs["envs"]["max_episode_steps"]
    other_kwargs = dict()
    envs = make_eval_envs(
        env_id=configs["envs"]["env_id"],
        num_envs=configs["num_eval_envs"],
        sim_backend=configs["sim_backend"],
        env_kwargs=env_kwargs,
        other_kwargs=other_kwargs,
        video_dir=f'runs/{run_name}/videos' if configs["capture_video"] else None
    )

    # ==========> 在配置文件中填充本体数据维度和动作空间维度
    configs["d_proprioception"] = envs.single_observation_space.spaces["agent"]["qpos"].shape[0] + \
                                  envs.single_observation_space.spaces["agent"]["qvel"].shape[0] + 1 + \
                                  envs.single_observation_space.spaces["extra"]["tcp_pose"].shape[0] + \
                                  envs.single_observation_space.spaces["extra"]["goal_pos"].shape[0]
    configs["d_action"] = envs.single_action_space.shape[0]
    print(f"d_proprioception {configs['d_proprioception']}, d_action {configs['d_action']}")

    # ==========> 载入数据集: 数据集预处理函数、数据集类、采样器和生成器
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
        ]
    )
    dataset = ManiskillDataset(
        chunk_size=configs["chunk_size"],  
        num_traj=configs["dataset"]["num_traj"],  # 从数据集中载入多少演示轨迹
        data_path=configs["dataset"]["path"],  # 数据集的文件路径
        device=device,
        dtype=dtype,
        transform=transform,
        args=configs["dataset"]  # 其他参数
    )
    total_iters = configs["total_iters"]
    updates_per_iter = configs.get("updates_per_iter", 1)
    sampler = RandomSampler(dataset, replacement=False)
    # drop_last=True 表示如果数据集划分少于 batch_size, 则丢掉剩余的样本
    batch_sampler = BatchSampler(sampler, batch_size=configs["batch_size"], drop_last=True)
    # 包装这个类, 可以确保数据集的采样过程依赖于 iteration, 而不依赖于数据集的大小...? 我查阅资料是这样看的...
    batch_sampler = IterationBasedBatchSampler(batch_sampler, configs["total_iters"])
    train_dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=configs["dataset"]["num_dataload_workers"],
        worker_init_fn=lambda worker_id: worker_init_fn(worker_id, base_seed=configs["seed"]),
    )
    # configs["num_demos"] = dataset.num_traj

    # ==========> 训练记录器: 本地用 Tensorboard, 云端用 Wandb
    if configs["wandb"]["need"]:
        import wandb

        config = configs
        config["eval_env_cfg"] = dict(
            **env_kwargs,
            num_envs=configs["num_eval_envs"],
            env_id=configs["envs"]["env_id"],
            env_horizon=configs["envs"]["max_episode_steps"]
        )
        wandb.init(
            project=configs["wandb"]["project_name"],
            config=config,
            name=run_name,
            sync_tensorboard=True  # 把 tensorboard 的记录数据传到 wandb 中
        )

    writer = SummaryWriter(f"runs/{run_name}")  # tensorboard 的初始化
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in configs.items()])),
    )
    # 在创建 buffer 之前加：
    cam_space = envs.single_observation_space.spaces["sensor_data"]["base_camera"]["rgb"]  # (H, W, 3)
    H, W = cam_space.shape[0], cam_space.shape[1]
    buffer = ReplayBuffer(capacity=200_000,
                      obs_shape=(3, H, W),
                      state_dim=configs["d_proprioception"],
                      act_shape=(configs["chunk_size"], configs["d_action"]),
                      device="cpu")


    def prefill_buffer_from_dataset(dataset, buffer, k, gamma):
        # 这些字段你在 ManiskillDataset.__init__ 里已经创建过
        traj_states  = dataset.trajectories["state"]     # list[T_i, state_dim] (GPU 或 CPU，看你之前实现)
        traj_rgbs    = dataset.trajectories["rgb"]       # list[T_i, 3, H, W]
        traj_actions = dataset.trajectories["actions"]   # list[T_i, act_dim]

        # 如果有终止/截断/成功标志则取，没有就用简单稀疏奖励占位
        term_list = dataset.trajectories.get("terminated", [None]*len(traj_actions))
        trunc_list = dataset.trajectories.get("truncated",  [None]*len(traj_actions))
        succ_list  = dataset.trajectories.get("success",    [None]*len(traj_actions))

        for i in range(len(traj_actions)):
            acts_i   = traj_actions[i]        # [T, da]
            states_i = traj_states[i]         # [T, sd]
            rgbs_i   = traj_rgbs[i]           # [T, 3, H, W]
            T = acts_i.shape[0]

            # 构造奖励序列：若数据中没有 reward，就用“到末尾给 1 否则 0”的稀疏占位
            r_seq = torch.zeros(T, dtype=acts_i.dtype, device=acts_i.device)
            if succ_list[i] is not None and bool(succ_list[i]):
                # 你的旧逻辑：在 T-2 处给 1
                if T >= 2:
                    r_seq[T-2] = 1.0

            # 若有 term/trunc，可用于 done 计算
            term_i = term_list[i]
            trunc_i = trunc_list[i]

            # 预计算折扣
            gammas = (gamma ** torch.arange(k, device=acts_i.device, dtype=acts_i.dtype))  # [k]

            for t in range(0, T - 1):     # 注意边界：至少要能取到下一帧
                t_end = min(t + k, T - 1)  # 用 t+k 截到最后一帧
                a_chunk = acts_i[t:t_end]  # [<=k, da]
                # pad 到 k（和你 Actor 的 pad 规则保持一致）
                if a_chunk.shape[0] < k:
                    pad = a_chunk[-1].unsqueeze(0).repeat(k - a_chunk.shape[0], 1)
                    a_chunk = torch.cat([a_chunk, pad], dim=0)  # [k, da]

                s_rgb   = rgbs_i[t]                # [3,H,W]
                s_state = states_i[t]              # [sd]
                s2_rgb  = rgbs_i[t_end]            # [3,H,W]
                s2_state= states_i[t_end]          # [sd]

                # done_k：窗口末端是否触到 episode 末尾
                done_k = float(t_end >= T - 1)
                if term_i is not None and trunc_i is not None:
                    # 若你有逐步的 term/trunc，可把窗口内任一步终止视为 done_k=1
                    # done_k = float(torch.any(term_i[t:t_end+1] | trunc_i[t:t_end+1]))

                    # 上面那行需要 term_i/trunc_i 为 tensor；没有就用简单末尾判断
                    pass

                # R_k = sum_{i=0}^{k-1} gamma^i * r_{t+i}（截断到 t_end）
                r_chunk = r_seq[t:t_end]     # [<=k]
                if r_chunk.shape[0] < k:
                    # 末尾补 0
                    r_chunk = torch.cat([r_chunk, torch.zeros(k - r_chunk.shape[0], device=r_chunk.device, dtype=r_chunk.dtype)])
                R_k = torch.dot(r_chunk, gammas).view(1)  # [1]

                buffer.add(s_rgb, s_state, a_chunk, R_k, s2_rgb, s2_state, torch.tensor([done_k], dtype=torch.float32, device=R_k.device))
    # ===== 预填充（把离线数据滑窗后写入 buffer）=====
    k = configs["chunk_size"]
    gamma = configs.get("gamma", 0.99)
    with torch.no_grad():
        prefill_buffer_from_dataset(dataset, buffer, k, gamma)
    print(f"Replay filled: {buffer.size} / {buffer.capacity}")

    # for batch in DataLoader(dataset, batch_size=64, shuffle=False):
    #     rgb_seq   = batch["rgb_seq"]      # [T,3,H,W]
    #     state_seq = batch["state_seq"]    # [T, state_dim]
    #     act_seq   = batch["actions_seq"]  # [T, act_dim]
    #     term = batch["terminated_seq"]    # [T]
    #     trunc = batch["truncated_seq"]    # [T]
    #     succ  = batch["success"]          # bool 或 [T]

    #     T = act_seq.shape[0]
    #     # 简单稀疏奖励：只在最后一步给 1，否则 0
    #     r_seq = torch.zeros(T, dtype=torch.float32, device=act_seq.device)
    #     if bool(succ):
    #         r_seq[T-2] = 1.0  # 你之前的写法（最后一步前一帧）

    #     for t in range(0, T - k):  # 窗口不越界
    #         a_chunk = act_seq[t:t+k]                       # [k, da]
    #         s_rgb   = rgb_seq[t]                           # [3,H,W]
    #         s_state = state_seq[t]                         # [state]
    #         s2_rgb  = rgb_seq[t+k]                         # [3,H,W]
    #         s2_state= state_seq[t+k]
    #         done_k  = bool(torch.any(term[t:t+k] | trunc[t:t+k]))

    #         # k 步折扣回报 R_k = sum_{i=0}^{k-1} gamma^i * r_{t+i}
    #         gammas = (gamma ** torch.arange(k, device=r_seq.device, dtype=r_seq.dtype))
    #         R_k = torch.dot(r_seq[t:t+k], gammas).view(1)  # [1]

    #         buffer.add(s_rgb, s_state, a_chunk, R_k, s2_rgb, s2_state, torch.tensor([done_k], dtype=torch.float32))


    FlowAgent = BCAgent(configs).to(device=device, dtype=dtype)
    FlowAgent_target = copy.deepcopy(FlowAgent).eval()
    critic = Critic(configs).to(device=device, dtype=dtype)
    critic_target = copy.deepcopy(critic).eval()

    # ==========> 优化器设置
    param_dicts = [
        {
            # 如果是 ACT 模型的参数, 则用 "lr" 的学习率
            "params": [p for n, p in FlowAgent.named_parameters() if "backbones" not in n and p.requires_grad]
        },
        {
            # 如果是视觉模型的参数, 则用 "lr" 的学习率
            "params": [p for n, p in FlowAgent.named_parameters() if "backbones" in n and p.requires_grad],
            "lr": configs["lr_backbone"],
        },
    ]
    optimizer = optim.AdamW(param_dicts, lr=configs["lr"], weight_decay=configs["weight_decay"])
    # 当迭代次数达到总迭代数的三分之二时, 学习率开始下降, 这比余弦学习率好多了!

    #critic 没使用lr衰减
    opt_critic = optim.AdamW(critic.parameters(), lr = configs["lr"], weight_decay=configs["weight_decay"])

    lr_drop = int((2 / 3) * configs["total_iters"])
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, lr_drop)

    #RL 相关超参数
    gamma        = configs.get("gamma", 0.99)
    tau          = configs.get("tau", 0.005)
    policy_noise = configs.get("policy_noise", 0.2)
    noise_clip   = configs.get("noise_clip", 0.5)
    policy_delay = configs.get("policy_delay", 2)
    lambda_bc    = configs.get("lambda_bc", 2.5)   # TD3-BC 的 BC 权重
    act_low  = envs.single_action_space.low
    act_high = envs.single_action_space.high

    # ---------------------------------------------------------------------------- #
    # 训练
    # ---------------------------------------------------------------------------- #
    print("训练开始!")
    FlowAgent.train()
    critic.train()

    best_eval_metrics = defaultdict(float)
    timings = defaultdict(float)

    def clip_to_space(a_np):
        # 保护性裁剪，防越界振荡
        return np.clip(a_np, act_low, act_high)
    
    def td3bc_update_step(step_i, B=256, k = configs["chunk_size"]):
        # 取 batch
        rgb, state, act, rew, rgb_next, state_next, done = buffer.sample(B)
        rgb = rgb.to(device, dtype=dtype, non_blocking=True)
        state = state.to(device, dtype=dtype, non_blocking=True)
        act = act.to(device, dtype=dtype, non_blocking=True)
        rgb_next = rgb_next.to(device, dtype=dtype, non_blocking=True)
        state_next = state_next.to(device, dtype=dtype, non_blocking=True)
        rew = rew.to(device, dtype=dtype, non_blocking=True)
        done = done.to(device, dtype=dtype, non_blocking=True)
        obs = {"rgb": rgb, "state": state}
        obs2 = {"rgb": rgb_next, "state": state_next}
        # ---------- Critic 更新 ----------
        with torch.no_grad():
            # target 动作：actor_targ 给单步
            a2_seq = FlowAgent_target.get_action(obs2)    # [B, chunk, da]
            # a2 = a2_seq[:, 0, :]                         # [B, da]
            # policy smoothing (TD3)
            noise = (torch.randn_like(a2_seq) * policy_noise).clamp_(-noise_clip, noise_clip)
            a2 = torch.clamp(a2_seq + noise, torch.as_tensor(act_low, device=device), torch.as_tensor(act_high, device=device))
            q_targ = critic_target(obs2, a2)         # [B,1]
            y = rew + (1.0 - done) * (gamma ** k) * q_targ

        q1= critic(obs, act)                   # act 是数据里的 a_t
        critic_loss = F.mse_loss(q1, y)
        opt_critic.zero_grad()
        critic_loss.backward()
        opt_critic.step()

        # ---------- Actor 更新（delay） ----------
        actor_loss_val = None
        bc_loss_val = None
        q_pi_mean = None
        if step_i % policy_delay == 0:
            bc_loss, a_pi_seq = FlowAgent(obs, act)        # [B, chunk, da]
            # a_pi = a_pi_seq[:, 0, :]                     # [B, da]
            q_pi = critic(obs, a_pi_seq)
            # bc_loss = F.mse_loss(a_pi_seq, act)              # 数据动作 vs actor 动作（单步）
            actor_loss = (-q_pi.mean() + lambda_bc * bc_loss)
            optimizer.zero_grad()
            actor_loss.backward()
            optimizer.step()
            actor_loss_val = actor_loss.item()
            bc_loss_val = bc_loss.item()
            q_pi_mean = q_pi.mean().item()

            # ---------- Polyak ----------
            with torch.no_grad():
                for p, pt in zip(critic.parameters(), critic_target.parameters()):
                    pt.data.mul_(1 - tau).add_(tau * p.data)
                for p, pt in zip(FlowAgent.parameters(), FlowAgent_target.parameters()):
                    pt.data.mul_(1 - tau).add_(tau * p.data)

        return critic_loss.item(), actor_loss_val, bc_loss_val, q1.mean().item(), q_pi_mean

    for cur_iter in tqdm(range(total_iters), desc="TD3-BC Updates"):
        for k in range(updates_per_iter):
            c_loss, a_loss, bc_loss, q1, q_pi = td3bc_update_step(step_i=cur_iter*updates_per_iter + k,
                                            B=configs.get("batch_size_rl", 256))
            writer.add_scalar("td3bc/critic_loss", c_loss, cur_iter*updates_per_iter + k)
            if a_loss is not None:
                writer.add_scalar("td3bc/actor_loss", a_loss, cur_iter*updates_per_iter + k)
            if bc_loss is not None:
                writer.add_scalar("td3bc/bc_loss", bc_loss, cur_iter*updates_per_iter + k)
            writer.add_scalar("td3bc/q1_value", q1, cur_iter*updates_per_iter + k)
            if q_pi is not None:
                writer.add_scalar("td3bc/q_pi_value", q_pi, cur_iter*updates_per_iter + k)

        # 每隔 configs["eval_frequency"] 次迭代进行一次评估
        if cur_iter % configs["eval_frequency"] == 0:
            FlowAgent.eval()
            critic.eval()
            last_tick = time.time()  # 记录评估开始时间
            with torch.no_grad():
                eval_metrics = evaluate(
                    n=configs["num_eval_episodes"],  # 评估时运行的测试 episode 数量
                    sample_fn=lambda obs: FlowAgent.get_action(obs),  # 训练中的模型
                    eval_envs=envs,  # 评估环境
                    device=device,  # 设备 (CPU 或 GPU)
                    dtype=dtype,  # 数据类型 (如 float32 或 float16)
                    transform=transform  # 用于预处理观测数据的变换函数
                )
                # 计算评估所消耗的时间并记录
                timings["eval"] += time.time() - last_tick
                # 打印评估时执行的 episode 数量
                print(f"Evaluated {len(eval_metrics['success_at_end'])} episodes")
                print("ep_return:", eval_metrics["ep_return"])

                # 遍历评估指标, 将每个指标取均值, 并写入日志
                for k in eval_metrics.keys():
                    eval_metrics[k] = np.mean(eval_metrics[k])
                    writer.add_scalar(f"eval/{k}", eval_metrics[k], cur_iter)
                    print(f"{k}: {eval_metrics[k]:.4f}")

                # 需要根据评估结果保存最优模型的指标 (通常是成功率)
                save_on_best_metrics = ["success_once", "success_at_end"]
                for k in save_on_best_metrics:
                    # 如果当前评估指标优于之前的最优值，则更新最优值并保存模型
                    if k in eval_metrics and eval_metrics[k] > best_eval_metrics[k]:
                        best_eval_metrics[k] = eval_metrics[k]
                        save_ckpt(run_name, FlowAgent, f"best_eval_{k}")
                        print(f'New best {k}_rate: {eval_metrics[k]:.4f}. Saving checkpoint.')
            FlowAgent.train()
        if cur_iter % configs["log_frequency"] == 0:
            # 对每次 iteration 做记录
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], cur_iter)
            
            writer.add_scalar("losses/Q", q1, cur_iter)
            for k, v in timings.items():
                writer.add_scalar(f"time/{k}", v, cur_iter)

        # 保存模型权重
        if configs["save_frequency"] is not None and cur_iter % configs["save_frequency"] == 0:
            save_ckpt(run_name, FlowAgent, str(cur_iter))

    envs.close()
    writer.close()