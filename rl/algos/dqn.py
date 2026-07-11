"""Deep Q-Network (DQN) with experience replay and a target network.

Used for the two continuous rooms (Room 4 Football, Room 5 Cross the Road), where the
state is real-valued and a Q-table is impractical — so a small MLP approximates
the action-value function Q(s, a).

The environment must expose:
    obs_dim, n_actions          -- observation size and discrete action count
    reset() -> obs (np.ndarray)
    step(action) -> (obs, reward, done, info)   info may contain "success"
    render_state() -> dict      -- positions etc. for replay rendering

:func:`train` returns a :class:`~rl.utils.TrainResult` with per-episode reward /
steps / success / ε plus a training-loss series, and greedy replay snapshots
recorded at several stages of training.
"""

from __future__ import annotations

import random
from collections import deque
from typing import Callable, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rl.utils import TrainResult


class MLP(nn.Module):
    def __init__(self, in_dim: int, n_actions: int, hidden: Sequence[int] = (128, 128)):
        super().__init__()
        layers, prev = [], in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.size = 0
        self.ptr = 0

    def add(self, o, a, r, no, d):
        i = self.ptr
        self.obs[i], self.next_obs[i] = o, no
        self.actions[i], self.rewards[i], self.dones[i] = a, r, float(d)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (self.obs[idx], self.actions[idx], self.rewards[idx],
                self.next_obs[idx], self.dones[idx])

    def __len__(self):
        return self.size


def greedy_rollout(env, model: nn.Module, max_steps: int, device="cpu") -> List[dict]:
    """One fully-greedy episode, recording render frames for replay."""
    model.eval()
    obs = env.reset()
    frames = [dict(env.render_state(), step=0, reward=0.0, cum_reward=0.0, done=False)]
    cum, done, steps = 0.0, False, 0
    with torch.no_grad():
        while not done and steps < max_steps:
            q = model(torch.as_tensor(obs, dtype=torch.float32, device=device))
            action = int(torch.argmax(q).item())
            obs, reward, done, info = env.step(action)
            cum += reward
            steps += 1
            frames.append(dict(env.render_state(), step=steps, reward=reward,
                               cum_reward=cum, done=done, **info))
    return frames


def train(
    env,
    *,
    episodes: int = 400,
    hidden: Sequence[int] = (128, 128),
    lr: float = 1e-3,
    gamma: float = 0.99,
    batch_size: int = 64,
    buffer_size: int = 50_000,
    eps_start: float = 1.0,
    eps_end: float = 0.05,
    exploration_fraction: float = 0.5,
    learn_start: int = 1000,
    target_update: int = 500,
    train_freq: int = 1,
    grad_clip: float = 10.0,
    max_steps: int = 400,
    seed: Optional[int] = 0,
    device: str = "cpu",
    progress_cb: Optional[Callable] = None,
    record_replays: bool = True,
    snapshot_fracs=(0.1, 0.5, 1.0),
    snapshot_labels=("Early", "Mid", "Final"),
) -> TrainResult:
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
    rng = random.Random(seed)

    obs_dim, nA = env.obs_dim, env.n_actions
    online = MLP(obs_dim, nA, hidden).to(device)
    target = MLP(obs_dim, nA, hidden).to(device)
    target.load_state_dict(online.state_dict())
    target.eval()
    opt = torch.optim.Adam(online.parameters(), lr=lr)
    buffer = ReplayBuffer(buffer_size, obs_dim)

    result = TrainResult()
    result.extra["training loss"] = []

    decay_episodes = max(1, int(episodes * exploration_fraction))
    global_step = 0

    milestones = {}
    for frac, label in zip(snapshot_fracs, snapshot_labels):
        milestones[max(1, int(round(frac * episodes)))] = label

    for ep in range(episodes):
        obs = env.reset()
        done, total, steps, loss_accum, n_updates = False, 0.0, 0, 0.0, 0
        info = {}
        # ε decays per episode, so very short episodes don't stall exploration
        eps = max(eps_end, eps_start - (eps_start - eps_end) * ep / decay_episodes)
        frames = []
        if record_replays:
            frames.append(dict(env.render_state(), step=0, reward=0.0,
                               cum_reward=0.0, done=False, epsilon=eps))
        while not done and steps < max_steps:
            if rng.random() < eps:
                action = rng.randrange(nA)
            else:
                with torch.no_grad():
                    q = online(torch.as_tensor(obs, dtype=torch.float32, device=device))
                    action = int(torch.argmax(q).item())
            nobs, reward, done, info = env.step(action)
            buffer.add(obs, action, reward, nobs, done)
            obs = nobs
            total += reward
            steps += 1
            global_step += 1
            if record_replays:
                frames.append(dict(env.render_state(), step=steps, reward=reward,
                                   cum_reward=total, done=done, action=action,
                                   epsilon=eps, **info))

            if len(buffer) >= learn_start and global_step % train_freq == 0:
                bo, ba, br, bno, bd = buffer.sample(batch_size)
                bo = torch.as_tensor(bo, device=device)
                bno = torch.as_tensor(bno, device=device)
                ba = torch.as_tensor(ba, device=device)
                br = torch.as_tensor(br, device=device)
                bd = torch.as_tensor(bd, device=device)
                # predicted Q(s,a) from the ONLINE net for the actions taken
                q_sa = online(bo).gather(1, ba.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    # TD target r + γ·max_a Q(s',a) from the FROZEN target net
                    # (a slowly-synced copy → stable targets); (1 − done) zeroes
                    # the bootstrap on terminal transitions.
                    q_next = target(bno).max(1).values
                    tgt = br + gamma * q_next * (1.0 - bd)
                # Huber (smooth-L1) loss: robust to the occasional large TD error
                loss = F.smooth_l1_loss(q_sa, tgt)
                opt.zero_grad()
                loss.backward()
                if grad_clip:
                    nn.utils.clip_grad_norm_(online.parameters(), grad_clip)
                opt.step()
                loss_accum += float(loss.item())
                n_updates += 1
                if global_step % target_update == 0:
                    target.load_state_dict(online.state_dict())

        result.log_episode(total, steps, bool(info.get("success", False)), eps)
        if record_replays:
            result.episode_replays.append(frames)
        result.extra["training loss"].append(loss_accum / max(n_updates, 1))

        if (ep + 1) in milestones:
            result.snapshots[milestones[ep + 1]] = greedy_rollout(env, online, max_steps, device)

        if progress_cb is not None:
            progress_cb(ep + 1, episodes, result)

    result.policy = online
    result.info["obs_dim"] = obs_dim
    return result
