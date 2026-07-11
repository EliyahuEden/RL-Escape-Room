"""Tabular temporal-difference control shared by SARSA (Room 2) and
Q-Learning (Room 3).

The two algorithms differ in a single line — the TD *target*:

* **SARSA** (on-policy):    target = r + γ · Q(s', a')   with a' ~ ε-greedy
* **Q-Learning** (off-policy): target = r + γ · max_a Q(s', a)

Everything else (ε-greedy behaviour, ε decay, logging, replay snapshots) is
identical, so it lives here once.  The environment only needs ``reset()``,
``step(a) -> (state, reward, done, info)``, ``n_actions`` and a
``render_state()`` dict for replay.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Callable, Dict, List, Optional

import numpy as np

from rl.utils import TrainResult


def epsilon_at(ep: int, eps_start: float, eps_end: float, eps_decay: float) -> float:
    return max(eps_end, eps_start * (eps_decay ** ep))


def eps_greedy(q_values: np.ndarray, eps: float, rng: random.Random) -> int:
    if rng.random() < eps:
        return rng.randrange(len(q_values))
    # argmax with random tie-breaking
    best = float(np.max(q_values))
    candidates = [i for i, v in enumerate(q_values) if v == best]
    return rng.choice(candidates)


def greedy_rollout(env, Q: Dict, n_actions: int, max_steps: int,
                   seed: int = 0) -> List[dict]:
    """Run one fully-greedy episode and record frames for replay."""
    saved_rng = env.rng
    env.rng = random.Random(seed)
    try:
        state = env.reset()
        frames = [dict(env.render_state(), step=0, reward=0.0, cum_reward=0.0,
                       done=False)]
        cum, done, steps = 0.0, False, 0
        while not done and steps < max_steps:
            q = Q[state] if state in Q else np.zeros(n_actions)
            action = int(np.argmax(q))
            state, reward, done, info = env.step(action)
            cum += reward
            steps += 1
            frames.append(dict(env.render_state(), step=steps, reward=reward,
                               cum_reward=cum, done=done, **info))
        return frames
    finally:
        env.rng = saved_rng


def train(
    env,
    *,
    algo: str = "sarsa",
    episodes: int = 800,
    alpha: float = 0.1,
    gamma: float = 0.95,
    eps_start: float = 1.0,
    eps_end: float = 0.05,
    eps_decay: float = 0.995,
    max_steps: int = 200,
    seed: Optional[int] = 0,
    progress_cb: Optional[Callable] = None,
    record_replays: bool = True,
    snapshot_fracs=(0.05, 0.5, 1.0),
    snapshot_labels=("Early", "Mid", "Final"),
) -> TrainResult:
    """Train SARSA or Q-Learning. ``algo`` ∈ {"sarsa", "qlearning"}."""
    assert algo in ("sarsa", "qlearning")
    rng = random.Random(seed)
    nA = env.n_actions
    Q: Dict = defaultdict(lambda: np.zeros(nA))
    result = TrainResult()
    result.extra["mean |TD error|"] = []

    # per-episode counters pulled from the env's ``info`` dict (only the ones that
    # actually occur are surfaced as learning graphs at the end)
    count_labels = {
        "camera": "Camera detections / episode",
        "caught": "Times caught by guards / episode",
        "trap": "Trap hits / episode",
        "crash": "Crashes / episode",
        "shortcut": "Shortcut tiles used / episode",
    }
    count_series: Dict[str, List[float]] = {k: [] for k in count_labels}

    milestones = {}
    for frac, label in zip(snapshot_fracs, snapshot_labels):
        milestones[max(1, int(round(frac * episodes)))] = label

    for ep in range(episodes):
        eps = epsilon_at(ep, eps_start, eps_end, eps_decay)
        state = env.reset()
        action = eps_greedy(Q[state], eps, rng)
        done, total, steps, td_abs = False, 0.0, 0, 0.0
        info = {}
        ep_counts = {k: 0 for k in count_labels}
        frames = []
        if record_replays:
            frames.append(dict(env.render_state(), step=0, reward=0.0,
                               cum_reward=0.0, done=False, epsilon=eps))
        while not done and steps < max_steps:
            action_taken = action
            nstate, reward, done, info = env.step(action)
            for k in count_labels:
                if info.get(k):
                    ep_counts[k] += 1
            # ── The ONE line that separates the two algorithms: the TD target ──
            if algo == "sarsa":
                # SARSA — ON-policy: bootstrap from the action we ACTUALLY take
                # next (a' ~ ε-greedy), so the cost of future exploration is
                # priced into Q → it learns a cautious policy.
                naction = eps_greedy(Q[nstate], eps, rng)
                target = reward + (0.0 if done else gamma * Q[nstate][naction])
            else:  # qlearning
                # Q-LEARNING — OFF-policy: bootstrap from the BEST next action
                # (max_a Q), ignoring our own exploration, so it learns the
                # optimal greedy policy no matter how we behave.
                target = reward + (0.0 if done else gamma * float(np.max(Q[nstate])))
                naction = eps_greedy(Q[nstate], eps, rng)
            # TD update: nudge Q(s,a) a fraction α toward the target (= TD error)
            td = target - Q[state][action]
            Q[state][action] += alpha * td
            td_abs += abs(td)
            state, action = nstate, naction
            total += reward
            steps += 1
            if record_replays:
                frames.append(dict(env.render_state(), step=steps, reward=reward,
                                   cum_reward=total, done=done, action=action_taken,
                                   epsilon=eps, **info))

        result.log_episode(total, steps, bool(info.get("success", False)), eps)
        if record_replays:
            result.episode_replays.append(frames)
        result.extra["mean |TD error|"].append(td_abs / max(steps, 1))
        for k in count_labels:
            count_series[k].append(ep_counts[k])

        if (ep + 1) in milestones:
            label = milestones[ep + 1]
            result.snapshots[label] = greedy_rollout(env, Q, nA, max_steps)

        if progress_cb is not None:
            progress_cb(ep + 1, episodes, result)

    # surface only the counters that actually occurred during training
    for key, label in count_labels.items():
        if any(count_series[key]):
            result.extra[label] = count_series[key]

    result.policy = {s: q.copy() for s, q in Q.items()}
    result.info["n_states_seen"] = len(Q)
    return result
