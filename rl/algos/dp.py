"""Dynamic Programming for Room 1 (known model).

The room exposes its environment as a :class:`TabularMDP` — the full set of
states, actions and the transition/reward model ``P[s][a] = [(prob, next_state,
reward, terminal), ...]``.  Because the model is known, we can compute the
*optimal* value function and policy directly with **value iteration** or
**policy iteration**, with no exploration at all.

Both solvers return the value function, the greedy policy, and a dict of
per-iteration diagnostic series used for the "learning"/convergence graphs:

* ``value_delta``    -- max change in V over the sweep (Bellman residual);
* ``start_value``    -- estimated value of the start state each iteration;
* ``policy_changes`` -- number of states whose greedy action changed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Hashable, List, Optional, Tuple

State = Hashable
Transition = Tuple[float, State, float, bool]  # prob, next_state, reward, terminal


@dataclass
class TabularMDP:
    states: List[State]
    actions: Tuple[int, ...]
    P: Dict[State, Dict[int, List[Transition]]]
    start: State
    terminals: set = field(default_factory=set)

    def is_terminal(self, s: State) -> bool:
        return s in self.terminals


def _q_value(mdp: TabularMDP, V: Dict[State, float], s: State, a: int, gamma: float) -> float:
    total = 0.0
    for prob, ns, reward, terminal in mdp.P[s][a]:
        total += prob * (reward + (0.0 if terminal else gamma * V[ns]))
    return total


def greedy_policy(mdp: TabularMDP, V: Dict[State, float], gamma: float) -> Dict[State, int]:
    policy: Dict[State, int] = {}
    for s in mdp.states:
        if mdp.is_terminal(s):
            policy[s] = mdp.actions[0]
            continue
        policy[s] = max(mdp.actions, key=lambda a: _q_value(mdp, V, s, a, gamma))
    return policy


def value_iteration(mdp: TabularMDP, gamma: float = 0.95, theta: float = 1e-4,
                    max_iter: int = 1000,
                    iteration_cb: Optional[Callable[[int, Dict[State, float], Dict[State, int]], None]] = None):
    """Classic value iteration. Returns ``(V, policy, diagnostics)``."""
    V = {s: 0.0 for s in mdp.states}
    diag = {"value_delta": [], "start_value": [], "policy_changes": []}
    prev_policy = None
    for _ in range(max_iter):
        delta = 0.0
        for s in mdp.states:
            if mdp.is_terminal(s):
                continue
            old = V[s]
            V[s] = max(_q_value(mdp, V, s, a, gamma) for a in mdp.actions)
            delta = max(delta, abs(old - V[s]))
        policy = greedy_policy(mdp, V, gamma)
        diag["value_delta"].append(delta)
        diag["start_value"].append(V[mdp.start])
        if prev_policy is not None:
            diag["policy_changes"].append(
                sum(1 for s in mdp.states if policy[s] != prev_policy[s])
            )
        if iteration_cb is not None:
            iteration_cb(len(diag["value_delta"]), V, policy)
        prev_policy = policy
        if delta < theta:
            break
    return V, greedy_policy(mdp, V, gamma), diag


def policy_evaluation(mdp: TabularMDP, policy: Dict[State, int], gamma: float,
                      theta: float, max_iter: int = 1000) -> Dict[State, float]:
    V = {s: 0.0 for s in mdp.states}
    for _ in range(max_iter):
        delta = 0.0
        for s in mdp.states:
            if mdp.is_terminal(s):
                continue
            old = V[s]
            V[s] = _q_value(mdp, V, s, policy[s], gamma)
            delta = max(delta, abs(old - V[s]))
        if delta < theta:
            break
    return V


def policy_iteration(mdp: TabularMDP, gamma: float = 0.95, theta: float = 1e-4,
                     max_iter: int = 200,
                     iteration_cb: Optional[Callable[[int, Dict[State, float], Dict[State, int]], None]] = None):
    """Policy iteration (evaluation + greedy improvement). Returns ``(V, policy, diag)``."""
    import random

    policy = {s: random.choice(mdp.actions) for s in mdp.states}
    diag = {"value_delta": [], "start_value": [], "policy_changes": []}
    V = {s: 0.0 for s in mdp.states}
    for _ in range(max_iter):
        V = policy_evaluation(mdp, policy, gamma, theta)
        new_policy = greedy_policy(mdp, V, gamma)
        changes = sum(1 for s in mdp.states if new_policy[s] != policy[s])
        diag["policy_changes"].append(changes)
        diag["start_value"].append(V[mdp.start])
        diag["value_delta"].append(float(changes))  # proxy residual: 0 when stable
        policy = new_policy
        if iteration_cb is not None:
            iteration_cb(len(diag["value_delta"]), V, policy)
        if changes == 0:
            break
    return V, policy, diag
