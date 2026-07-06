"""Function approximation for the continuous rooms (4 & 5).

The continuous state spaces make a Q-table impossible, so Q(s, a) is
approximated with a small MLP trained by DQN (experience replay + target
network) — see :mod:`rl.algos.dqn`.  This module re-exports the
approximator itself for clarity.
"""
from rl.algos.dqn import MLP as QNetwork  # noqa: F401
from rl.algos.dqn import ReplayBuffer  # noqa: F401
