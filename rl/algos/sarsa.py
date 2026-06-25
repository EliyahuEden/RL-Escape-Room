"""SARSA (on-policy TD control) — used in Room 2.

Thin wrapper over :func:`rl.algos.td_core.train` with ``algo="sarsa"``.
"""

from __future__ import annotations

from functools import partial

from rl.algos.td_core import train as _train

train = partial(_train, algo="sarsa")
