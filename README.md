# 🚪 Reinforcement-Learning Escape Room

An interactive **escape-room game** where an agent must solve five themed rooms,
each one a different Reinforcement-Learning problem and **a different algorithm**.
Every room lets you tune all of the algorithm's hyper-parameters, **train** the
agent, watch **live learning/exploration graphs**, and **replay** episodes from
different stages of training to see what the policy learned.

| Room | Theme | Algorithm | Model | Main task |
|------|-------|-----------|-------|-----------|
| 1 | 🟡 Pacman | **Dynamic Programming** | Known | Collect every coin, then exit |
| 2 | 💎 Museum Heist | **SARSA** (on-policy) | Unknown | Steal the diamond, dodge cameras/traps, escape |
| 3 | 🏎️ Racing | **Q-Learning** (off-policy) | Unknown | Reach the finish fast (boosters, oil, mud, short-cut) |
| 4 | ⚽ Football | **DQN** (function approx.) | Unknown, continuous | Beat defenders + keeper and score |
| 5 | 🚧 Obstacles *(optional)* | **DQN + sensors** | Unknown, continuous | Cross a *random* room avoiding obstacles |

Difficulty increases across the rooms: a known-model planning problem → cautious
on-policy control → aggressive off-policy control → continuous control with a
neural network → a generalising, sensor-based policy tested on unseen rooms.

---

## Running it

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (use: source .venv/bin/activate on macOS/Linux)
pip install -r requirements.txt
streamlit run app.py
```

A browser tab opens with a sidebar room selector. Pick a room → set parameters →
**Train/Solve** → inspect the graphs → use the **replay** controls at the bottom.

> Rooms 1–3 train in seconds. Rooms 4–5 (DQN) train in roughly **1–2 minutes** on
> a CPU; a live progress bar and reward curve are shown while training.

---

## Project structure

```
app.py                  Streamlit entry point + room routing
rl/
  envs/
    grid_base.py        Shared 10x10 geometry + slippery-cell transition model
    pacman.py           Room 1 environment (+ explicit MDP for DP)
    museum.py           Room 2 environment
    racing.py           Room 3 environment
    football.py         Room 4 continuous environment
    obstacles.py        Room 5 continuous environment (dynamic layouts + sensors)
  algos/
    dp.py               Value iteration & policy iteration
    td_core.py          Shared tabular TD control
    sarsa.py            SARSA  (on-policy)  wrapper
    qlearning.py        Q-Learning (off-policy) wrapper
    dqn.py              Deep Q-Network (replay buffer + target network, PyTorch)
  utils.py              Seeding, smoothing, TrainResult (metrics + replay snapshots)
ui/
  render.py             Grid / pitch rendering + metrics dataframe
  common.py             Live-training progress, learning graphs, replay player
  room1..room5_*.py     One Streamlit page per room
```

The three grid rooms share one **slippery-cell model** (`grid_base.py`): on an icy
cell the chosen action is, with probability *slip*, replaced by a uniformly random
direction. Dynamic Programming reads the *full* distribution
(`cell_transitions`); SARSA/Q-Learning only *sample* it (`sample_cell`), so a
policy found by DP is directly comparable to a model-free one.

---

## Room 1 — Pacman · Dynamic Programming

**Idea.** The maze model is fully known, so we compute the optimal policy exactly
with value/policy iteration — no exploration needed.

* **State:** `(cell, coin_mask)` — position `(row, col)` plus a bitmask of the
  coins still on the board. With 4 coins on a 10×10 maze ≈ **1,072 reachable
  states**. The single **final state** is standing on the door with all coins
  collected.
* **Actions:** Up / Down / Left / Right (blocked moves keep you in place).
* **Dynamics:** several **icy cells** randomise the move (slip). While any coin
  remains the **door is locked** — trying it bumps you back.
* **Rewards:** `+10` per coin · `+100` exit · `−1` per step · `−10` trying the
  locked door · `−5` on a slip.

**Best parameters.** `γ = 0.95`, convergence `θ = 1e-4`, `slip = 0.15`.
Value iteration converges in **~180 sweeps**; the optimal greedy policy escapes in
**~38 steps** (return ≈ +102). Lower γ makes the agent short-sighted (it may
ignore distant coins relative to the step cost); higher slip lowers the start-state
value because detours around ice become worthwhile.

**Graphs.** Bellman residual per sweep (convergence), start-state value per sweep,
and policy-changes per sweep (→0 when stable). **Replay** shows the greedy policy
after 1 sweep, after 5 sweeps, and at convergence — you watch the plan improve.

---

## Room 2 — Museum Heist · SARSA

**Idea.** A **cliff-walking** problem so the on-policy nature of SARSA is visible.
A sidebar selector offers two layouts:

* **Museum maze (hard, default):** a comb-maze of walls. The diamond is on the
  left edge; the *short* way out (~11 steps) runs east along an **icy edge straight
  into camera cells**, while the only **camera-free** escape is a long detour up
  through the maze and down the far side (~27 steps).
* **Open hall (easy):** an open room with a single camera cliff along the bottom.

In both, stepping into a camera trips the alarm and the thief is dragged back to
the start.

* **State:** `(cell, has_diamond)` — ≤200 states (the maze walls reduce this).
  **Final state:** reaching the exit *with* the diamond.
* **Actions:** Up / Down / Left / Right.
* **Dynamics:** cameras (cliff), trap(s), and icy cells; the exit is locked until
  the diamond is taken.
* **Rewards:** `+30` diamond · `+100` exit · `−1` step · `−25` camera (alarm) ·
  `−20` trap · `−5` slip · `−5` trying to exit without the loot.

**Best parameters.**
*Hard maze:* `episodes = 1500`, `α = 0.2`, `γ = 0.97`, `ε: 1.0 → 0.02`
(decay 0.997 / episode), `slip = 0.05` → **100 % success**; the greedy policy takes
the **27-step camera-free detour with zero camera hits** — it deliberately avoids
the shorter icy edge route.
*Easy hall:* `episodes = 800`, `γ = 0.95`, decay `0.99` → escapes in ~15 steps,
keeping a safety margin from the cliff.

**What it tests.** Learning a *safe* path, not just the shortest one — SARSA
evaluates the ε-greedy policy it actually follows and therefore "fears" its own
exploration near the cliff. This is the natural contrast with Room 3's Q-Learning,
which would hug the short, risky edge.

---

## Room 3 — Racing · Q-Learning

**Idea.** A winding **serpentine circuit**. The safe racing line snakes the whole
track (~54 steps); a tempting **short-cut** drops straight down the middle column
through **oil** (slippery, costly) and **mud** (slow) and reaches the finish in
~18 steps. Off-policy Q-Learning chases the highest-value line.

* **State:** `(cell, booster_mask)` — position plus which one-off **boosters** have
  been taken (so a booster can't be farmed). **Final state:** crossing the finish.
* **Actions:** Up / Down / Left / Right.
* **Rewards:** `+100` finish · `+15` booster (once) · `−1` step · `−10` mud ·
  `−20` slipping on oil · `−30` leaving the track (driving into the barrier).

**Best parameters.** `episodes = 1500`, `α = 0.2`, `γ = 0.97`, `ε: 1.0 → 0.05`
(decay 0.997), `oil slip = 0.2`. → **100 % success**, an ~18-step short-cut lap.
**Raise the oil-slip probability or the oil penalty** and the short-cut stops being
worth it — the agent grows cautious and takes the long serpentine.

**Bonus — watch both models race.** Tick *"Also train SARSA for comparison"* to
train SARSA on the same track. You then get (a) overlaid reward/steps curves and
(b) a **🏁 Race**: both learned greedy policies run **together** on one track —
🔴 Q-Learning vs 🔵 SARSA — so you can watch their racing lines and see who finishes
first.

---

## Room 4 — Football Final Shot · DQN

**Idea.** A continuous 10×10 m pitch. A **Q-table is impractical** for real-valued
state, so a small MLP approximates `Q(s, a)`.

* **Core state:** `(x, y, vₓ, v_y)`; `dt = 0.02 s`, velocity components discrete in
  `{−1, 0, 1} m/s`. The agent commits to a velocity for `action_repeat` ticks
  (frame-skip = 10), so one decision moves it ~0.2 m.
* **Observation (network input):** the core state **plus** the keeper's position +
  **patrol direction** and each defender's position **relative** to the player
  (10 base features + 3 per defender).
* **The kick is real.** On a shoot action the **ball leaves the player** and flies
  on its own (simulated at the 0.02 s tick): you choose the **power** (soft/hard →
  ball speed) and the **curve** (bend left / straight / right → a Magnus-style
  sideways acceleration). The replay animates the ball flight.
* **Actions (11):** Up · Down · Left · Right · Stay · {soft, hard} × {curve-left,
  straight, curve-right}. A kick **inside** the area ends the episode (goal / save /
  miss); a kick from **outside** is a wasted touch (small penalty, play continues).
* **Defenders** (configurable count, **random start positions every game**) chase the
  player; the **keeper patrols side to side** across the goal mouth. Score by timing
  the shot for when the keeper has drifted off your line, or by bending the ball
  around him.
* **Rewards:** `+300` goal · `−10` saved (on target but stopped) · `−30` missed the
  goal (wide) · `+10` entering the area · `−1` per step **dwelling** in the area (so
  the agent shoots promptly) · `−25` shot-clock timeout · `+20` dodging a defender ·
  `−50` tackled · `−30` out of bounds · `−5` wasted shot · `−0.3` step, **plus
  distance-to-goal shaping** (`3 ×` metres gained). The save < miss gap deliberately
  rewards getting shots *on target* first, then learning to beat the keeper.
* **Final state:** the ball in the net.

**Best parameters.** `episodes ≈ 600–800`, hidden `(128, 128)`, `lr = 1e-3`,
`γ = 0.99`, batch `64`, `ε → 0.05` over 50 % of episodes, target-net update every
`500` steps, `learn_start = 1500`. Typical learned **greedy** result with 2–3
defenders: **~40 % goals**, the rest split between keeper saves and being tackled
while dribbling in — a genuinely hard control task. **Difficulty knobs:** number of
**defenders** (≥1; they create the urgency that makes the agent commit to a shot),
**defender speed**, and **keeper patrol speed**. The agent learns to dribble into
the area, line up inside the mouth, wait for the keeper to drift off its line, and
finish (sometimes bending the ball with curve).

> Design notes: the right edge is the goal line (clamped, not "out"), so the only
> positive ending is a real shot; outside-area shots are non-terminal so random
> exploration isn't forced to end episodes early. These two choices were essential
> to make the sparse goal reward discoverable.

---

## Room 5 *(optional)* — Dynamic Obstacles · DQN + sensors

**Idea.** Cross to the exit while avoiding circular **obstacles** (width 0.5 m). A
**new random layout every episode** means the agent can't memorise a map — it must
learn a **reactive** policy from local **sensors**, which is exactly what lets the
trained policy be dropped into a **brand-new random room** at the end.

* **State:** `(x, y, vₓ, v_y)` (continuous, same physics as Room 4, 5 move actions).
* **Observation:** own dynamics + direction to exit + the nearest obstacles whose
  **centre** is within **`sensor_range` metres** (the "see X metres ahead" control),
  each as a relative position + closeness. Empty slots read as "clear".
  `obs_dim = 6 + 3 × (obstacles sensed)`.
* **Rewards:** `+200` reaching the exit · `−100` hitting an obstacle · `−30` out of
  bounds · `−0.5` grazing (within 0.6 m of an obstacle) · `−0.1` step, plus
  rightward progress shaping. **Final state:** crossing the exit line.

**Best parameters.** `episodes ≈ 600`, hidden `(128, 128)`, `lr = 1e-3`, `γ = 0.99`,
`sensor_range = 3.0 m`, `6 obstacles`, frame-skip 10. **Shrink the sensor range**
to make the agent short-sighted (much harder); **add obstacles** to crowd the room.
After training, hit **"Generate & test"** to spawn an unseen room and replay the
learned policy in it.

**What it tests.** Generalisation via local observation + function approximation —
the agent solves rooms it has never seen.

---

## Graphs & replay (every room)

* **Learning graphs:** episode reward (raw + smoothed), steps-to-finish, rolling
  success rate, ε-decay (exploration), plus algorithm-specific series
  (DP convergence residual; mean |TD-error|; DQN training loss).
* **Replay:** snapshots are recorded at several training stages (DP: 1/5/converged
  sweeps; TD/DQN: *Early / Mid / Final* greedy rollouts). Scrub with the step slider
  or hit **▶ Animate** to watch the agent move, with per-step and cumulative reward.

## Notes on the algorithms

* **DP** uses the known model directly — optimal, but only feasible because the
  state space is small and enumerable.
* **SARSA vs Q-Learning** differ in a single line (the TD target) — implemented once
  in `td_core.py`. SARSA is the cautious one (Room 2's cliff), Q-Learning the
  aggressive one (Room 3's short-cut).
* **DQN** adds experience replay + a target network for stable function
  approximation, with ε-greedy exploration decayed per episode.
