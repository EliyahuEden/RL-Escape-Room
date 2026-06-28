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
(or oily) cell the move is, with probability *slip*, deflected **sideways** —
half the slip mass to each perpendicular direction. So *slip = 0.2* gives
**80 % intended / 10 % left / 10 % right** (Room 1 ice) and *slip = 0.3* gives
**70 / 15 / 15** (Room 3 oil). Dynamic Programming reads the *full* distribution
(`cell_transitions`); SARSA/Q-Learning only *sample* it (`sample_cell`). A blocked
move also reports a wall-hit so the rooms can penalise it.

---

## Room 1 — Pacman · Dynamic Programming

**Idea.** The maze model is fully known, so we compute the optimal policy exactly
with value/policy iteration — no exploration needed.

* **State:** `(cell, coin_mask, guard_t)` — position, a bitmask of the coins still
  on the board, and the **guard's patrol step**. (Without the guard it's just
  `(cell, coin_mask)`.) The **final state** is standing on the door with all coins
  collected; landing on the guard goes to a single absorbing *caught* state.
* **Actions:** Up / Down / Left / Right (blocked moves keep you in place).
* **Dynamics:** several **icy cells** deflect the move sideways (80/10/10 at
  *slip = 0.2*); the **door is locked** until all coins are collected. A **guard**
  walks a **fixed, fully-known back-and-forth patrol** — so it stays compatible with
  Dynamic Programming (`guard_t` indexes the patrol; its position is a deterministic
  function of the step). Catching the agent ends the episode.
* **Rewards:** `+10` per coin · `+100` exit · `−1` per step · `−10` trying the
  locked door · `−50` caught by the guard · `−5` slip · `−5` hitting a wall.

**Best parameters.** `γ = 0.95`, `θ = 1e-4`, `slip = 0.2`, 3 coins + patrol guard
→ ≈ **2,100 states**, value iteration converges in ~170 sweeps and the optimal
greedy policy collects the coins, dodges the patrol and escapes (≈ 36 steps).
Fewer coins / shorter patrol → fewer states; the app caps interactive DP at 80 000
states.

**Graphs.** Bellman residual per sweep (convergence), start-state value per sweep,
policy-changes per sweep, a **final value heatmap + greedy-policy arrows**, and a
**replay** of the greedy rollout after each DP sweep — you watch the plan improve.

---

## Room 2 — Museum Heist · SARSA

**Idea.** A thief must steal the **diamond** and reach the **exit**, choosing
between a **short dangerous route** (past cameras and patrol guards) and a
**longer safe one**. The model is **unknown** — SARSA learns purely from
experience. A sidebar lets you use a **randomly generated** museum (cameras,
traps, icy tiles, patrol guards) or a fixed layout. Every generated map is
**BFS-verified** before use: it always has a valid Start→Diamond→Exit path, the
**short route passes through the danger** (a wall of cameras + a trap, guards
nearby), and a **danger-free safe route exists and is meaningfully longer** —
guaranteeing the cautious-vs-greedy trade-off SARSA is meant to expose.

* **State:** `(cell, has_diamond)`, plus `guard_t` (the guards' shared patrol
  step) when guards are present, so the agent can learn their timing.
  **Final state:** reaching the exit *with* the diamond.
* **Actions:** Up / Down / Left / Right.
* **Cameras** are **vision zones**: entering one costs a penalty but does **not**
  end the episode (you keep moving). **Patrol guards** walk fixed routes — being
  **caught ends the heist** with a big penalty. The exit is locked without the loot.
* **Rewards:** `+30` diamond · `+120` escape with the diamond · `−1` step ·
  `−25` per step inside a camera zone · `−50` caught by a guard (terminal) ·
  `−20` trap · `−10` trying to exit without the diamond · `−5` slip · `−5` wall.

**Best parameters.** `episodes = 1500`, `α = 0.2`, `γ = 0.97`, `ε: 1.0 → 0.02`
(decay 0.997 / episode), `slip = 0.05` → **100 % success**; the greedy policy
steals the diamond and escapes (~14 steps) while keeping camera exposure and guard
risk to a minimum.

**Graphs.** Reward + moving-average, steps, rolling success rate, ε-decay, **camera
detections per episode**, **times caught by guards**, and trap hits — plus a replay
of **any** training episode (including exploratory failures).

**What it tests.** Learning a *safe* path, not just the shortest one — SARSA
evaluates the ε-greedy policy it actually follows and therefore avoids routes where
its own exploration could walk it into a guard. The natural contrast with Room 3's
Q-Learning, which hugs the short, risky line.

---

## Room 3 — Racing · Q-Learning

**Idea.** A **randomly generated track** (or a fixed one). Every generated map is
**BFS-verified** to provide a longer **safe route** *and* a shorter **oily
short-cut**: the short-cut's oil cells are guaranteed crash-risky (flanked by
walls or crash zones ✕), a **boost** sits just past it, and **mud** slows the safe
route — so the shortcut is meaningfully shorter but a slip there can fling the car
into a wall/crash zone and **end the run**. Off-policy Q-Learning chases the
highest-value line, so it's the algorithm that decides whether the gamble pays off.

* **State:** `(cell, booster_mask)` — position plus which one-off **boosters** have
  been taken (so a booster can't be farmed). **Final state:** crossing the finish.
* **Actions:** Up / Down / Left / Right.
* **Crash rule:** a slip (oil, 70/15/15) that throws the car into a wall/edge — or
  driving into a crash zone — **ends the episode**. A normal blocked move just bumps
  (wall) or nudges the edge (leave-track), staying on the track.
* **Rewards:** `+150` finish · `+20` booster (once) · `−1` step · `−5` mud ·
  `−100` crash (terminal) · `−30` leaving the track edge · `−5` hitting a wall.

**Best parameters.** `episodes = 2000`, `α = 0.2`, `γ = 0.97`, `ε: 1.0 → 0.05`
(decay 0.997), `oil slip = 0.3` → ~**87 % success** with a **~13 % crash rate**:
Q-Learning takes the short-cut and crashes sometimes. **Lower the slip** to make the
short-cut safer (more shortcut use), **raise it** and the agent learns to play safe
on the long serpentine.

**Graphs.** Reward + moving-average, steps-to-finish, rolling success rate, ε-decay,
**crashes per episode (crash rate)**, **short-cut tiles used per episode**, plus a
replay of any training episode. Tick *"Also train SARSA for comparison"* for overlaid
curves and a **🏁 race** of both greedy policies on the same track.

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
* **Actions (15):** 9 moves — Up · Down · Left · Right · Stay · the 4 **diagonals**
  (unit-normalised so diagonals aren't faster) — plus 6 kicks = {soft, hard} ×
  {curve-left, straight, curve-right}. A kick **inside** the area ends the episode
  (goal / save / miss); a kick from **outside** is a wasted touch (small penalty,
  play continues). Defenders and the keeper already move in continuous directions.
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
