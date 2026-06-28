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
| 3 | 🏎️ Street Race | **Q-Learning** (off-policy) | Unknown | Race through oil, mud, boosters & crash barriers |
| 4 | ⚽ Football | **DQN** (function approx.) | Unknown, continuous | Beat defenders + keeper and score |
| 5 | 🐔 Cross the Road *(optional)* | **DQN + sensors** | Unknown, continuous | Cross moving traffic as a chicken |

Difficulty increases across the rooms: a known-model planning problem → cautious
on-policy control → aggressive off-policy control → continuous control with a
neural network → a generalising, sensor-based policy tested on unseen rooms.

---

## Running it

```bash
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
    museum.py           Room 2 environment (museum heist)
    racing.py           Room 3 environment (dungeon escape)
    football.py         Room 4 continuous environment
    obstacles.py        Room 5 continuous environment (moving traffic + sensors)
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

**Idea.** A robber breaks into a museum to steal the **diamond** from the vault
and escape. The museum has gallery rooms connected by corridors, security
**cameras**, **laser traps**, **patrol guards**, and **slippery marble floors**.
The model is **unknown** — SARSA learns from experience.

The default layout is a museum floor plan with a vault at the top, exhibition
galleries in the middle, and a heavy camera surveillance zone near the exit.
Different generated museums create different heist challenges.

* **State:** `(cell, has_diamond)`, plus `guard_phase` when patrol guards are
  enabled. **Final state:** reaching the exit *with* the diamond.
* **Actions:** Up / Down / Left / Right.
* **Dynamics:** cameras cost a heavy penalty per step in their zone, traps hurt,
  guards end the heist if they catch you, marble floors deflect movement.
* **Rewards:** `+30` diamond · `+100` escape · `−1` step · `−50` camera zone ·
  `−15` trap · `−50` caught by guard (terminal) · `−5` slip · `−2` wall.

**Best parameters.** `episodes = 1500`, `α = 0.2`, `γ = 0.97`, `ε: 1.0 → 0.02`
(decay 0.997 / episode), `slip = 0.1` → **100 % success**. The greedy policy
finds a clean path through the museum, avoids cameras, and escapes with the
diamond.

---

## Room 3 — Street Race · Q-Learning

**Idea.** A car races through a **street circuit** from start to finish,
navigating oil spills, mud patches, crash barriers, and collecting boosters.
The winding streets create multiple routes with different risk/reward tradeoffs.

This is harder than Room 2: the track has more hazards, boosters add bitmask
states (increasing the state space), and oil dynamics create stochastic risk
(slipping into a crash barrier ends the race).

* **State:** `(cell, booster_mask)` — position plus a bitmask of boosters
  already collected. **Final state:** crossing the finish line.
* **Actions:** Up / Down / Left / Right.
* **Dynamics:** oil spills deflect movement sideways (slip), mud costs penalty,
  crash barriers are terminal, boosters are one-time pickups.
* **Rewards:** `+200` finish · `+15` booster (once) · `−1` step · `−5` mud ·
  `−200` crash (terminal) · `−30` off-track · `−5` wall hit.

**Best parameters.** `episodes = 2000`, `α = 0.2`, `γ = 0.97`, `ε: 1.0 → 0.05`
(decay 0.997), `oil slip = 0.2` → Q-Learning aggressively learns the fastest
racing line, collecting boosters on the way.

**Bonus — SARSA comparison.** Tick *"Also train SARSA"* to see the on-policy
contrast. Overlaid reward/steps curves and a **🏁 race** of both greedy
policies on the same track.

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

## Room 5 *(optional)* — Cross the Road · DQN + sensors

**Idea.** A chicken starts on the left sidewalk and must reach the far edge of a
10×10 m road. Cars stream vertically through alternating lanes, wrap in from
off-map, and keep moving while the chicken chooses up / down / left / right /
wait. A **new traffic pattern every episode** means the agent cannot memorise one
layout — it must learn a **reactive** crossing policy from local **sensors**.

* **State:** `(x, y, vₓ, v_y)` (continuous, same physics style as Room 4, 5 move
  actions).
* **Observation:** own dynamics + direction to the far sidewalk + the nearest cars
  inside **`sensor_range` metres**, each as relative position, vertical velocity,
  and closeness. Empty slots read as "clear".
  `obs_dim = 6 + 4 × (cars sensed)`.
* **Rewards:** `+250` crossing the road · `−140` being hit by traffic · `−40` out
  of bounds · a small near-traffic penalty · `−0.08` per step, plus rightward
  progress shaping. **Final state:** reaching the right edge.

**Best parameters.** `episodes ≈ 800`, hidden `(128, 128)`, `lr = 1e-3`, `γ = 0.99`,
`sensor_range = 3.5 m`, `14 cars`, frame-skip 10. **Shrink the sensor range** to
make the chicken short-sighted; **add cars** or increase traffic speed to make the
road more crowded. After training, hit **"Generate & test"** to spawn unseen
traffic and replay the learned policy in it.

**What it tests.** Generalisation via local observation + function approximation —
the agent solves traffic patterns it has never seen.

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
