# 🕹️ RL Escape Room

**Five themed escape rooms · five reinforcement-learning algorithms · one interactive game website.**

**🌐 Live site:** **[rl-escape-room.onrender.com](https://rl-escape-room.onrender.com)** — the app
running online (free Render instance: the first load after it has been idle can take up to a
minute to wake up, and every wake starts with fresh, untrained rooms — training data is
session-only by design).

An agent is locked inside five rooms — a Pacman maze, a museum heist, a street race, a football
final and a deadly road crossing. Each room is a different RL problem solved by a different
algorithm, and the whole project runs as a real web app: a **Python (FastAPI) backend** that
trains the agents and records everything, and a **React + Canvas frontend** that renders each
room as an animated game, streams live training charts, and replays recorded episodes frame by
frame.

| Room | Theme | Algorithm | Environment |
|------|-------|-----------|-------------|
| 01 | 🟡 Pacman Maze | Dynamic Programming (Value / Policy Iteration) | 10×10 grid, known model |
| 02 | 💎 Museum Heist | SARSA (on-policy TD) | 10×10 grid, unknown model |
| 03 | 🏎️ Street Racing | Q-Learning (off-policy TD) | 10×10 grid, unknown model |
| 04 | ⚽ Football Striker | DQN (function approximation) | 10×10 m continuous pitch |
| 05 | 🐔 Cross the Road | DQN + local sensors | 10×10 m continuous road, moving traffic |

---

## Quick start

**Prerequisites (install once):** [Python 3.10+](https://python.org) (64-bit, tick *"Add
python.exe to PATH"*) and [Node.js 18+](https://nodejs.org). Everything else is automatic.

### ▶ One click (Windows)

Double-click **`start.bat`**. On the first run it sets up everything by itself — creates the
Python venv, installs the Python packages (PyTorch is big, give it a few minutes) and the
frontend packages. It then launches in **developer mode** (backend + a hot-reload frontend on
**http://localhost:5173**) so edits show up live.

For the clean single-server production build instead, run **`start.bat prod`** — it builds the
site and serves everything from **http://localhost:8000**. (`start.bat dev` = the default;
`start.bat build` forces a fresh production build.)

On macOS / Linux: `./start.sh` (dev) and `./start.sh prod` do the same.

### ▶ Or copy-paste (Windows PowerShell, from the project folder)

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
cd frontend; npm install; npm run build; cd ..
.\.venv\Scripts\python -m backend.api.main
```

Then open **http://localhost:8000**, pick a room, press **START TRAINING**.

### ▶ Developer mode (hot reload while editing frontend code)

`start.bat dev` — opens the backend (port 8000) and the Vite dev server
(**http://localhost:5173**, proxies `/api` to the backend) in two windows. Manually that is
`python -m backend.api.main` in one terminal and `cd frontend; npm run dev` in another.
After editing frontend code, `start.bat build` rebuilds the bundle served on port 8000.

> **If something goes wrong**
> * **404 / "not built" message on http://localhost:8000** — the frontend was never built;
>   run `start.bat build` (or `cd frontend; npm run build`) and restart the server.
> * **`[winerror 10048] only one usage of each socket address`** — another copy of the server
>   already holds port 8000; close it (find it with `netstat -ano | findstr :8000`).
> * **`No module named fastapi` / `torch`** — you're using the wrong Python; always run
>   through `start.bat` or `.\.venv\Scripts\python`, not the system `python`.
> * **"running scripts is disabled on this system"** — call `npm.cmd` instead of `npm`, or
>   run once: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.
> * PowerShell 5 chains commands with `;` (there is no `&&`), and
>   `python -m backend.api.main` must run from the **project root**, not from `frontend/`.

**Session-only training data:** artefacts under `results/` live only while the server runs —
they are deleted when the server shuts down (and any crash leftovers are cleared on the next
start), so every launch begins with five untrained rooms and the repo never accumulates
training output (`results/` is gitignored too).

---

## Architecture

```
rl/                  ← the RL core (single source of truth, unchanged)
  envs/              pacman.py · museum.py · racing.py · football.py · obstacles.py
  algos/             dp.py · sarsa.py · qlearning.py · td_core.py · dqn.py

backend/             ← web layer wrapped AROUND rl/ (no algorithm logic here)
  envs/, algorithms/ thin re-exports of rl/ under the backend layout
  training/
    train.py         room registry (hyperparameter schemas) + per-room trainers
    evaluate.py      greedy evaluation from saved models + cross-room summary
    replay_recorder.py  RecordingEnv wrapper + JSON replay files
    frames.py        per-room replay-frame builders + layout snapshots
  api/               FastAPI: rooms, training, metrics, replay, config routes
  utils/             paths · JSON serialization · metric series building

frontend/            ← React 18 + Vite + Recharts + HTML5 Canvas
  src/components/    RoomCard · TrainingControls · MetricsDashboard · GameCanvas
                     EpisodeReplay · PolicyView · AlgorithmPanel · ResultsComparison
  src/render/        canvas game renderers (one visual identity per room)
  src/rooms/         per-room briefing views
  src/pages/         Home · Rooms · Room · Algorithms · Results

results/             ← everything training produces, as plain JSON (session-scoped:
  metrics/room{N}.json      wiped on shutdown)  per-episode series + summary
  replays/room{N}/*.json    frame-by-frame episode replays + index.json
  models/                   pickled Q-tables / DP policies · DQN checkpoints (.pt)
  policies/room{N}.json     value heatmap + greedy-arrow export (rooms 1–3)
  configs/room{N}.json      saved hyperparameter overrides (the only part that persists)
```

**Key design point:** the algorithms in `rl/algos` were **not modified**. The backend records
replays by wrapping each environment in a `RecordingEnv` (same `reset()/step()` interface),
and builds metrics from the `TrainResult` object the algorithms already produce. Training runs
in a background thread; the frontend polls `/api/train/{id}/status` for live charts.

### How the frontend and backend communicate

```
GET  /api/rooms                room cards: status, best reward, success rate
GET  /api/rooms/{id}           detail: hyperparameter schema, layout, action names
POST /api/train/{id}           start background training  {params: {...}}
GET  /api/train/{id}/status    live progress + downsampled metric series (polled)
POST /api/train/{id}/stop      request stop
POST /api/evaluate/{id}        run + record 10 greedy episodes from the saved model
GET  /api/metrics/{id}         full saved metric series + summary
GET  /api/policy/{id}          value heatmap / policy arrows (rooms 1–3)
GET  /api/replay/{id}          replay catalogue
GET  /api/replay/{id}/{ep}     one replay: {meta: {..., layout}, frames: [...]}
GET/POST /api/config/{id}      persist hyperparameter overrides
GET  /api/results/summary      cross-room comparison
```

### Replay JSON format

Each replay stores its own layout snapshot (so it renders correctly even after settings
change) plus compact frames:

```jsonc
{
  "meta": { "kind": "eval", "label": "Greedy run 3", "reward": 102.0,
            "steps": 38, "success": true, "layout": { "type": "pacman", ... } },
  "frames": [
    { "t": 12, "a": 3, "r": 9.0, "cum": -3.0, "done": false, "ev": ["coin"],
      "p": [4, 7], "coins": [[2, 2]], "guard": [8, 8], "open": false }
  ]
}
```

Room-specific frame fields: Pacman `p/coins/guard/open` · Museum `p/d/guards/alarm` ·
Racing `p/ncp/open` (+ `rv` rival car in race evals) · Football `p/defs/keeper/shoot` (+ `fl/ball/z` flight frames so kicks
animate) · Cross the Road `p/cars` (car colors/sizes live in the layout).
During training only **milestone episodes** are recorded (1, 2, spread, last) plus every
greedy evaluation episode — enough to *see* learning without gigabytes of JSON.

---

## The rooms

### Room 1 — Pacman Maze · Dynamic Programming
* **State** `(cell, coin-bitmask [, guard position/phase])` — the same cell needs different
  actions depending on which coins remain, so the bitmask is part of the state.
* **Actions** Up / Down / Left / Right (4).
* **Rewards** step −1 · coin +10 · exit +100 · locked-door bump −10 · slip −5 · wall −5 · guard −50.
* **Terminal** standing on the door with all coins collected (or caught by the guard).
* **Dynamics** ice tiles deflect the move sideways with probability `slip_prob` — the stochastic
  transitions are part of the *known* model handed to Value/Policy Iteration.
* **The maze** there are **two independent routes** to the exit and the door has **two open
  approaches**, so a guard can no longer trap the agent by camping the single doorway. The guard
  starts in the **centre** and, in **chase** mode (the default), hunts the agent from the very
  first move — its cell joins the DP state. **Patrol** mode walks a fixed loop with a far smaller
  state space, and **guard speed** is adjustable (a faster chaser costs no extra states).
* **Difficulty** on a generated map you control the **coin** and **ice** counts; each coin
  doubles the DP state space, so with a chasing guard keep coins modest (patrol mode is cheap).
* **What to look at** the Bellman residual Δ dropping (log chart), and the Policy tab: the value
  landscape flips completely between "coins left" and "all collected".

### Room 2 — Museum Heist · SARSA
* **State** `(cell, has_diamond, guard patrol phase)` while the museum is quiet; once the
  **alarm** is raised it becomes `(cell, has_diamond, alarmed, guard positions)` — chasing
  guards depend on history, so their cells must enter the state to stay Markov.
* **Actions** 4-connected moves.
* **Rewards** step −1 · diamond +30 · escape +100 · camera −50 **+ permanent manhunt** (one
  sighting and every guard abandons its patrol to chase you for the rest of the episode) ·
  trap −15 · caught −50 (terminal) · slip −5.
* **Why SARSA** on-policy learning prices the exploration risk into Q, so the thief keeps a
  safety margin around cameras — visible in the replays. Generated layouts are verified to
  contain a camera/trap/patrol-free stealth route.

### Room 3 — Grand Prix · Q-Learning **vs a SARSA rival**
* **The circuit** a proper F1-style ribbon on a **10×10 grid** — a grass infield ringed by the
  track, red/white kerbs, a wall of TecPro crash barriers, a **chicane** on the back straight,
  gravel run-off and **iridescent oil slicks** on the outer-loop straights (slippery — a slip
  slides the car sideways and costs time, but only the barriers crash). Your Q-Learning car
  races a SARSA rival trained on the identical circuit with
  identical hyperparameters. The lap has **checkpoint gates** (cross 1, then 2, then the finish
  opens) and every gate exists twice — once on each route — so both are full laps.
* **The two lines** it's *cliff walking* staged as a Grand Prix: the **main straight** (9 steps)
  runs S→F right along the barrier wall — safe when driven greedily, fatal to exploratory
  wobbles — while the safe **outer loop** detours the long way round through the gravel (≈25
  steps), nowhere near anything terminal.
* **State** `(cell, next-checkpoint)` per car — each tracks its own lap progress.
* **Rewards** step −1 · checkpoint +40 · finish +200 · crash −200 (terminal) · gravel −5 ·
  oil-slip −5 · locked-finish bump −10 · off-track penalties. First car home wins the race.
* **Why it works** Q-Learning's off-policy max-target ignores its own exploration accidents and
  learns the barrier-hugging main straight; SARSA's on-policy target prices every ε-wobble into
  the barriers and settles on the safe outer loop. Same table, same settings — the race verdict
  *is* the off-policy/on-policy distinction (minimum ε is kept at 0.15 so it survives). The SARSA
  rival's seed is chosen so it greedily completes the *safe* route, guaranteeing the contrast.
* **Watch** the reward chart overlays both learning curves; eval replays show both cars racing
  side by side on their two routes with a 🏁 RACES WON score, and the loser finishes on camera.

### Room 4 — Football Striker · DQN
A continuous 10×10 m pitch (`dt = 0.02 s`) with two modes:

* **Match** — dribble past chasing defenders into the shooting zone, then shoot past a patrolling
  keeper. **State** continuous `x, y, Vx, Vy`, keeper position + patrol direction, per-defender
  relative vectors (10 + 3·defenders). **Actions** 15: 8 move directions + stay + 6 kicks
  (soft/hard × straight/curve-L/curve-R). **Rewards** goal +300 · save −10 · miss −30 · tackled
  −50 · out-of-bounds −30 · progress shaping · step −0.3 · shot-clock timeout −25.
* **Free kick** — a **single kick** at goal, taken from a **truly random position every episode**
  (any distance × any angle). The defensive **wall grows with proximity to goal** (2–5 players)
  and is set **diagonally** on the ball→goal line — so a side free kick gets a wall angled between
  the ball and the net, just like the real thing. The ball has 3D physics (rises/falls in z);
  the striker must clear or curl around the wall *and* beat the keeper in one shot. **State**
  keeper position + direction, kick spot, and up to 5 (padded) wall-player vectors — a fixed
  length even though the wall size changes. **Actions** 18 one-shot kicks (3 aims × 2 powers × 3
  curves). **Rewards** goal +300 · save/blocked/miss penalties.
* **Why DQN** no table can cover a continuous state; an MLP (128×128) approximates Q(s,a) with
  experience replay + a target network.

### Room 5 — Cross the Road · DQN + sensors
* **State** `x, y, Vx, Vy`, direction to goal, and **6 sensor slots** × (relative position,
  velocity, closeness) for the nearest cars in range — the agent never sees the whole map.
* **Actions** 9: Up / Down / Left / Right / Stay **+ the 4 diagonals** (unit-normalised so a
  diagonal dash is not faster than a straight one).
* **Rewards** crossed +250 · collision −140 (terminal) · off-road −40 (terminal) · near-miss
  penalty · progress shaping.
* **Generalisation** obstacle count and positions are dynamic and the traffic can be
  re-randomised every episode, so the policy must react to sensors instead of memorising a
  pattern — you can build a fresh random road after training and test the learned policy on it.

---

## Hyperparameters (Training Dashboard)

### Optimal parameters (the shipped defaults)

Each room's default hyperparameters are the ones that solve it well — found by sweeping and
measuring greedy success over hundreds of episodes. Load a room, press **START TRAINING**, and
these are what run:

| Room · Algorithm | Optimal defaults | Result (greedy) |
|---|---|---|
| **1 Pacman · Value Iteration** | γ 0.95 · θ 1e-4 · ≤250 sweeps · slip 0.2 | **optimal policy** (DP is exact) |
| **2 Museum · SARSA** | α 0.1 · γ 0.95 · ε 1.0→0.05 (×0.995/ep) · 2000 ep · slip 0.1 | **~96%** escape |
| **3 Racing · Q-Learning** | α 0.2 · γ 0.95 · ε 1.0→**0.15** (kept high on purpose) · 1200 ep · slip 0.2 | **beats SARSA 100%**, 9-step lap |
| **4 Football — match · DQN** | lr 5e-4 · γ 0.98 · expl 0.65 · batch 128 · 900 ep · 3 defenders | **~63%** goals |
| **4 Football — free kick · DQN** | same DQN · `kick_spot=random` · dynamic wall (2–5) | **~67%** single-kick |
| **5 Cross the Road · DQN** | lr 1e-3 · γ 0.99 · expl 0.5 · batch 64 · 900 ep · 14 cars | **~67%** crossing |

Notes: Room 3 keeps a high minimum ε (0.15) so the on-policy/off-policy split survives; the
DQN rooms use a 128×128 MLP with a 50 000-step replay buffer and a target net refreshed every
500 steps. The free kick trains on 8× as many (one-kick) episodes as the shared `episodes`
control implies, since each episode is a single step.

### All the controls

Every control in the UI comes from the backend schema in `backend/training/train.py`:

* **DP (Room 1):** method (value/policy iteration), γ, convergence threshold θ, max iterations,
  evaluation episodes, slip probability, guard on/off + **behaviour (chase / patrol) + speed**,
  and — for generated maps — **coin** and **ice** counts.
* **SARSA / Q-Learning (Rooms 2–3):** α, γ, ε start / min / decay, episodes, max steps, slip
  probability (+ alarm toggle in Room 2), and generated-map difficulty counts — **cameras,
  laser traps, patrol guards, marble tiles** (Room 2); **oil slicks, gravel, checkpoints** (Room 3).
* **Map studio (Rooms 1–3):** each grid room has a live map preview and a **🎲 New random
  layout** button — adjust the difficulty counts (marked ◇), watch the generated map update in
  the preview, then train on it. The generators keep every map solvable (Room 1 guarantees two
  routes to the exit; Room 2 verifies a stealth route; Room 3 keeps the express/safe contrast).
* **DQN (Rooms 4–5):** learning rate, γ, ε start / min, exploration fraction, batch size,
  replay buffer size, target-network update frequency, episodes, max steps
  (+ mode / free-kick spot / defenders / keeper speed in Room 4;
  cars / speeds / sensor range / randomise in Room 5).

Press **SAVE CONFIG** to persist a setup as the room default (`results/configs/`).

## Reading the graphs

* **Reward per episode + moving average** — the main learning curve; the average should climb.
* **Steps per episode** — falling steps = shorter, more direct solutions (grid rooms).
* **Success / failure rate (rolling 50)** — fraction of recent episodes that reached the goal.
* **ε over time** — the exploration schedule; learning usually accelerates as ε decays.
* **Mean |TD error|** (rooms 2–3) — shrinking TD error = the value estimates are converging.
* **DQN loss** (rooms 4–5) — should stay bounded; explosions mean lr / target-update trouble.
* **Bellman residual Δ** (room 1, log scale) — DP's convergence certificate.
* **Hazard events** — cameras / catches / traps / crashes per episode, falling as the agent learns.
* Rewards are **not comparable across rooms** (different scales) — compare success rates.

## Replays

Every training run records a spread of **milestone episodes** (first, evenly spread, last) so you
can watch the behaviour evolve stage by stage. The greedy evaluation then runs many episodes for
a stable success rate (up to 500 for the one-kick free kick) but keeps ~15 of them as replays.
The Replay tab plays them with **play / pause / reset /
step ±1 / 0.25×–8× speed / scrubbing**, an interpolated animated canvas, and a state monitor
showing the current state, action name, step reward, cumulative reward, per-step events and
the terminal outcome. Football kicks are expanded into ball-flight frames; Cross-the-Road
replays draw the live sensor ring.

## Training each room

From the UI: room page → TRAIN → START TRAINING. From the CLI:

```bash
python -m backend.training.train --room 3 --set episodes=1500 alpha=0.15
```
