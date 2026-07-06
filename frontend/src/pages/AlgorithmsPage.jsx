import React, { useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import AlgorithmPanel from '../components/AlgorithmPanel.jsx';

const ALGOS = [
  {
    id: 'dp', roomId: 1, emoji: '🧠', name: 'Dynamic Programming',
    family: 'Model-based · exact',
    intro: 'When the environment model — every transition probability and reward — is fully '
      + 'known, no exploration is needed at all. Value Iteration sweeps the whole state space, '
      + 'repeatedly applying the Bellman optimality backup until values stop changing; Policy '
      + 'Iteration alternates full policy evaluation with greedy improvement. Both converge to '
      + 'the provably optimal policy. In Room 1 the maze, the coins, the ice-slip probabilities '
      + 'and even the guard\'s patrol are all part of the known model.',
    update: 'V(s) ← max_a Σ_s\' P(s\'|s,a) · [ r + γ·V(s\') ]',
    properties: [
      'Requires the full transition model P(s\'|s,a) — no sampling, no episodes.',
      'Value iteration: one Bellman-optimality backup per state per sweep.',
      'Policy iteration: evaluate π completely, then improve greedily; fewer, heavier iterations.',
      'Convergence measured by the Bellman residual Δ (see the log-scale chart after training).',
      'The coin bitmask multiplies the state space: 10×10 cells × 2ⁿ coin sets (× guard states).',
    ],
    hyper: [
      ['γ (discount)', 'How much future reward matters. Higher γ values the +100 exit more against per-step costs.'],
      ['θ (threshold)', 'Stop sweeping when the largest value change falls below θ — precision vs compute.'],
      ['method', 'value_iteration or policy_iteration — same optimum, different convergence path.'],
    ],
  },
  {
    id: 'sarsa', roomId: 2, emoji: '🕶️', name: 'SARSA',
    family: 'Model-free · on-policy TD',
    intro: 'SARSA learns from raw experience — State, Action, Reward, next State, next Action. '
      + 'It is on-policy: the TD target uses the action the ε-greedy policy actually takes next, '
      + 'so exploration noise is baked into the learned values. Near cliffs (cameras, guards) '
      + 'that makes SARSA cautious — it learns the value of the exploring policy it truly runs, '
      + 'which is exactly what you want in a museum full of alarms.',
    update: 'Q(s,a) ← Q(s,a) + α·[ r + γ·Q(s\',a\') − Q(s,a) ]',
    properties: [
      'On-policy: a\' is sampled from the same ε-greedy policy — no max operator.',
      'Learns "safe" values: the cost of possibly slipping into a camera zone is priced in.',
      'ε decays per episode: explore early, exploit late.',
      'The TD error |δ| shrinking over episodes is the clearest sign of convergence.',
      'Tabular: one Q value per (cell, diamond, guard-phase, alarm) state and action.',
    ],
    hyper: [
      ['α (learning rate)', 'How strongly each TD error updates Q. Too high oscillates, too low crawls.'],
      ['γ (discount)', 'Future vs immediate reward — the +100 escape must outweigh many −1 steps.'],
      ['ε start / min / decay', 'The exploration schedule. Decay too fast → the thief never finds the vault route.'],
      ['episodes', 'More episodes = more state-action visits = smoother Q estimates.'],
    ],
  },
  {
    id: 'qlearning', roomId: 3, emoji: '🏁', name: 'Q-Learning',
    family: 'Model-free · off-policy TD',
    intro: 'Q-Learning\'s TD target uses max_a Q(s\', a) — the best next action, regardless of what '
      + 'the exploring behaviour policy actually does. That single max makes it off-policy: it '
      + 'learns the optimal greedy racing line even while the car is still exploring randomly. '
      + 'The flip side: it prices risk optimistically, hugging oil slicks and barriers tighter '
      + 'than SARSA would — perfect for a street race.',
    update: 'Q(s,a) ← Q(s,a) + α·[ r + γ·max_a\' Q(s\',a\') − Q(s,a) ]',
    properties: [
      'Off-policy: learns about the greedy policy while following an ε-greedy one.',
      'Converges to Q* under standard conditions, independent of the exploration policy.',
      'Compared to SARSA it takes the risky-but-optimal line — watch it shave past oil slicks.',
      'The booster bitmask in the state lets the same cell have different optimal actions.',
      'Crash counts per episode falling while rewards rise = learning the risk boundary.',
    ],
    hyper: [
      ['α (learning rate)', 'TD-error step size, as in SARSA.'],
      ['γ (discount)', 'High γ needed: the +200 finish is many steps away from the start line.'],
      ['ε schedule', 'Exploration must survive long enough to discover the booster detours.'],
      ['min boosters', 'Environment knob: how many pads unlock the finish — reshapes the whole route.'],
    ],
  },
  {
    id: 'dqn', roomId: 4, emoji: '🧬', name: 'Deep Q-Network (DQN)',
    family: 'Function approximation · neural TD',
    intro: 'Rooms 4 and 5 have continuous states — real-valued positions, velocities, sensor '
      + 'readings. No table can enumerate them, so a small neural network approximates Q(s, a) '
      + 'for all actions at once. Two classic tricks stabilise the training: an experience replay '
      + 'buffer breaks the correlation between consecutive samples, and a frozen target network '
      + 'keeps the regression target from chasing its own tail.',
    update: 'L(θ) = E[( r + γ·max_a\' Q(s\',a\'; θ⁻) − Q(s,a; θ) )²]',
    properties: [
      'MLP 128×128 maps the observation vector to one Q value per action.',
      'Experience replay: learn from random past transitions, not just the latest one.',
      'Target network θ⁻ updated every N steps — the "chasing a moving target" fix.',
      'ε-greedy exploration with a linear decay over a fraction of training.',
      'The loss curve is diagnostic: explosions usually mean the lr or target-update is off.',
    ],
    hyper: [
      ['learning rate', 'Adam step size for the network — the most sensitive knob in the room.'],
      ['batch size / buffer size', 'How many and how diverse the replayed transitions are.'],
      ['target update', 'Steps between target-network syncs. Small = unstable, huge = slow.'],
      ['exploration fraction', 'Portion of training over which ε anneals to its minimum.'],
      ['γ (discount)', '0.99 — continuous rooms need long horizons for shaped progress rewards.'],
    ],
  },
];

export default function AlgorithmsPage() {
  const { hash } = useLocation();

  useEffect(() => {
    if (hash) {
      const el = document.getElementById(hash.slice(1));
      if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  }, [hash]);

  return (
    <main className="page">
      <h1 className="display" style={{ fontSize: 28, marginBottom: 6 }}>
        <span className="grad-text">THE ALGORITHMS</span>
      </h1>
      <p className="subtle" style={{ marginBottom: 26, maxWidth: 780 }}>
        Four families of reinforcement learning, ordered exactly like the rooms: from
        planning with a perfect model, through tabular temporal-difference learning,
        to neural function approximation. Every card links to the room where you can
        watch the algorithm at work.
      </p>
      <div className="stack">
        {ALGOS.map((a) => <AlgorithmPanel key={a.id} algo={a} />)}
      </div>
    </main>
  );
}
