import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { api } from '../api.js';
import RoomSelector from '../components/RoomSelector.jsx';

const STEPS = [
  ['01', 'Pick a room', 'Five themed rooms, five algorithms — from a fully-known maze solved by Dynamic Programming to sensor-driven traffic solved by DQN.'],
  ['02', 'Tune & train', 'Set the hyperparameters (α, γ, ε-decay, buffers…) and watch the reward curves, success rate and exploration decay live while the agent learns.'],
  ['03', 'Watch the replay', 'Every training milestone and every greedy evaluation is recorded frame by frame — play, scrub, and step through exactly what the agent did.'],
  ['04', 'Compare results', 'Value heatmaps, policy arrows, loss curves and a cross-room comparison connect the behaviour you see to the math underneath.'],
];

export default function Home() {
  const [rooms, setRooms] = useState([]);
  const [error, setError] = useState('');

  useEffect(() => {
    api.rooms().then(({ rooms: r }) => setRooms(r))
      .catch((e) => setError(String(e.message || e)));
  }, []);

  const trained = rooms.filter((r) => r.trained).length;
  const episodes = rooms.reduce((s, r) => s + (r.episodes || 0), 0);

  return (
    <main className="page">
      <section className="hero">
        <div className="tagline">REINFORCEMENT LEARNING · FINAL PROJECT</div>
        <h1><span className="grad-text">RL ESCAPE ROOM</span></h1>
        <p className="lead">
          An agent locked inside five themed rooms — a Pacman maze, a museum heist,
          a street race, a football final and a deadly road crossing. Each room is a
          different reinforcement-learning problem solved by a different algorithm:
          <b> Dynamic Programming, SARSA, Q-Learning</b> and <b>Deep Q-Networks</b>.
        </p>
        <div className="hero-actions">
          <Link to="/rooms" className="btn btn-primary">▸ ENTER THE ROOMS</Link>
          <Link to="/algorithms" className="btn">THE ALGORITHMS</Link>
          <Link to="/results" className="btn">RESULTS</Link>
        </div>
      </section>

      {error && (
        <div className="panel" style={{ borderColor: 'var(--red)', marginBottom: 20 }}>
          <span style={{ color: 'var(--red)' }}>
            ⚠ Backend unreachable ({error}) — start it with{' '}
            <span className="mono">python -m backend.api.main</span>
          </span>
        </div>
      )}

      <div className="stat-strip">
        <div className="stat-tile"><div className="big">5</div><div className="lbl">Escape rooms</div></div>
        <div className="stat-tile"><div className="big">4</div><div className="lbl">RL algorithms</div></div>
        <div className="stat-tile"><div className="big">{trained}/5</div><div className="lbl">Rooms trained</div></div>
        <div className="stat-tile"><div className="big">{episodes.toLocaleString()}</div><div className="lbl">Episodes logged</div></div>
      </div>

      {rooms.length > 0 && (
        <section style={{ marginBottom: 40 }}>
          <h1 className="display" style={{ fontSize: 20, margin: '0 0 16px' }}>
            <span className="grad-text">▮ THE FIVE ROOMS</span>
          </h1>
          <RoomSelector rooms={rooms} />
        </section>
      )}

      <section>
        <h1 className="display" style={{ fontSize: 20, margin: '0 0 16px' }}>
          <span className="grad-text">▮ HOW IT WORKS</span>
        </h1>
        <div className="step-grid stagger">
          {STEPS.map(([num, title, desc]) => (
            <div className="panel step-card" key={num}>
              <div className="step-num">{num}</div>
              <h4>{title}</h4>
              <p className="subtle" style={{ margin: 0 }}>{desc}</p>
            </div>
          ))}
        </div>
      </section>
    </main>
  );
}
