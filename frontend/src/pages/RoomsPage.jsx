import React, { useEffect, useState } from 'react';
import { api } from '../api.js';
import RoomSelector from '../components/RoomSelector.jsx';

export default function RoomsPage() {
  const [rooms, setRooms] = useState(null);
  const [error, setError] = useState('');

  const load = () => api.rooms()
    .then(({ rooms: r }) => setRooms(r))
    .catch((e) => setError(String(e.message || e)));

  useEffect(() => {
    load();
    const t = setInterval(load, 5000); // keep training badges fresh
    return () => clearInterval(t);
  }, []);

  return (
    <main className="page">
      <h1 className="display" style={{ fontSize: 28, marginBottom: 6 }}>
        <span className="grad-text">SELECT A ROOM</span>
      </h1>
      <p className="subtle" style={{ marginBottom: 26, maxWidth: 720 }}>
        Each room is a self-contained RL problem with its own environment, reward design
        and algorithm. Train it, inspect the learning curves, then replay what the agent
        actually learned.
      </p>
      {error && <p style={{ color: 'var(--red)' }}>⚠ {error} — is the backend running on port 8000?</p>}
      {!rooms && !error && <div className="loader" />}
      {rooms && <RoomSelector rooms={rooms} />}
    </main>
  );
}
