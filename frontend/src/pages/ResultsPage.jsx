import React, { useEffect, useState } from 'react';
import { api } from '../api.js';
import ResultsComparison from '../components/ResultsComparison.jsx';

export default function ResultsPage() {
  const [rooms, setRooms] = useState(null);
  const [error, setError] = useState('');

  useEffect(() => {
    api.summary()
      .then(({ rooms: r }) => setRooms(r))
      .catch((e) => setError(String(e.message || e)));
  }, []);

  return (
    <main className="page">
      <h1 className="display" style={{ fontSize: 28, marginBottom: 6 }}>
        <span className="grad-text">RESULTS & COMPARISON</span>
      </h1>
      <p className="subtle" style={{ marginBottom: 26, maxWidth: 720 }}>
        One row per room: training volume, best and recent rewards, and the success rate
        of the final greedy policy over 10 recorded evaluation episodes.
      </p>
      {error && <p style={{ color: 'var(--red)' }}>⚠ {error}</p>}
      {!rooms && !error && <div className="loader" />}
      {rooms && <ResultsComparison rooms={rooms} />}
    </main>
  );
}
