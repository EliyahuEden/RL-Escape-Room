import React from 'react';
import { Link } from 'react-router-dom';

/** One explanation card on the Algorithms page. */
export default function AlgorithmPanel({ algo }) {
  return (
    <div className="panel algo-card" id={algo.id} data-room={algo.roomId}>
      <div className="algo-head">
        <span className="algo-badge">{algo.emoji}</span>
        <h2>{algo.name}</h2>
        <span className="chip accent">{algo.family}</span>
        <span className="spacer" />
        <Link className="btn btn-sm" to={`/room/${algo.roomId}`}>
          USED IN ROOM {String(algo.roomId).padStart(2, '0')} →
        </Link>
      </div>
      <p className="subtle" style={{ maxWidth: 900 }}>{algo.intro}</p>
      <div className="grid-2" style={{ marginTop: 12 }}>
        <div>
          <p className="mono-label" style={{ marginBottom: 8 }}>Update rule</p>
          <div className="code-block">{algo.update}</div>
          <p className="mono-label" style={{ margin: '16px 0 8px' }}>Key properties</p>
          <ul className="subtle" style={{ margin: 0, paddingLeft: 18, lineHeight: 1.8 }}>
            {algo.properties.map((p) => <li key={p}>{p}</li>)}
          </ul>
        </div>
        <div>
          <p className="mono-label" style={{ marginBottom: 8 }}>Hyperparameters that matter</p>
          <div className="spec-table">
            {algo.hyper.map(([k, v]) => (
              <div className="spec-row" key={k}>
                <span className="k">{k}</span>
                <span className="v subtle">{v}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
