import React, { useEffect, useState } from 'react';
import { api } from '../api.js';

const ARROWS = ['↑', '↓', '←', '→'];

function valueColor(v, vmin, vmax) {
  if (v === null || v === undefined) return 'transparent';
  const t = vmax > vmin ? (v - vmin) / (vmax - vmin) : 0.5;
  // cold violet → hot cyan/green
  const hue = 265 - t * 115;
  return `hsla(${hue}, 75%, ${22 + t * 26}%, 0.92)`;
}

/**
 * Value-function heatmap + greedy policy arrows for the tabular rooms.
 * Toggle between the room's two flag settings (e.g. before / after the
 * diamond is stolen).
 */
export default function PolicyView({ room }) {
  const [data, setData] = useState(null);
  const [flag, setFlag] = useState(0);
  const [showValues, setShowValues] = useState(true);

  useEffect(() => {
    api.policy(room.id).then((d) => setData(d)).catch(() => setData(null));
  }, [room.id]);

  if (!data) return <p className="subtle mono">loading policy…</p>;
  if (!data.available) {
    return (
      <div className="panel" style={{ textAlign: 'center', padding: 44 }}>
        <span className="subtle">
          {room.id <= 3
            ? 'No saved policy yet — train this room first.'
            : 'Policy grids only exist for the tabular rooms (1–3); this room uses a neural network over a continuous state.'}
        </span>
      </div>
    );
  }

  const pol = data.policy;
  const fl = pol.flags[Math.min(flag, pol.flags.length - 1)];
  const walls = new Set((pol.walls || []).map(([r, c]) => `${r},${c}`));
  const flat = fl.values.flat().filter((v) => v !== null);
  const vmin = Math.min(...flat);
  const vmax = Math.max(...flat);

  return (
    <div className="stack">
      <div className="row-gap">
        {pol.flags.map((f, i) => (
          <button key={f.label} className={`btn btn-sm ${i === flag ? 'btn-primary' : ''}`}
            onClick={() => setFlag(i)}>
            {f.label}
          </button>
        ))}
        <span className="spacer" />
        <label className="checkbox-row">
          <input type="checkbox" checked={showValues}
            onChange={(e) => setShowValues(e.target.checked)} />
          show V(s)
        </label>
      </div>
      <div className="grid-2">
        <div className="panel accent">
          <h3 className="panel-title">GREEDY POLICY + VALUE HEATMAP</h3>
          <div className="policy-grid"
            style={{ gridTemplateColumns: `repeat(${pol.size}, 1fr)` }}>
            {fl.values.map((row, r) => row.map((v, c) => {
              const isWall = walls.has(`${r},${c}`);
              const a = fl.actions[r][c];
              return (
                <div key={`${r}-${c}`} className="policy-cell"
                  title={v !== null ? `(${r},${c}) V=${v}` : `(${r},${c})`}
                  style={{
                    background: isWall ? 'var(--policy-wall)' : valueColor(v, vmin, vmax),
                    border: isWall ? '1px solid var(--line-bright)' : '1px solid transparent',
                  }}>
                  {!isWall && a !== null && ARROWS[a]}
                  {!isWall && showValues && v !== null && (
                    <span className="val">{Math.round(v)}</span>
                  )}
                </div>
              );
            }))}
          </div>
        </div>
        <div className="panel">
          <h3 className="panel-title">HOW TO READ THIS</h3>
          <p className="subtle">
            Every cell shows the <b style={{ color: 'var(--text)' }}>greedy action</b> (the
            arrow) and the <b style={{ color: 'var(--text)' }}>state value</b> — the expected
            discounted return when acting optimally from that cell. Warm bright cells are
            close to the goal; cold violet cells are far away or dangerous.
          </p>
          <p className="subtle">
            The two buttons switch the room flag: the optimal route changes completely once
            the objective (coins / diamond / boosters) has been collected — the same maze,
            but a different value landscape.
          </p>
          <div className="row-gap" style={{ marginTop: 10 }}>
            <span className="chip cyan">V(s) max {Math.round(vmax)}</span>
            <span className="chip violet">V(s) min {Math.round(vmin)}</span>
          </div>
        </div>
      </div>
    </div>
  );
}
