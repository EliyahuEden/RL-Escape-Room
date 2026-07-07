import React from 'react';
import { Link } from 'react-router-dom';
import {
  Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts';

// theme-aware via CSS variables (resolved live, so the toggle re-skins charts)
const ROOM_COLORS = ['var(--gold)', 'var(--purple)', 'var(--red)', 'var(--green)', 'var(--sky)'];
const AXIS = { stroke: 'var(--chart-axis)', fontSize: 10, tickLine: false };
const GRID = { stroke: 'var(--chart-grid)', strokeDasharray: '3 6' };
const TOOLTIP = {
  contentStyle: {
    background: 'var(--tooltip-bg)', border: '1px solid var(--line-bright)',
    borderRadius: 8, color: 'var(--text)',
    fontFamily: '"JetBrains Mono", monospace', fontSize: 12,
  },
  labelStyle: { color: 'var(--muted)' },
};

function fmt(x, digits = 0) {
  return x === null || x === undefined ? '—' : (+x).toFixed(digits);
}

/** Cross-room comparison: table + bar charts. */
export default function ResultsComparison({ rooms }) {
  const chartRows = rooms.map((r, i) => ({
    name: `R${r.id}`,
    success: r.eval_success_rate !== null && r.eval_success_rate !== undefined
      ? +(r.eval_success_rate * 100).toFixed(0)
      : (r.success_rate !== null && r.success_rate !== undefined
        ? +(r.success_rate * 100).toFixed(0) : 0),
    best: r.best_reward ?? 0,
    color: ROOM_COLORS[i % ROOM_COLORS.length],
  }));

  return (
    <div className="stack">
      <div className="panel">
        <h3 className="panel-title">ROOM COMPARISON</h3>
        <div style={{ overflowX: 'auto' }}>
          <table className="data-table">
            <thead>
              <tr>
                <th>Room</th><th>Algorithm</th><th>Trained</th><th>Episodes</th>
                <th>Best reward</th><th>Avg reward (last 50)</th>
                <th>Train success</th><th>Greedy eval success</th>
                <th>Train time</th><th>Last trained</th>
              </tr>
            </thead>
            <tbody>
              {rooms.map((r, i) => (
                <tr key={r.id}>
                  <td>
                    <Link to={`/room/${r.id}`} style={{ color: ROOM_COLORS[i], textDecoration: 'none', fontWeight: 600 }}>
                      {String(r.id).padStart(2, '0')} · {r.name}
                    </Link>
                  </td>
                  <td>{r.algorithm}</td>
                  <td>{r.trained ? '✔' : '—'}</td>
                  <td>{r.episodes ?? '—'}</td>
                  <td>{fmt(r.best_reward, 1)}</td>
                  <td>{fmt(r.avg_reward_last50, 1)}</td>
                  <td>{r.success_rate != null ? `${Math.round(r.success_rate * 100)}%` : '—'}</td>
                  <td>{r.eval_success_rate != null ? `${Math.round(r.eval_success_rate * 100)}%` : '—'}</td>
                  <td>{r.train_time != null ? `${r.train_time}s` : '—'}</td>
                  <td className="subtle">{r.timestamp ?? '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="grid-2">
        <div className="panel chart-card">
          <p className="chart-title">Greedy success rate (%)</p>
          <ResponsiveContainer width="100%" height={230}>
            <BarChart data={chartRows} margin={{ top: 5, right: 12, bottom: 0, left: -12 }}>
              <CartesianGrid {...GRID} />
              <XAxis dataKey="name" {...AXIS} />
              <YAxis {...AXIS} width={50} domain={[0, 100]} />
              <Tooltip {...TOOLTIP} cursor={{ fill: 'var(--chart-cursor)' }} />
              <Bar dataKey="success" radius={[4, 4, 0, 0]} isAnimationActive={false}>
                {chartRows.map((r) => <Cell key={r.name} fill={r.color} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
        <div className="panel chart-card">
          <p className="chart-title">Best episode reward</p>
          <ResponsiveContainer width="100%" height={230}>
            <BarChart data={chartRows} margin={{ top: 5, right: 12, bottom: 0, left: -12 }}>
              <CartesianGrid {...GRID} />
              <XAxis dataKey="name" {...AXIS} />
              <YAxis {...AXIS} width={50} />
              <Tooltip {...TOOLTIP} cursor={{ fill: 'var(--chart-cursor)' }} />
              <Bar dataKey="best" radius={[4, 4, 0, 0]} isAnimationActive={false}>
                {chartRows.map((r) => <Cell key={r.name} fill={r.color} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
      <p className="subtle">
        Rewards are <b>not comparable across rooms</b> (each room has its own reward
        scale) — compare success rates, and compare rewards only within a room across
        hyperparameter settings.
      </p>
    </div>
  );
}
