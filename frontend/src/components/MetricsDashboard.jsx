import React from 'react';
import {
  Bar, BarChart, CartesianGrid, Legend, Line, LineChart,
  ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts';

const AXIS = { stroke: '#39456e', fontSize: 10, tickLine: false };
const GRID = { stroke: '#182042', strokeDasharray: '3 6' };
const TOOLTIP = {
  contentStyle: {
    background: '#0c1226', border: '1px solid #2c3768', borderRadius: 8,
    fontFamily: '"JetBrains Mono", monospace', fontSize: 12,
  },
  labelStyle: { color: '#7e88ab' },
};

function toRows(series, xKey, keys) {
  const x = series[xKey] || [];
  return x.map((xv, i) => {
    const row = { x: xv };
    keys.forEach((k) => {
      const arr = series[k];
      if (arr && arr[i] !== undefined && arr[i] !== null) row[k] = +(+arr[i]).toFixed(3);
    });
    return row;
  });
}

function ChartCard({ title, children }) {
  return (
    <div className="panel chart-card">
      <p className="chart-title">{title}</p>
      <ResponsiveContainer width="100%" height={195}>
        {children}
      </ResponsiveContainer>
    </div>
  );
}

function Lines({ rows, defs, logY = false, ...rest }) {
  return (
    <LineChart data={rows} {...rest} margin={{ top: 5, right: 12, bottom: 0, left: -10 }}>
      <CartesianGrid {...GRID} />
      <XAxis dataKey="x" {...AXIS} />
      <YAxis {...AXIS} width={54} scale={logY ? 'log' : 'auto'}
        domain={logY ? ['auto', 'auto'] : undefined} allowDataOverflow={logY} />
      <Tooltip {...TOOLTIP} />
      {defs.length > 1 && <Legend wrapperStyle={{ fontSize: 11 }} />}
      {defs.map(([k, color, width]) => (
        <Line key={k} type="monotone" dataKey={k} stroke={color}
          strokeWidth={width || 1.4} dot={false} isAnimationActive={false} />
      ))}
    </LineChart>
  );
}

/**
 * The chart wall. `series` comes either from the live training status
 * (polled) or from the saved metrics file — same shape either way.
 */
export default function MetricsDashboard({ series, summary, evalSummary }) {
  if (!series || !Object.keys(series).length) {
    return (
      <div className="panel" style={{ textAlign: 'center', padding: 44 }}>
        <span className="subtle blink-slow mono">
          NO TRAINING DATA YET — PRESS START TRAINING
        </span>
      </div>
    );
  }

  const charts = [];

  // DP convergence charts (room 1)
  if (series.dp_delta) {
    const dpRows = toRows(series, 'dp_iteration',
      ['dp_delta', 'dp_start_value', 'dp_policy_changes']);
    charts.push(
      <ChartCard key="dpd" title="Bellman residual Δ per sweep (log scale)">
        <Lines rows={dpRows.filter((r) => r.dp_delta > 0)}
          defs={[['dp_delta', '#ff4d5a', 2]]} logY />
      </ChartCard>,
      <ChartCard key="dpv" title="Value of the start state per sweep">
        <Lines rows={dpRows} defs={[['dp_start_value', '#38bdf8', 2]]} />
      </ChartCard>,
    );
    if (series.dp_policy_changes) {
      charts.push(
        <ChartCard key="dpc" title="Policy changes per iteration">
          <Lines rows={dpRows} defs={[['dp_policy_changes', '#fbbf24', 2]]} />
        </ChartCard>,
      );
    }
  }

  const epRows = toRows(series, 'episode', [
    'reward', 'reward_avg', 'steps', 'epsilon', 'success_rate', 'failure_rate',
    'td_error', 'loss', 'camera_hits', 'caught', 'trap_hits', 'crashes', 'shortcuts',
  ]);

  if (series.reward) {
    charts.push(
      <ChartCard key="rw"
        title={series.dp_delta ? 'Greedy evaluation reward per episode'
          : 'Reward per episode (+ moving average)'}>
        <Lines rows={epRows}
          defs={[['reward', 'rgba(139,92,246,0.4)', 1], ['reward_avg', '#ffd23f', 2.2]]} />
      </ChartCard>,
    );
  }
  if (series.steps) {
    charts.push(
      <ChartCard key="st" title="Steps per episode">
        <Lines rows={epRows} defs={[['steps', '#38bdf8', 1.4]]} />
      </ChartCard>,
    );
  }
  if (series.success_rate) {
    charts.push(
      <ChartCard key="sr" title="Success / failure rate (rolling 50)">
        <Lines rows={epRows}
          defs={[['success_rate', '#34d399', 2], ['failure_rate', 'rgba(255,77,90,0.6)', 1.4]]} />
      </ChartCard>,
    );
  }
  if (series.epsilon) {
    charts.push(
      <ChartCard key="eps" title="Exploration rate ε over time">
        <Lines rows={epRows} defs={[['epsilon', '#a78bfa', 2]]} />
      </ChartCard>,
    );
  }
  if (series.td_error) {
    charts.push(
      <ChartCard key="td" title="Mean |TD error| per episode">
        <Lines rows={epRows} defs={[['td_error', '#f472b6', 1.6]]} />
      </ChartCard>,
    );
  }
  if (series.loss && series.loss.some((v) => v)) {
    charts.push(
      <ChartCard key="loss" title="DQN training loss (per-episode mean)">
        <Lines rows={epRows.filter((r) => r.loss !== undefined)}
          defs={[['loss', '#fb923c', 1.6]]} />
      </ChartCard>,
    );
  }
  const hazardDefs = [
    ['camera_hits', '#ff4d5a', 'Camera detections'],
    ['caught', '#f87171', 'Caught by guards'],
    ['trap_hits', '#fb923c', 'Trap hits'],
    ['crashes', '#ff4d5a', 'Crashes'],
    ['shortcuts', '#38bdf8', 'Shortcut tiles used'],
  ].filter(([k]) => series[k] && series[k].some((v) => v));
  if (hazardDefs.length) {
    charts.push(
      <ChartCard key="hz" title="Hazard events per episode">
        <Lines rows={epRows} defs={hazardDefs.map(([k, c]) => [k, c, 1.5])} />
      </ChartCard>,
    );
  }

  return (
    <div className="stack">
      {(summary || evalSummary) && (
        <div className="row-gap">
          {summary?.episodes != null && <span className="chip">EPISODES {summary.episodes}</span>}
          {summary?.best_reward != null && (
            <span className="chip gold"
              title={summary.best_episode ? `episode ${summary.best_episode} · ${summary.best_steps} steps` : ''}>
              BEST {Math.round(summary.best_reward)}
              {summary.best_episode ? ` @ EP ${summary.best_episode}` : ''}
            </span>
          )}
          {summary?.avg_reward_last50 != null && (
            <span className="chip cyan">AVG(50) {Math.round(summary.avg_reward_last50)}</span>
          )}
          {summary?.success_rate_last50 != null && (
            <span className="chip green">
              TRAIN SUCCESS {(summary.success_rate_last50 * 100).toFixed(0)}%
            </span>
          )}
          {evalSummary?.success_rate != null && (
            <span className="chip green">
              GREEDY EVAL {(evalSummary.success_rate * 100).toFixed(0)}%
            </span>
          )}
          {summary?.train_time != null && (
            <span className="chip">⏱ {summary.train_time}s</span>
          )}
          {summary?.stopped && <span className="chip amber">STOPPED EARLY</span>}
        </div>
      )}
      <div className="chart-grid">{charts}</div>
    </div>
  );
}
