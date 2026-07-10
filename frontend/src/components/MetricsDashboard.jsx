import React from 'react';
import {
  Bar, BarChart, CartesianGrid, Legend, Line, LineChart,
  ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts';

// theme-aware via CSS variables (resolved live, so the toggle re-skins charts)
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
          defs={[['dp_delta', 'var(--red)', 2]]} logY />
      </ChartCard>,
      <ChartCard key="dpv" title="Value of the start state per sweep">
        <Lines rows={dpRows} defs={[['dp_start_value', 'var(--sky)', 2]]} />
      </ChartCard>,
    );
    if (series.dp_policy_changes) {
      charts.push(
        <ChartCard key="dpc" title="Policy changes per iteration">
          <Lines rows={dpRows} defs={[['dp_policy_changes', 'var(--amber)', 2]]} />
        </ChartCard>,
      );
    }
  }

  const epRows = toRows(series, 'episode', [
    'reward', 'reward_avg', 'steps', 'epsilon', 'success_rate', 'failure_rate',
    'td_error', 'loss', 'camera_hits', 'caught', 'trap_hits', 'crashes', 'shortcuts',
    'crash_rate', 'rival_crash_rate', 'rival_reward_avg', 'rival_success_rate',
  ]);
  const hasRival = !!series.rival_reward_avg;

  if (series.reward) {
    charts.push(
      <ChartCard key="rw"
        title={series.dp_delta ? 'Greedy evaluation reward per episode'
          : hasRival ? 'Reward: Q-Learning car vs SARSA rival (moving average)'
            : 'Reward per episode (+ moving average)'}>
        <Lines rows={epRows}
          defs={[['reward', 'rgba(139,92,246,0.4)', 1], ['reward_avg', 'var(--gold)', 2.2],
            ...(hasRival ? [['rival_reward_avg', 'var(--purple)', 2]] : [])]} />
      </ChartCard>,
    );
  }
  if (series.steps) {
    charts.push(
      <ChartCard key="st" title="Steps per episode">
        <Lines rows={epRows} defs={[['steps', 'var(--sky)', 1.4]]} />
      </ChartCard>,
    );
  }
  if (series.success_rate) {
    charts.push(
      <ChartCard key="sr" title={hasRival
        ? 'Success rate: Q-Learning car vs SARSA rival (rolling 50)'
        : 'Success / failure rate (rolling 50)'}>
        <Lines rows={epRows}
          defs={[['success_rate', 'var(--green)', 2],
            ...(hasRival ? [['rival_success_rate', 'var(--purple)', 1.8]]
              : [['failure_rate', 'rgba(255,77,90,0.6)', 1.4]])]} />
      </ChartCard>,
    );
  }
  if (series.crash_rate) {
    charts.push(
      <ChartCard key="crash" title={hasRival
        ? 'Crashes into the cliff during training — Q-Learning vs SARSA rival (rolling 75)'
        : 'Crashes into the cliff during training (rolling 75)'}>
        <Lines rows={epRows}
          defs={[['crash_rate', 'var(--red)', 2],
            ...(hasRival ? [['rival_crash_rate', 'var(--purple)', 1.8]] : [])]} />
      </ChartCard>,
    );
  }
  if (series.epsilon) {
    charts.push(
      <ChartCard key="eps" title="Exploration rate ε over time">
        <Lines rows={epRows} defs={[['epsilon', 'var(--purple)', 2]]} />
      </ChartCard>,
    );
  }
  if (series.td_error) {
    charts.push(
      <ChartCard key="td" title="Mean |TD error| per episode">
        <Lines rows={epRows} defs={[['td_error', 'var(--pink)', 1.6]]} />
      </ChartCard>,
    );
  }
  if (series.loss && series.loss.some((v) => v)) {
    charts.push(
      <ChartCard key="loss" title="DQN training loss (per-episode mean)">
        <Lines rows={epRows.filter((r) => r.loss !== undefined)}
          defs={[['loss', 'var(--orange)', 1.6]]} />
      </ChartCard>,
    );
  }
  // (crashes get their own rolling-rate chart above; keep the rest here)
  const hazardDefs = [
    ['camera_hits', 'var(--red)', 'Camera detections'],
    ['caught', 'var(--red-2)', 'Caught by guards'],
    ['trap_hits', 'var(--orange)', 'Trap hits'],
    ['shortcuts', 'var(--sky)', 'Shortcut tiles used'],
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
          {evalSummary?.beat_rival != null && (
            <span className="chip violet">
              🏁 RACES WON {(evalSummary.beat_rival * 100).toFixed(0)}%
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
