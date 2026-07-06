import React, { useState } from 'react';
import { api } from '../api.js';

/**
 * Schema-driven hyperparameter console + train/stop/evaluate actions.
 * The schema comes from the backend room registry, so adding a parameter
 * in Python automatically grows a control here.
 */
export default function TrainingControls({ room, status, onTrain, onStop,
                                           onEvaluate, busyEval }) {
  const [values, setValues] = useState(() => ({ ...(room.values || {}) }));
  const [saveMsg, setSaveMsg] = useState('');

  const running = status?.state === 'running';
  const set = (k, val) => setValues((v) => ({ ...v, [k]: val }));

  const renderField = (p) => {
    if (p.type === 'bool') {
      return (
        <label key={p.key} className="checkbox-row" title={p.help || ''}>
          <input type="checkbox" checked={!!values[p.key]} disabled={running}
            onChange={(e) => set(p.key, e.target.checked)} />
          {p.label}
        </label>
      );
    }
    if (p.type === 'select') {
      return (
        <div key={p.key} className="field" title={p.help || ''}>
          <label>{p.label}</label>
          <select value={String(values[p.key])} disabled={running}
            onChange={(e) => {
              const raw = e.target.value;
              const opt = p.options.find((o) => String(o) === raw);
              set(p.key, opt !== undefined ? opt : raw);
            }}>
            {p.options.map((o) => (
              <option key={String(o)} value={String(o)}>{String(o)}</option>
            ))}
          </select>
        </div>
      );
    }
    const step = p.step ?? (p.type === 'int' ? 1 : 0.01);
    return (
      <div key={p.key} className="field" title={p.help || ''}>
        <label>{p.label}</label>
        <input type="number" value={values[p.key]}
          min={p.min ?? undefined} max={p.max ?? undefined} step={step}
          disabled={running}
          onChange={(e) => set(p.key, p.type === 'int'
            ? parseInt(e.target.value || 0, 10)
            : parseFloat(e.target.value || 0))} />
        {p.min !== null && p.max !== null && (
          <input type="range" value={values[p.key]}
            min={p.min} max={p.max} step={step} disabled={running}
            onChange={(e) => set(p.key, p.type === 'int'
              ? parseInt(e.target.value, 10) : parseFloat(e.target.value))} />
        )}
      </div>
    );
  };

  const saveConfig = async () => {
    try {
      await api.saveConfig(room.id, values);
      setSaveMsg('saved ✔');
    } catch (e) {
      setSaveMsg(`save failed: ${e.message}`);
    }
    setTimeout(() => setSaveMsg(''), 2500);
  };

  const pctDone = status?.total ? Math.round((status.episode / status.total) * 100) : 0;

  return (
    <div className="panel accent">
      <h3 className="panel-title">TRAINING CONSOLE — {room.algorithm}</h3>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(185px, 1fr))', gap: 14 }}>
        {room.params.map(renderField)}
      </div>

      <div className="row-gap" style={{ marginTop: 18 }}>
        <button className="btn btn-primary" disabled={running}
          onClick={() => onTrain(values)}>
          ▸ START TRAINING
        </button>
        <button className="btn btn-danger" disabled={!running} onClick={onStop}>
          ■ STOP
        </button>
        <button className="btn btn-green" disabled={running || !room.trained || busyEval}
          onClick={() => onEvaluate()}
          title="Run 10 greedy episodes with the saved model and record them as replays">
          {busyEval ? 'EVALUATING…' : '✓ EVALUATE POLICY'}
        </button>
        <span className="spacer" />
        <button className="btn btn-sm" disabled={running} onClick={saveConfig}
          title="Persist these hyperparameters as the room's default config">
          SAVE CONFIG
        </button>
        <button className="btn btn-sm" disabled={running}
          onClick={() => {
            const v = {};
            room.params.forEach((p) => { v[p.key] = p.default; });
            setValues(v);
          }}>
          RESET DEFAULTS
        </button>
        {saveMsg && <span className="subtle mono">{saveMsg}</span>}
      </div>

      {status && status.state !== 'idle' && (
        <div style={{ marginTop: 18 }}>
          <div className="row-gap" style={{ marginBottom: 6 }}>
            <span className={`chip ${running ? 'amber' : status.state === 'finished' ? 'green' : status.state === 'error' ? 'red' : 'cyan'}`}>
              {status.state}
            </span>
            <span className="subtle">{status.message}</span>
            <span className="spacer" />
            {status.elapsed != null && <span className="subtle mono">{status.elapsed}s</span>}
            <span className="subtle mono">
              {status.episode}/{status.total} episodes
            </span>
          </div>
          <div className="progress-track">
            <div className={`progress-fill ${running && !pctDone ? 'striped' : ''}`}
              style={{ width: `${running && !pctDone ? 100 : pctDone}%` }} />
          </div>
          {status.error && (
            <div style={{ color: 'var(--red)', marginTop: 8, fontSize: 13 }}>
              ERROR: {status.error}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
