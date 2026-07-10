import React, { useEffect, useRef, useState } from 'react';
import { api } from '../api.js';
import GameCanvas from './GameCanvas.jsx';

/**
 * Schema-driven hyperparameter console + train/stop/evaluate actions.
 * The schema comes from the backend room registry, so adding a parameter
 * in Python automatically grows a control here.
 *
 * Controls flagged `regen` in the schema (coin / guard / camera / oil counts,
 * map seed …) shape a *generated* layout: changing one flips the map source to
 * "generated" and refreshes the live preview so you can see the difficulty you
 * are dialling in before you train.
 */
export default function TrainingControls({ room, status, onTrain, onStop,
                                           onEvaluate, onPreview, busyEval }) {
  const [values, setValues] = useState(() => ({ ...(room.values || {}) }));
  const [saveMsg, setSaveMsg] = useState('');

  const running = status?.state === 'running';
  const hasMap = room.params.some((p) => p.key === 'map_mode');
  const regenKeys = room.params.filter((p) => p.regen).map((p) => p.key);

  const set = (k, val) => setValues((v) => {
    const nv = { ...v, [k]: val };
    // touching a difficulty/layout count means you want a generated map
    const p = room.params.find((pp) => pp.key === k);
    if (p && p.regen && hasMap && k !== 'map_seed') nv.map_mode = 'generated';
    return nv;
  });

  // debounced live preview: whenever the map source or a regen count changes,
  // rebuild the layout on the backend and repaint the canvas.
  const firstRun = useRef(true);
  const previewSig = hasMap
    ? JSON.stringify([values.map_mode, ...regenKeys.map((k) => values[k])])
    : '';
  useEffect(() => {
    if (!hasMap || !onPreview) return undefined;
    if (firstRun.current) { firstRun.current = false; return undefined; }
    const t = setTimeout(() => onPreview(values), 350);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [previewSig]);

  const newRandomLayout = () => {
    const seed = Math.floor(Math.random() * 9999);
    const nv = { ...values, map_mode: 'generated', map_seed: seed };
    setValues(nv);
    if (onPreview) onPreview(nv);
  };

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
        <label>{p.label}{p.regen && <span className="regen-dot" title="shapes the generated map"> ◇</span>}</label>
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

      {hasMap && (
        <div className="map-studio">
          <div className="map-studio-preview">
            <GameCanvas layout={room.layout} size={260} />
          </div>
          <div className="map-studio-controls">
            <div className="subtle" style={{ marginBottom: 8 }}>
              Map source: <b>{values.map_mode}</b>
              {values.map_mode === 'generated' && <> · seed <b>{values.map_seed}</b></>}
            </div>
            <button className="btn btn-primary" disabled={running}
              onClick={newRandomLayout}
              title="Roll a fresh randomised layout built from the difficulty counts below">
              🎲 New random layout
            </button>
            {values.map_mode === 'generated' && (
              <button className="btn btn-sm" disabled={running}
                style={{ marginTop: 8 }}
                onClick={() => set('map_mode', 'classic')}
                title="Go back to the curated hand-made map">
                ↺ Use curated map
              </button>
            )}
            <p className="subtle" style={{ marginTop: 10, fontSize: 12 }}>
              Controls marked ◇ shape the generated map. Adjust them and the
              preview updates live; press START TRAINING to learn on this map.
            </p>
          </div>
        </div>
      )}

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
            if (hasMap && onPreview) onPreview(v);
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
