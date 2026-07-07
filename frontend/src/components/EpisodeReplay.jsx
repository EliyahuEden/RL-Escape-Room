import React, { useEffect, useRef, useState } from 'react';
import { api, GOOD_EVENTS } from '../api.js';
import GameCanvas from './GameCanvas.jsx';

const SPEEDS = [0.25, 0.5, 1, 2, 4, 8];
// playback pace in frames/second at 1x, per room family
const BASE_FPS = { pacman: 7, museum: 7, racing: 8, football: 13, crossroad: 13 };

/**
 * Episode replay viewer: pick a recorded episode (training milestones +
 * greedy evaluations), watch the agent frame by frame with play / pause /
 * step / speed / scrubbing, and a live state-action-reward readout.
 */
export default function EpisodeReplay({ room }) {
  const [episodes, setEpisodes] = useState([]);
  const [selected, setSelected] = useState(null);
  const [replay, setReplay] = useState(null);
  const [cursor, setCursor] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const rafRef = useRef();
  const lastT = useRef(0);

  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const { episodes: eps } = await api.replays(room.id);
        if (!alive) return;
        setEpisodes(eps);
        if (eps.length) pick(eps[eps.length - 1].id);
      } catch { /* backend offline */ }
    })();
    return () => { alive = false; };
  }, [room.id]);

  const pick = async (id) => {
    setSelected(id);
    setPlaying(false);
    setCursor(0);
    try {
      const data = await api.replay(room.id, id);
      setReplay(data);
      setPlaying(true);
    } catch { setReplay(null); }
  };

  // playback clock (rAF, delta-timed)
  useEffect(() => {
    if (!playing || !replay) return undefined;
    lastT.current = performance.now();
    const fps = BASE_FPS[room.type] || 8;
    const tick = (now) => {
      const dt = (now - lastT.current) / 1000;
      lastT.current = now;
      setCursor((c) => {
        const next = c + dt * fps * speed;
        if (next >= replay.frames.length - 1) {
          setPlaying(false);
          return replay.frames.length - 1;
        }
        return next;
      });
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafRef.current);
  }, [playing, speed, replay, room.type]);

  const frames = replay?.frames || [];
  const fi = Math.min(frames.length - 1, Math.floor(cursor));
  const frame = frames[fi];
  const actionNames = room.action_names || [];
  const effLayout = replay?.meta?.layout || room.layout;
  const isGrid = room.id <= 3;

  const terminalLabel = replay?.meta
    ? (replay.meta.success ? '✔ SUCCESS' : `✖ ${replay.meta.fail_reason || 'failed'}`)
    : '';

  return (
    <div className="replay-layout">
      {/* -------- episode list -------- */}
      <div className="panel">
        <h3 className="panel-title">RECORDED EPISODES</h3>
        <div className="replay-list">
          {episodes.length === 0 && (
            <span className="subtle">No recordings yet — train this room first.</span>
          )}
          {episodes.map((e) => (
            <div key={e.id}
              className={`replay-item ${selected === e.id ? 'selected' : ''}`}
              onClick={() => pick(e.id)}>
              <div className="row-gap" style={{ justifyContent: 'space-between' }}>
                <span className={`chip ${e.kind === 'eval' ? 'green' : 'amber'}`}
                  style={{ fontSize: 9.5 }}>
                  {e.kind === 'eval' ? 'EVAL' : 'TRAINING'}
                </span>
                <span className={`rw ${e.reward >= 0 ? 'pos' : 'neg'}`}>
                  {Math.round(e.reward)}
                </span>
              </div>
              <div style={{ fontSize: 12.5, marginTop: 4 }}>{e.label}</div>
              <div className="subtle" style={{ fontSize: 11, marginTop: 2 }}>
                {e.steps} steps · {e.success ? '✔ success' : `✖ ${e.fail_reason || 'failed'}`}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* -------- viewport + transport controls -------- */}
      <div>
        <GameCanvas layout={effLayout} frames={frames} cursor={cursor}
          playing={playing}
          overlay={replay ? `${room.name} — ${replay.meta.label || selected}` : room.name} />
        <div className="playbar">
          <button className="btn btn-sm" onClick={() => setPlaying((p) => !p)}
            disabled={!replay} title={playing ? 'Pause' : 'Play'}>
            {playing ? '❚❚' : '▶'}
          </button>
          <button className="speed-btn" disabled={!replay} title="Reset to start"
            onClick={() => { setPlaying(false); setCursor(0); }}>
            ⟲
          </button>
          <button className="speed-btn" disabled={!replay} title="Step back"
            onClick={() => { setPlaying(false); setCursor((c) => Math.max(0, Math.floor(c) - 1)); }}>
            −1
          </button>
          <button className="speed-btn" disabled={!replay} title="Step forward"
            onClick={() => { setPlaying(false); setCursor((c) => Math.min(frames.length - 1, Math.floor(c) + 1)); }}>
            +1
          </button>
          <input type="range" min={0} max={Math.max(0, frames.length - 1)}
            step={0.01} value={cursor} disabled={!replay}
            onChange={(e) => { setPlaying(false); setCursor(parseFloat(e.target.value)); }} />
          {SPEEDS.map((s) => (
            <button key={s} className={`speed-btn ${speed === s ? 'active' : ''}`}
              onClick={() => setSpeed(s)}>
              {s}×
            </button>
          ))}
        </div>
      </div>

      {/* -------- state monitor -------- */}
      <div className="panel">
        <h3 className="panel-title">STATE MONITOR</h3>
        {!frame ? (
          <span className="subtle">select an episode</span>
        ) : (
          <div className="statline">
            <div className="row"><span className="k">STEP</span>
              <span className="v">{frame.t} / {replay?.meta?.steps ?? '—'}</span></div>
            <div className="row"><span className="k">ACTION</span>
              <span className="v" style={{ color: 'var(--sky)' }}>
                {frame.a === null || frame.a === undefined
                  ? (frame.fl ? 'ball in flight' : '—')
                  : (actionNames[frame.a] ?? frame.a)}
              </span></div>
            <div className="row"><span className="k">REWARD</span>
              <span className="v" style={{ color: frame.r >= 0 ? 'var(--green)' : 'var(--red)' }}>
                {frame.r > 0 ? '+' : ''}{frame.r}
              </span></div>
            <div className="row"><span className="k">CUMULATIVE</span>
              <span className="v" style={{ color: frame.cum >= 0 ? 'var(--green)' : 'var(--red)' }}>
                {frame.cum > 0 ? '+' : ''}{frame.cum}
              </span></div>
            {frame.p && (
              <div className="row"><span className="k">{isGrid ? 'CELL (row, col)' : 'POSITION (x, y)'}</span>
                <span className="v">
                  ({Number(frame.p[0]).toFixed(isGrid ? 0 : 2)},
                  {' '}{Number(frame.p[1]).toFixed(isGrid ? 0 : 2)})
                </span></div>
            )}
            {'coins' in frame && (
              <div className="row"><span className="k">COINS LEFT</span>
                <span className="v">{frame.coins.length} {frame.open ? '· DOOR OPEN' : '· door locked'}</span></div>
            )}
            {'d' in frame && (
              <div className="row"><span className="k">DIAMOND</span>
                <span className="v">{frame.d ? '✔ STOLEN' : '✖ in the vault'}</span></div>
            )}
            {'alarm' in frame && frame.alarm && (
              <div className="row"><span className="k">ALARM</span>
                <span className="v" style={{ color: 'var(--red)' }}>⚠ ACTIVE</span></div>
            )}
            {'ncp' in frame && (
              <div className="row"><span className="k">CHECKPOINTS</span>
                <span className="v">{frame.ncp} ✓ {frame.open ? '· FINISH OPEN' : '· finish locked'}</span></div>
            )}
            {'rv' in frame && (
              <div className="row"><span className="k">RIVAL (SARSA)</span>
                <span className="v">({frame.rv[0]}, {frame.rv[1]})</span></div>
            )}
            {'shoot' in frame && (
              <div className="row"><span className="k">SHOOTING ZONE</span>
                <span className="v">{frame.shoot ? '✔ INSIDE' : '✖ outside'}</span></div>
            )}
            {'cars' in frame && (
              <div className="row"><span className="k">CARS TRACKED</span>
                <span className="v">{frame.cars.length}</span></div>
            )}
            <div className="row"><span className="k">TERMINAL</span>
              <span className="v">{frame.done ? terminalLabel : 'running…'}</span></div>
            <div style={{ marginTop: 6, minHeight: 30 }}>
              {(frame.ev || []).map((e) => (
                <span key={e}
                  className={`event-badge ${GOOD_EVENTS.has(e.toLowerCase()) ? 'good' : ''}`}>
                  {e}
                </span>
              ))}
            </div>
            {replay?.meta && (
              <div className="subtle" style={{ fontSize: 11 }}>
                episode reward {Math.round(replay.meta.reward)} · {replay.meta.steps} steps ·{' '}
                {terminalLabel}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
