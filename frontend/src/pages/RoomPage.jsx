import React, { useEffect, useRef, useState } from 'react';
import { Link, useParams, useSearchParams } from 'react-router-dom';
import { api } from '../api.js';
import TrainingControls from '../components/TrainingControls.jsx';
import MetricsDashboard from '../components/MetricsDashboard.jsx';
import EpisodeReplay from '../components/EpisodeReplay.jsx';
import PolicyView from '../components/PolicyView.jsx';
import PacmanView from '../rooms/PacmanView.jsx';
import HeistView from '../rooms/HeistView.jsx';
import RacingView from '../rooms/RacingView.jsx';
import FootballView from '../rooms/FootballView.jsx';
import CrossRoadView from '../rooms/CrossRoadView.jsx';

const VIEWS = { 1: PacmanView, 2: HeistView, 3: RacingView, 4: FootballView, 5: CrossRoadView };
const TAB_KEYS = { overview: 'OVERVIEW', train: 'TRAIN', replay: 'REPLAY', policy: 'POLICY' };

function Difficulty({ level }) {
  return (
    <span className="difficulty">
      {'◆'.repeat(level)}<span className="off">{'◆'.repeat(5 - level)}</span>
    </span>
  );
}

export default function RoomPage() {
  const { id } = useParams();
  const roomId = parseInt(id, 10);
  const [searchParams] = useSearchParams();
  const [room, setRoom] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [status, setStatus] = useState(null);
  const [tab, setTab] = useState('OVERVIEW');
  const [evalBusy, setEvalBusy] = useState(false);
  const [evalResult, setEvalResult] = useState(null);
  const [replayKey, setReplayKey] = useState(0);
  const [error, setError] = useState('');
  const pollRef = useRef(null);

  const tabs = roomId <= 3
    ? ['OVERVIEW', 'TRAIN', 'REPLAY', 'POLICY']
    : ['OVERVIEW', 'TRAIN', 'REPLAY'];

  const loadAll = async () => {
    try {
      const [detail, met, st] = await Promise.all([
        api.room(roomId), api.metrics(roomId), api.status(roomId),
      ]);
      setRoom(detail);
      setMetrics(met.trained ? met : null);
      setStatus(st);
      if (st.state === 'running') startPolling();
    } catch (e) {
      setError(String(e.message || e));
    }
  };

  useEffect(() => {
    const wanted = TAB_KEYS[searchParams.get('tab')] || 'OVERVIEW';
    setTab(wanted);
    setRoom(null);
    setMetrics(null);
    setEvalResult(null);
    setError('');
    loadAll();
    return stopPolling;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [roomId]);

  const stopPolling = () => {
    if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; }
  };

  const startPolling = () => {
    stopPolling();
    pollRef.current = setInterval(async () => {
      try {
        const st = await api.status(roomId);
        setStatus(st);
        if (st.state !== 'running') {
          stopPolling();
          const met = await api.metrics(roomId);
          setMetrics(met.trained ? met : null);
          api.room(roomId).then(setRoom).catch(() => {});
          setReplayKey((k) => k + 1); // refresh the replay list
        }
      } catch { /* keep polling */ }
    }, 700);
  };

  const onTrain = async (params) => {
    setError('');
    setEvalResult(null);
    try {
      await api.train(roomId, params);
      setStatus({ state: 'running', episode: 0, total: params.episodes || 0, message: 'starting…' });
      startPolling();
    } catch (e) { setError(String(e.message || e)); }
  };

  const onStop = async () => { try { await api.stop(roomId); } catch { /* noop */ } };

  const onEvaluate = async () => {
    setEvalBusy(true);
    setError('');
    try {
      const res = await api.evaluate(roomId, 10);
      setEvalResult(res.result);
      const met = await api.metrics(roomId);
      setMetrics(met.trained ? met : null);
      setReplayKey((k) => k + 1);
    } catch (e) { setError(String(e.message || e)); }
    setEvalBusy(false);
  };

  if (!room) {
    return (
      <main className="page">
        <p className="mono" style={{ color: 'var(--muted)' }}>
          LOADING ROOM {roomId}<span className="blink-slow">▌</span>
        </p>
        {error && (
          <p style={{ color: 'var(--red)' }}>
            {error} — is the backend running on port 8000?
          </p>
        )}
      </main>
    );
  }

  const View = VIEWS[roomId];
  const running = status?.state === 'running';
  const liveSeries = running && status.series ? status.series : metrics?.series;
  const liveSummary = running ? null : metrics?.summary;

  return (
    <main className="page" data-room={roomId}>
      <div className="row-gap" style={{ marginBottom: 6 }}>
        <Link to="/rooms" className="hud-link" style={{ paddingLeft: 0 }}>← ALL ROOMS</Link>
      </div>
      <div className="room-header">
        <h1>
          <span className="num">{String(roomId).padStart(2, '0')}</span> {room.name}
        </h1>
        <span className="chip accent">{room.algorithm}</span>
        <Difficulty level={room.difficulty} />
        {room.trained && !running && <span className="chip green">✔ TRAINED</span>}
        {running && <span className="chip amber">● TRAINING</span>}
      </div>

      <div className="tabs">
        {tabs.map((t) => (
          <button key={t} className={`tab ${tab === t ? 'active' : ''}`}
            onClick={() => setTab(t)}>
            {t}
            {t === 'TRAIN' && running && ' ●'}
          </button>
        ))}
      </div>

      {error && (
        <div className="panel" style={{ borderColor: 'var(--red)', marginBottom: 16 }}>
          <span style={{ color: 'var(--red)' }}>⚠ {error}</span>
        </div>
      )}

      {tab === 'OVERVIEW' && <View room={room} />}

      {tab === 'TRAIN' && (
        <div className="stack fade-in">
          <TrainingControls room={room} status={status}
            onTrain={onTrain} onStop={onStop}
            onEvaluate={onEvaluate} busyEval={evalBusy} />
          {evalResult && (
            <div className="panel">
              <h3 className="panel-title">EVALUATION REPORT — 10 GREEDY EPISODES</h3>
              <div className="row-gap">
                <span className="chip green">
                  SUCCESS {(evalResult.success_rate * 100).toFixed(0)}%
                </span>
                <span className="chip cyan">AVG REWARD {Math.round(evalResult.avg_reward)}</span>
                <span className="chip">AVG STEPS {Math.round(evalResult.avg_steps)}</span>
                <span className="subtle">
                  Each evaluation episode was recorded — watch them in the Replay tab.
                </span>
              </div>
            </div>
          )}
          <MetricsDashboard series={liveSeries} summary={liveSummary}
            evalSummary={metrics?.eval} />
        </div>
      )}

      {tab === 'REPLAY' && (
        <div className="fade-in">
          <EpisodeReplay key={replayKey} room={room} />
        </div>
      )}

      {tab === 'POLICY' && (
        <div className="fade-in">
          <PolicyView key={`${replayKey}-p`} room={room} />
        </div>
      )}
    </main>
  );
}
