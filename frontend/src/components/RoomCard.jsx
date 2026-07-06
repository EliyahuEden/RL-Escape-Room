import React from 'react';
import { Link, useNavigate } from 'react-router-dom';

const ICONS = { pacman: '🟡', diamond: '💎', car: '🏎️', ball: '⚽', chicken: '🐔' };

function Difficulty({ level }) {
  return (
    <span className="difficulty" title={`difficulty ${level}/5`}>
      {'◆'.repeat(level)}
      <span className="off">{'◆'.repeat(5 - level)}</span>
    </span>
  );
}

function pct(x) {
  return x === null || x === undefined ? '—' : `${Math.round(x * 100)}%`;
}

/** Level-select card: stats + Train / Replay / Explanation actions. */
export default function RoomCard({ room }) {
  const navigate = useNavigate();
  const sr = room.eval_success_rate ?? room.success_rate;

  return (
    <div className="room-card" data-room={room.id}>
      <div className="card-scene" onClick={() => navigate(`/room/${room.id}`)}
        style={{ cursor: 'pointer' }}>
        <span className="scene-icon">{ICONS[room.icon] || '🎮'}</span>
        <span className="room-num">{String(room.id).padStart(2, '0')}</span>
        {room.training && <span className="chip red training-tag">● TRAINING</span>}
      </div>
      <div className="card-body">
        <span className="room-sub">{room.subtitle}</span>
        <h3 className="room-name">{room.name}</h3>
        <div className="row-gap">
          <span className="chip accent">{room.algorithm}</span>
          <Difficulty level={room.difficulty} />
        </div>
        <div className="room-desc">{room.description.slice(0, 110)}…</div>
        <div className="room-stats">
          <div className="room-stat">
            <div className="val">{room.trained ? (room.episodes ?? '—') : '—'}</div>
            <div className="lbl">Episodes</div>
          </div>
          <div className="room-stat">
            <div className="val">{room.best_reward ?? '—'}</div>
            <div className="lbl">Best reward</div>
          </div>
          <div className="room-stat">
            <div className={`val ${sr >= 0.5 ? 'good' : ''}`}>{pct(sr)}</div>
            <div className="lbl">Success</div>
          </div>
        </div>
        <div className="row-gap" style={{ marginTop: 2 }}>
          {room.training ? (
            <span className="chip amber">agent in the room…</span>
          ) : room.trained ? (
            <span className="chip green">✔ TRAINED</span>
          ) : (
            <span className="chip">UNTRAINED</span>
          )}
        </div>
        <div className="card-actions">
          <Link className="btn btn-primary" to={`/room/${room.id}?tab=train`}>Train</Link>
          <Link className="btn" to={`/room/${room.id}?tab=replay`}>Replay</Link>
          <Link className="btn" to={`/algorithms#${room.algo_id}`}>Explain</Link>
        </div>
      </div>
    </div>
  );
}
