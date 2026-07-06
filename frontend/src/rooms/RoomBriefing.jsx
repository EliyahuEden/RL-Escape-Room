import React from 'react';
import GameCanvas from '../components/GameCanvas.jsx';

/**
 * Shared briefing layout used by the five room views: an always-alive
 * canvas preview of the room next to the mission spec (state, actions,
 * rewards) and a themed tile legend.
 */
export default function RoomBriefing({ room, legend, tips }) {
  return (
    <div className="briefing-grid fade-in">
      <div>
        <GameCanvas layout={room.layout} frames={[]} overlay={`${room.name} — LIVE PREVIEW`} />
      </div>
      <div className="stack">
        <div className="panel accent">
          <h3 className="panel-title">MISSION BRIEFING</h3>
          <p className="subtle" style={{ marginTop: 0 }}>{room.description}</p>
          <div className="spec-table">
            <div className="spec-row">
              <span className="k">Algorithm</span>
              <span className="v">{room.algorithm}</span>
            </div>
            <div className="spec-row">
              <span className="k">State</span>
              <span className="v mono" style={{ fontSize: 12.5 }}>{room.state}</span>
            </div>
            <div className="spec-row">
              <span className="k">Actions</span>
              <span className="v mono" style={{ fontSize: 12.5 }}>{room.actions}</span>
            </div>
            <div className="spec-row">
              <span className="k">Rewards</span>
              <span className="v mono" style={{ fontSize: 12.5 }}>{room.rewards}</span>
            </div>
          </div>
        </div>
        <div className="panel">
          <h3 className="panel-title">ROOM LEGEND</h3>
          <div className="legend-grid">
            {legend.map(([color, text]) => (
              <div className="legend-item" key={text}>
                <span className="legend-swatch" style={{ background: color }} />
                {text}
              </div>
            ))}
          </div>
        </div>
        {tips && (
          <div className="panel">
            <h3 className="panel-title">WHAT THE AGENT MUST LEARN</h3>
            <ul className="subtle" style={{ margin: 0, paddingLeft: 18, lineHeight: 1.8 }}>
              {tips.map((t) => <li key={t}>{t}</li>)}
            </ul>
          </div>
        )}
      </div>
    </div>
  );
}
