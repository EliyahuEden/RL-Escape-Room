import React from 'react';
import { Link, NavLink } from 'react-router-dom';

const LINKS = [
  ['/', 'Home'],
  ['/rooms', 'Rooms'],
  ['/algorithms', 'Algorithms'],
  ['/results', 'Results'],
];

/** Fixed top HUD + ambient arcade background behind every page. */
export default function Layout({ training, children }) {
  return (
    <>
      <div className="arcade-bg" />
      <header className="hud">
        <Link to="/" className="hud-logo">
          <span className="logo-mark">🕹️</span>
          RL <b>ESCAPE ROOM</b>
        </Link>
        <nav className="hud-nav">
          {LINKS.map(([to, name]) => (
            <NavLink key={to} to={to} end={to === '/'}
              className={({ isActive }) => `hud-link ${isActive ? 'active' : ''}`}>
              {name}
            </NavLink>
          ))}
        </nav>
        <div className="hud-status">
          <span className={`led ${training ? 'busy' : ''}`} />
          {training ? 'AGENT TRAINING…' : 'SYSTEMS READY'}
        </div>
      </header>
      {children}
    </>
  );
}
