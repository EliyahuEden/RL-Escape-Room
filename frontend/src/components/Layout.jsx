import React, { useEffect, useState } from 'react';
import { Link, NavLink } from 'react-router-dom';

const LINKS = [
  ['/', 'Home'],
  ['/rooms', 'Rooms'],
  ['/algorithms', 'Algorithms'],
  ['/results', 'Results'],
];

/** Fixed top HUD + ambient arcade background behind every page. */
export default function Layout({ training, children }) {
  // index.html applies the saved theme before first paint; this just mirrors it
  const [theme, setTheme] = useState(
    () => localStorage.getItem('theme') || 'dark',
  );

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem('theme', theme);
  }, [theme]);

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
        <button
          className="theme-toggle"
          onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
          title={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
          aria-label="Toggle color theme">
          {theme === 'dark' ? '☀️' : '🌙'}
        </button>
      </header>
      {children}
    </>
  );
}
