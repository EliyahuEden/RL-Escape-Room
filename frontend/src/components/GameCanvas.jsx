import React, { useEffect, useRef } from 'react';
import { renderScene } from '../render/renderers.js';

/**
 * The game viewport. Renders continuously with requestAnimationFrame so
 * every room stays alive (pulsing coins, camera sweeps, moving traffic
 * dashes) even when the replay is paused. Dynamic props are read through
 * a ref to avoid re-mounting the loop.
 */
export default function GameCanvas({ layout, frames, cursor = 0,
                                     overlay = '', playing = false, size = 620 }) {
  const canvasRef = useRef(null);
  const stateRef = useRef({});
  stateRef.current = { layout, frames, cursor };

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    let raf;
    const draw = () => {
      const s = stateRef.current;
      renderScene(ctx, canvas.width, canvas.height, {
        layout: s.layout,
        frames: s.frames,
        cursor: s.cursor,
        timeMs: performance.now(),
      });
      raf = requestAnimationFrame(draw);
    };
    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }, []);

  return (
    <div className="canvas-wrap">
      <div className="canvas-hud">
        <span>{overlay}</span>
        {playing && <span className="rec-dot">&#9679; REPLAY</span>}
      </div>
      <canvas ref={canvasRef} width={size} height={size} />
    </div>
  );
}
