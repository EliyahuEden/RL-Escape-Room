/* ============================================================
   renderScene(ctx, W, H, scene) — dispatch to the per-room
   canvas renderer.

   scene = { layout, frames, cursor, timeMs }
   All ambient animation derives from timeMs, so rooms stay
   alive (pulsing coins, camera sweeps, traffic dashes) even
   while the replay is paused or idle.
   ============================================================ */
import { renderMuseum, renderPacman, renderRacing } from './gridRooms.js';
import { renderCrossroad, renderFootball } from './continuous.js';

const RENDERERS = {
  pacman: renderPacman,
  museum: renderMuseum,
  racing: renderRacing,
  football: renderFootball,
  crossroad: renderCrossroad,
};

/** Synthetic first frame so an untrained room still shows a live scene. */
function idleFrame(layout) {
  switch (layout.type) {
    case 'pacman':
      return { t: 0, r: 0, cum: 0, ev: [], p: layout.start, coins: layout.coins,
               guard: layout.guard ? layout.guard.route[0] : null, open: false };
    case 'museum':
      return { t: 0, r: 0, cum: 0, ev: [], p: layout.start, d: 0,
               guards: (layout.guard_routes || []).map((rt) => rt[0]), alarm: false };
    case 'racing':
      return { t: 0, r: 0, cum: 0, ev: [], p: layout.start,
               b: layout.boosters, open: false };
    case 'football':
      return { t: 0, r: 0, cum: 0, ev: [], p: layout.kick || [1.0, 5.0],
               defs: [], keeper: [layout.keeper_x, (layout.goal_lo + layout.goal_hi) / 2],
               shoot: layout.mode === 'freekick' };
    case 'crossroad':
      return { t: 0, r: 0, cum: 0, ev: [], p: layout.start || [0.45, 5.0], cars: [] };
    default:
      return { t: 0, r: 0, cum: 0, ev: [] };
  }
}

export function renderScene(ctx, W, H, scene) {
  ctx.clearRect(0, 0, W, H);
  const { layout } = scene;
  if (!layout || !RENDERERS[layout.type]) {
    ctx.fillStyle = '#05070f';
    ctx.fillRect(0, 0, W, H);
    return;
  }
  let { frames } = scene;
  if (!frames || !frames.length) frames = [idleFrame(layout)];
  RENDERERS[layout.type](ctx, W, H, { ...scene, frames });
}
