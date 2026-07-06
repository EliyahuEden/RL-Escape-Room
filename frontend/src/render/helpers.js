/* Shared canvas drawing utilities for all room renderers. */

export const COL = {
  floor: '#0c1226', floorLine: '#141b36', wall: '#1d2648', wallTop: '#293359',
  text: '#dbe2f4', muted: '#7e88ab',
  gold: '#ffd23f', violet: '#a78bfa', red: '#ff4d5a', green: '#34d399',
  cyan: '#38bdf8', pink: '#f472b6', amber: '#fbbf24', ice: '#7dd3fc',
};

export function hash2(r, c) {
  const n = Math.sin(r * 127.1 + c * 311.7) * 43758.5453;
  return n - Math.floor(n);
}

export function lerp(a, b, t) { return a + (b - a) * t; }

/** [prevFrame, nextFrame, subFrameFraction] at a float cursor. */
export function frameAt(frames, cursor) {
  if (!frames || !frames.length) return [null, null, 0];
  const i0 = Math.max(0, Math.min(frames.length - 1, Math.floor(cursor)));
  const i1 = Math.min(frames.length - 1, i0 + 1);
  return [frames[i0], frames[i1], Math.min(1, Math.max(0, cursor - i0))];
}

/** Interpolate a [a,b] position; big jumps (teleports/resets) snap. */
export function lerpPos(p0, p1, t, snapDist = 3.0) {
  if (!p0) return p1;
  if (!p1) return p0;
  const d = Math.abs(p0[0] - p1[0]) + Math.abs(p0[1] - p1[1]);
  if (d > snapDist) return t < 0.5 ? p0 : p1;
  return [lerp(p0[0], p1[0], t), lerp(p0[1], p1[1], t)];
}

export function glow(ctx, x, y, r, color, alpha = 0.5) {
  const g = ctx.createRadialGradient(x, y, 0, x, y, r);
  g.addColorStop(0, color);
  g.addColorStop(1, 'transparent');
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.fillStyle = g;
  ctx.fillRect(x - r, y - r, r * 2, r * 2);
  ctx.restore();
}

export function roundRect(ctx, x, y, w, h, r) {
  const rr = Math.min(r, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + rr, y);
  ctx.arcTo(x + w, y, x + w, y + h, rr);
  ctx.arcTo(x + w, y + h, x, y + h, rr);
  ctx.arcTo(x, y + h, x, y, rr);
  ctx.arcTo(x, y, x + w, y, rr);
  ctx.closePath();
}

export function label(ctx, x, y, text, color = COL.muted, size = 10, font = 'JetBrains Mono') {
  ctx.save();
  ctx.font = `${size}px "${font}", monospace`;
  ctx.fillStyle = color;
  ctx.textAlign = 'center';
  ctx.globalAlpha = 0.85;
  ctx.fillText(text, x, y);
  ctx.restore();
}

/** "+10" / "-50" reward popup floating above the agent. */
export function floatingReward(ctx, x, y, rw, t) {
  if (rw === undefined || rw === null || Math.abs(rw) < 2) return;
  ctx.save();
  ctx.globalAlpha = Math.max(0, 0.95 - t);
  ctx.font = 'bold 14px "Chakra Petch", sans-serif';
  ctx.textAlign = 'center';
  ctx.fillStyle = rw > 0 ? COL.green : COL.red;
  ctx.shadowColor = rw > 0 ? COL.green : COL.red;
  ctx.shadowBlur = 9;
  ctx.fillText(`${rw > 0 ? '+' : ''}${Math.round(rw)}`, x, y - 18 - t * 16);
  ctx.restore();
}

export function fullFlash(ctx, W, H, color, alpha) {
  if (alpha <= 0) return;
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.fillStyle = color;
  ctx.fillRect(0, 0, W, H);
  ctx.restore();
}

/** Big centred outcome text ("GOAL!", "ESCAPED", "CRASHED"). */
export function outcomeText(ctx, W, H, text, color, timeMs) {
  const pulse = 1 + 0.04 * Math.sin(timeMs / 160);
  ctx.save();
  ctx.translate(W / 2, H / 2);
  ctx.scale(pulse, pulse);
  ctx.font = 'bold 42px "Chakra Petch", sans-serif';
  ctx.textAlign = 'center';
  ctx.lineWidth = 7;
  ctx.strokeStyle = 'rgba(0,0,0,0.65)';
  ctx.strokeText(text, 0, 8);
  ctx.fillStyle = color;
  ctx.shadowColor = color;
  ctx.shadowBlur = 26;
  ctx.fillText(text, 0, 8);
  ctx.restore();
}

/** Fading movement trail (list of canvas-space [x, y] points). */
export function drawTrail(ctx, pts, r, color) {
  for (let i = 0; i < pts.length; i++) {
    const a = (i / pts.length) * 0.30;
    ctx.save();
    ctx.globalAlpha = a;
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(pts[i][0], pts[i][1], r * (0.28 + 0.5 * (i / pts.length)), 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  }
}

/** Collect the last `n` frame positions before the cursor (canvas space). */
export function trailPoints(frames, cursor, n, toXY) {
  const pts = [];
  const end = Math.floor(cursor);
  for (let i = Math.max(0, end - n); i < end; i++) {
    const f = frames[i];
    if (f && f.p) pts.push(toXY(f.p));
  }
  return pts;
}

/** Success / failure classification of a terminal frame. */
export function outcomeOf(frame) {
  if (!frame || !frame.done) return null;
  const ev = frame.ev || [];
  const good = ev.some((e) => ['escaped', 'finished', 'crossed', 'success'].includes(e)
    || e.toLowerCase().startsWith('goal'));
  return good ? 'good' : 'bad';
}
