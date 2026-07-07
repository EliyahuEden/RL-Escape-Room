/* ============================================================
   Canvas renderers for the three 10x10 grid rooms — each with
   its own visual identity:

     pacman  — neon-blue arcade maze, glowing coins, animated
               mouth, ghost guard, locked/unlocked exit door
     museum  — marble stealth floor, camera sweep zones, laser
               traps, patrolling guards, alarm mode
     racing  — asphalt street circuit, city blocks, oil/mud,
               boosters, checkered finish, rotating car
   ============================================================ */
import {
  COL, drawTrail, floatingReward, frameAt, fullFlash, glow, hash2, label,
  lerp, lerpPos, outcomeOf, outcomeText, roundRect, trailPoints,
} from './helpers.js';

const key = (rc) => `${rc[0]},${rc[1]}`;

function gridGeom(W, H, N) {
  const pad = 26;
  const cell = (Math.min(W, H) - pad * 2) / N;
  return {
    pad,
    cell,
    cx: (c) => pad + c * cell + cell / 2,
    cy: (r) => pad + r * cell + cell / 2,
    tile: (r, c) => [pad + c * cell, pad + r * cell],
  };
}

function agentPos(g, f0, f1, t) {
  const p = lerpPos(f0?.p, f1?.p, t);
  return p ? [g.cx(p[1]), g.cy(p[0])] : null;
}

function terminalFx(ctx, W, H, f0, t, timeMs, goodText, badTextMap) {
  const oc = outcomeOf(f0);
  if (!oc) return;
  const alpha = Math.max(0, 0.18 - t * 0.16);
  fullFlash(ctx, W, H, oc === 'good' ? COL.green : COL.red, alpha);
  const ev = f0.ev || [];
  if (oc === 'good') {
    outcomeText(ctx, W, H, goodText, COL.green, timeMs);
  } else {
    let text = 'FAILED';
    for (const [evName, txt] of Object.entries(badTextMap)) {
      if (ev.includes(evName)) { text = txt; break; }
    }
    outcomeText(ctx, W, H, text, COL.red, timeMs);
  }
}

/* ============================================================
   PACMAN — Room 1
   ============================================================ */
export function renderPacman(ctx, W, H, scene) {
  const { layout, frames, cursor, timeMs } = scene;
  const N = layout.size;
  const g = gridGeom(W, H, N);
  const [f0, f1, t] = frameAt(frames, cursor);
  const walls = new Set((layout.walls || []).map(key));

  // floor: deep arcade blue-black with faint dots
  ctx.fillStyle = '#04060f';
  ctx.fillRect(0, 0, W, H);
  for (let r = 0; r < N; r++) {
    for (let c = 0; c < N; c++) {
      if (walls.has(`${r},${c}`)) continue;
      const [x, y] = g.tile(r, c);
      ctx.fillStyle = '#070b19';
      ctx.fillRect(x, y, g.cell, g.cell);
      ctx.fillStyle = 'rgba(70,90,180,0.10)';
      ctx.fillRect(g.cx(c) - 0.8, g.cy(r) - 0.8, 1.6, 1.6);
    }
  }

  // classic pacman walls: rounded neon-blue blocks with glow
  (layout.walls || []).forEach(([r, c]) => {
    const [x, y] = g.tile(r, c);
    ctx.save();
    ctx.fillStyle = '#0a123a';
    roundRect(ctx, x + 2, y + 2, g.cell - 4, g.cell - 4, 5);
    ctx.fill();
    ctx.strokeStyle = '#2649ff';
    ctx.lineWidth = 1.6;
    ctx.shadowColor = '#2649ff';
    ctx.shadowBlur = 7;
    roundRect(ctx, x + 3, y + 3, g.cell - 6, g.cell - 6, 4);
    ctx.stroke();
    ctx.restore();
  });

  // ice tiles with animated shimmer
  (layout.slippery || []).forEach(([r, c]) => {
    const [x, y] = g.tile(r, c);
    ctx.fillStyle = 'rgba(125,211,252,0.12)';
    ctx.fillRect(x + 1, y + 1, g.cell - 2, g.cell - 2);
    const sh = ((timeMs / 16 + (r * 31 + c * 17)) % (g.cell * 2)) - g.cell;
    ctx.save();
    ctx.beginPath();
    ctx.rect(x, y, g.cell, g.cell);
    ctx.clip();
    ctx.strokeStyle = 'rgba(165,225,255,0.35)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(x + sh, y + g.cell);
    ctx.lineTo(x + sh + g.cell * 0.7, y);
    ctx.stroke();
    ctx.restore();
    label(ctx, g.cx(c), y + g.cell - 4, '❄', 'rgba(165,225,255,0.5)', g.cell * 0.26);
  });

  // guard patrol route
  if (layout.guard && layout.guard.route && layout.guard.route.length > 1) {
    ctx.save();
    ctx.strokeStyle = 'rgba(255,77,90,0.35)';
    ctx.setLineDash([4, 6]);
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    layout.guard.route.forEach(([r, c], i) => {
      if (i === 0) ctx.moveTo(g.cx(c), g.cy(r)); else ctx.lineTo(g.cx(c), g.cy(r));
    });
    ctx.stroke();
    ctx.restore();
  }

  // coins: pulsing gold dots (remaining set comes from the current frame)
  const coins = f0 ? f0.coins : layout.coins;
  (coins || []).forEach(([r, c], i) => {
    const p = 1 + 0.14 * Math.sin(timeMs / 240 + i * 1.7);
    const x = g.cx(c);
    const y = g.cy(r);
    glow(ctx, x, y, g.cell * 0.55, 'rgba(255,210,63,0.7)', 0.4);
    ctx.save();
    ctx.fillStyle = COL.gold;
    ctx.shadowColor = COL.gold;
    ctx.shadowBlur = 10;
    ctx.beginPath();
    ctx.arc(x, y, g.cell * 0.16 * p, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = '#fff6cf';
    ctx.beginPath();
    ctx.arc(x - 1.5, y - 1.5, g.cell * 0.05, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  });

  // exit door — locked (red seam) or open (green glow)
  const open = f0 ? f0.open : false;
  {
    const [r, c] = layout.door;
    const [x, y] = g.tile(r, c);
    const dx = x + 3, dy = y + 3, dw = g.cell - 6, dh = g.cell - 6;
    ctx.save();
    if (open) {
      glow(ctx, g.cx(c), g.cy(r), g.cell * 1.1, 'rgba(52,211,153,0.8)', 0.5);
      ctx.fillStyle = '#0b3327';
      roundRect(ctx, dx, dy, dw, dh, 4);
      ctx.fill();
      ctx.strokeStyle = COL.green;
      ctx.lineWidth = 2;
      ctx.shadowColor = COL.green;
      ctx.shadowBlur = 12;
      roundRect(ctx, dx, dy, dw, dh, 4);
      ctx.stroke();
      // door swings open
      const swing = 0.5 + 0.5 * Math.sin(timeMs / 500);
      ctx.fillStyle = 'rgba(52,211,153,0.45)';
      ctx.fillRect(dx + 2, dy + 2, (dw - 4) * (0.35 + 0.2 * swing), dh - 4);
      label(ctx, g.cx(c), y - 4, 'EXIT OPEN', COL.green, 9);
    } else {
      ctx.fillStyle = '#241019';
      roundRect(ctx, dx, dy, dw, dh, 4);
      ctx.fill();
      const seam = 0.5 + 0.5 * Math.sin(timeMs / 420);
      ctx.strokeStyle = `rgba(255,77,90,${0.5 + 0.4 * seam})`;
      ctx.lineWidth = 2;
      ctx.shadowColor = COL.red;
      ctx.shadowBlur = 8 + 6 * seam;
      roundRect(ctx, dx, dy, dw, dh, 4);
      ctx.stroke();
      // padlock
      ctx.shadowBlur = 0;
      const lx = g.cx(c), ly = g.cy(r) + 2;
      ctx.fillStyle = COL.red;
      roundRect(ctx, lx - 5, ly - 3, 10, 8, 2);
      ctx.fill();
      ctx.strokeStyle = COL.red;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(lx, ly - 3, 3.6, Math.PI, 0);
      ctx.stroke();
    }
    ctx.restore();
  }

  // ghost guard
  const guard = f0 && f0.guard ? lerpPos(f0.guard, f1?.guard, t) : null;
  if (guard) {
    const gx = g.cx(guard[1]);
    const gy = g.cy(guard[0]);
    const rr = g.cell * 0.32;
    const wob = Math.sin(timeMs / 130);
    glow(ctx, gx, gy, rr * 2.6, 'rgba(255,77,90,0.65)', 0.4);
    ctx.save();
    ctx.fillStyle = COL.red;
    ctx.shadowColor = COL.red;
    ctx.shadowBlur = 10;
    ctx.beginPath();
    ctx.arc(gx, gy - rr * 0.18, rr, Math.PI, 0);
    // wavy ghost skirt
    const base = gy + rr * 0.82;
    ctx.lineTo(gx + rr, base);
    for (let i = 0; i < 3; i++) {
      const x1 = gx + rr - (i * 2 + 1) * (rr / 3);
      const x2 = gx + rr - (i * 2 + 2) * (rr / 3);
      ctx.quadraticCurveTo(x1, base + (4 + wob * 2), x2, base);
    }
    ctx.closePath();
    ctx.fill();
    ctx.shadowBlur = 0;
    // eyes look at the player
    const a = agentPos(g, f0, f1, t);
    const ang = a ? Math.atan2(a[1] - gy, a[0] - gx) : 0;
    for (const s of [-1, 1]) {
      ctx.fillStyle = '#fff';
      ctx.beginPath();
      ctx.arc(gx + s * rr * 0.38, gy - rr * 0.15, rr * 0.24, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = '#1c2445';
      ctx.beginPath();
      ctx.arc(gx + s * rr * 0.38 + Math.cos(ang) * rr * 0.1,
              gy - rr * 0.15 + Math.sin(ang) * rr * 0.1, rr * 0.12, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.restore();
  }

  // trail + pacman agent with animated mouth
  const toXY = (p) => [g.cx(p[1]), g.cy(p[0])];
  drawTrail(ctx, trailPoints(frames, cursor, 9, toXY), g.cell * 0.3, COL.gold);
  const a = agentPos(g, f0, f1, t);
  if (a) {
    const rr = g.cell * 0.34;
    // face the movement direction
    let ang = 0; // canvas-space heading: atan2(d_row, d_col)
    if (f0?.p && f1?.p && (f0.p[0] !== f1.p[0] || f0.p[1] !== f1.p[1])) {
      ang = Math.atan2(f1.p[0] - f0.p[0], f1.p[1] - f0.p[1]);
    }
    const mouth = 0.06 + 0.22 * Math.abs(Math.sin(timeMs / 110));
    glow(ctx, a[0], a[1], rr * 2.8, 'rgba(255,210,63,0.8)', 0.42);
    ctx.save();
    ctx.translate(a[0], a[1]);
    ctx.rotate(ang);
    ctx.fillStyle = COL.gold;
    ctx.shadowColor = COL.gold;
    ctx.shadowBlur = 14;
    ctx.beginPath();
    ctx.moveTo(0, 0);
    ctx.arc(0, 0, rr, mouth * Math.PI, (2 - mouth) * Math.PI);
    ctx.closePath();
    ctx.fill();
    ctx.shadowBlur = 0;
    ctx.fillStyle = '#1a1305';
    ctx.beginPath();
    ctx.arc(rr * 0.1, -rr * 0.5, rr * 0.13, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
    floatingReward(ctx, a[0], a[1], f0?.r, t);
  }

  terminalFx(ctx, W, H, f0, t, timeMs, 'ESCAPED!', { caught: 'CAUGHT!' });
}

/* ============================================================
   MUSEUM — Room 2
   ============================================================ */
export function renderMuseum(ctx, W, H, scene) {
  const { layout, frames, cursor, timeMs } = scene;
  const N = layout.size;
  const g = gridGeom(W, H, N);
  const [f0, f1, t] = frameAt(frames, cursor);
  const walls = new Set((layout.walls || []).map(key));
  const devices = new Set((layout.camera_devices || []).map(key));
  const alarm = !!(f0 && f0.alarm);

  // marble checker floor; the vault rows get a golden tint
  ctx.fillStyle = '#07070f';
  ctx.fillRect(0, 0, W, H);
  for (let r = 0; r < N; r++) {
    for (let c = 0; c < N; c++) {
      if (walls.has(`${r},${c}`)) continue;
      const [x, y] = g.tile(r, c);
      const even = (r + c) % 2 === 0;
      ctx.fillStyle = even ? '#12101f' : '#0d0c18';
      if (r <= 1) ctx.fillStyle = even ? '#191225' : '#140f1f';
      ctx.fillRect(x, y, g.cell, g.cell);
      // marble veins
      if (hash2(r, c) > 0.72) {
        ctx.save();
        ctx.globalAlpha = 0.07;
        ctx.strokeStyle = '#a78bfa';
        ctx.beginPath();
        ctx.moveTo(x + 2, y + g.cell * hash2(c, r));
        ctx.quadraticCurveTo(x + g.cell / 2, y + g.cell * 0.3,
                             x + g.cell - 2, y + g.cell * hash2(r + 1, c));
        ctx.stroke();
        ctx.restore();
      }
      ctx.strokeStyle = 'rgba(90,80,140,0.10)';
      ctx.strokeRect(x + 0.5, y + 0.5, g.cell - 1, g.cell - 1);
    }
  }

  // walls: display cases / pillars
  (layout.walls || []).forEach(([r, c]) => {
    const [x, y] = g.tile(r, c);
    const isDevice = devices.has(`${r},${c}`);
    ctx.save();
    ctx.fillStyle = '#232036';
    roundRect(ctx, x + 1.5, y + 1.5, g.cell - 3, g.cell - 3, 3);
    ctx.fill();
    ctx.fillStyle = '#312c4b';
    roundRect(ctx, x + 3.5, y + 3.5, g.cell - 7, (g.cell - 7) * 0.4, 2);
    ctx.fill();
    if (isDevice) {
      // wall-mounted camera with swiveling lens
      const cxp = g.cx(c);
      const cyp = g.cy(r);
      const sweep = Math.sin(timeMs / 900 + r + c) * 0.7;
      ctx.fillStyle = '#0a0a14';
      roundRect(ctx, cxp - 7, cyp - 4, 14, 8, 2);
      ctx.fill();
      ctx.fillStyle = alarm ? COL.red : '#f87171';
      ctx.shadowColor = COL.red;
      ctx.shadowBlur = 8;
      ctx.beginPath();
      ctx.arc(cxp + 5 * Math.cos(sweep), cyp, 2.2, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.restore();
  });

  // camera vision zones: red overlay with animated sweep bar
  (layout.cameras || []).forEach(([r, c]) => {
    const [x, y] = g.tile(r, c);
    const puls = 0.5 + 0.5 * Math.sin(timeMs / (alarm ? 220 : 600) + c * 0.9);
    ctx.save();
    ctx.fillStyle = `rgba(255,77,90,${0.09 + 0.07 * puls})`;
    ctx.fillRect(x, y, g.cell, g.cell);
    ctx.beginPath();
    ctx.rect(x, y, g.cell, g.cell);
    ctx.clip();
    const sx = ((timeMs / 9) % (g.cell * 2.4)) - g.cell;
    ctx.fillStyle = 'rgba(255,77,90,0.18)';
    ctx.fillRect(x + sx, y, g.cell * 0.35, g.cell);
    ctx.restore();
    label(ctx, g.cx(c), g.cy(r) + 3, '◉', `rgba(255,120,130,${0.35 + 0.3 * puls})`, g.cell * 0.3);
  });

  // laser traps
  (layout.traps || []).forEach(([r, c]) => {
    const [x, y] = g.tile(r, c);
    const on = Math.sin(timeMs / 300 + r * 2.1) > -0.6;
    ctx.save();
    ctx.fillStyle = 'rgba(0,0,0,0.4)';
    ctx.fillRect(x + 2, y + 2, g.cell - 4, g.cell - 4);
    if (on) {
      ctx.strokeStyle = COL.red;
      ctx.lineWidth = 1.4;
      ctx.shadowColor = COL.red;
      ctx.shadowBlur = 6;
      for (let i = 1; i <= 3; i++) {
        const yy = y + (g.cell / 4) * i;
        ctx.beginPath();
        ctx.moveTo(x + 3, yy);
        ctx.lineTo(x + g.cell - 3, yy);
        ctx.stroke();
      }
    }
    // emitter posts
    ctx.shadowBlur = 0;
    ctx.fillStyle = '#3a3552';
    ctx.fillRect(x + 1, y + 2, 3, g.cell - 4);
    ctx.fillRect(x + g.cell - 4, y + 2, 3, g.cell - 4);
    ctx.restore();
  });

  // slippery marble
  (layout.slippery || []).forEach(([r, c]) => {
    const [x, y] = g.tile(r, c);
    ctx.fillStyle = 'rgba(125,211,252,0.10)';
    ctx.fillRect(x + 1, y + 1, g.cell - 2, g.cell - 2);
    label(ctx, g.cx(c), g.cy(r) + 3, '〜', 'rgba(165,225,255,0.5)', g.cell * 0.3);
  });

  // guard patrol routes
  (layout.guard_routes || []).forEach((route) => {
    if (route.length < 2) return;
    ctx.save();
    ctx.strokeStyle = 'rgba(251,191,36,0.28)';
    ctx.setLineDash([3, 6]);
    ctx.lineWidth = 1.4;
    ctx.beginPath();
    route.forEach(([r, c], i) => {
      if (i === 0) ctx.moveTo(g.cx(c), g.cy(r)); else ctx.lineTo(g.cx(c), g.cy(r));
    });
    ctx.stroke();
    ctx.restore();
  });

  // exit
  {
    const [r, c] = layout.exit;
    const x = g.cx(c);
    const y = g.cy(r);
    glow(ctx, x, y, g.cell * 0.9, 'rgba(52,211,153,0.6)', 0.35);
    ctx.save();
    ctx.strokeStyle = COL.green;
    ctx.lineWidth = 2;
    ctx.shadowColor = COL.green;
    ctx.shadowBlur = 10;
    roundRect(ctx, x - g.cell * 0.32, y - g.cell * 0.38, g.cell * 0.64, g.cell * 0.76, 3);
    ctx.stroke();
    ctx.restore();
    label(ctx, x, y - g.cell * 0.48, 'EXIT', COL.green, 8.5);
  }

  // diamond (until stolen)
  const hasD = f0 ? !!f0.d : false;
  if (!hasD) {
    const [r, c] = layout.diamond;
    const x = g.cx(c);
    const y = g.cy(r) + Math.sin(timeMs / 420) * 2.5;
    const s = g.cell * 0.26;
    glow(ctx, x, y, g.cell * 0.9, 'rgba(56,189,248,0.9)', 0.5);
    ctx.save();
    ctx.fillStyle = COL.cyan;
    ctx.shadowColor = COL.cyan;
    ctx.shadowBlur = 14;
    ctx.beginPath();
    ctx.moveTo(x, y - s);
    ctx.lineTo(x + s * 0.85, y - s * 0.2);
    ctx.lineTo(x, y + s);
    ctx.lineTo(x - s * 0.85, y - s * 0.2);
    ctx.closePath();
    ctx.fill();
    ctx.fillStyle = 'rgba(255,255,255,0.85)';
    ctx.beginPath();
    ctx.moveTo(x, y - s * 0.75);
    ctx.lineTo(x + s * 0.3, y - s * 0.22);
    ctx.lineTo(x - s * 0.3, y - s * 0.22);
    ctx.closePath();
    ctx.fill();
    // sparkle
    const sp = (timeMs / 500 + r) % 1;
    if (sp < 0.35) {
      ctx.strokeStyle = `rgba(255,255,255,${0.8 - sp * 2})`;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x + s * 0.6, y - s * 0.7 - 4);
      ctx.lineTo(x + s * 0.6, y - s * 0.7 + 4);
      ctx.moveTo(x + s * 0.6 - 4, y - s * 0.7);
      ctx.lineTo(x + s * 0.6 + 4, y - s * 0.7);
      ctx.stroke();
    }
    ctx.restore();
  }

  // guards
  (f0?.guards || []).forEach((gp, i) => {
    const gq = f1?.guards?.[i];
    const p = lerpPos(gp, gq, t);
    const gx = g.cx(p[1]);
    const gy = g.cy(p[0]);
    const rr = g.cell * 0.3;
    glow(ctx, gx, gy, rr * 3.4, 'rgba(251,191,36,0.5)', alarm ? 0.5 : 0.3);
    ctx.save();
    // body
    ctx.fillStyle = alarm ? '#7f1d1d' : '#1f2937';
    ctx.shadowColor = alarm ? COL.red : '#000';
    ctx.shadowBlur = 8;
    ctx.beginPath();
    ctx.arc(gx, gy, rr, 0, Math.PI * 2);
    ctx.fill();
    ctx.shadowBlur = 0;
    // cap
    ctx.fillStyle = alarm ? '#991b1b' : '#111827';
    ctx.beginPath();
    ctx.arc(gx, gy - rr * 0.35, rr * 0.72, Math.PI, 0);
    ctx.fill();
    ctx.fillRect(gx - rr * 0.72, gy - rr * 0.38, rr * 1.44, rr * 0.18);
    // flashlight dot
    ctx.fillStyle = COL.amber;
    ctx.shadowColor = COL.amber;
    ctx.shadowBlur = 8;
    ctx.beginPath();
    ctx.arc(gx + rr * 0.75, gy + rr * 0.1, 2.4, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  });

  // thief agent
  const toXY = (p) => [g.cx(p[1]), g.cy(p[0])];
  drawTrail(ctx, trailPoints(frames, cursor, 9, toXY), g.cell * 0.28, COL.violet);
  const a = agentPos(g, f0, f1, t);
  if (a) {
    const rr = g.cell * 0.3;
    glow(ctx, a[0], a[1], rr * 3, 'rgba(167,139,250,0.8)', 0.4);
    ctx.save();
    // hooded body
    ctx.fillStyle = '#312e59';
    ctx.shadowColor = COL.violet;
    ctx.shadowBlur = 12;
    ctx.beginPath();
    ctx.arc(a[0], a[1], rr, 0, Math.PI * 2);
    ctx.fill();
    ctx.shadowBlur = 0;
    // visor
    ctx.fillStyle = COL.cyan;
    roundRect(ctx, a[0] - rr * 0.62, a[1] - rr * 0.28, rr * 1.24, rr * 0.36, 2);
    ctx.fill();
    ctx.restore();
    // stolen diamond floats above
    if (hasD) {
      const dy = a[1] - rr - 8 + Math.sin(timeMs / 300) * 2;
      ctx.save();
      ctx.fillStyle = COL.cyan;
      ctx.shadowColor = COL.cyan;
      ctx.shadowBlur = 10;
      ctx.beginPath();
      ctx.moveTo(a[0], dy - 5);
      ctx.lineTo(a[0] + 4.5, dy - 1);
      ctx.lineTo(a[0], dy + 5);
      ctx.lineTo(a[0] - 4.5, dy - 1);
      ctx.closePath();
      ctx.fill();
      ctx.restore();
    }
    floatingReward(ctx, a[0], a[1], f0?.r, t);
  }

  // alarm overlay
  if (alarm) {
    const p = 0.5 + 0.5 * Math.sin(timeMs / 180);
    ctx.save();
    ctx.strokeStyle = `rgba(255,77,90,${0.35 + 0.4 * p})`;
    ctx.lineWidth = 6;
    ctx.strokeRect(3, 3, W - 6, H - 6);
    ctx.restore();
    label(ctx, W / 2, 18, '⚠ ALARM — GUARDS ALERTED ⚠', `rgba(255,110,120,${0.6 + 0.4 * p})`, 12, 'Chakra Petch');
  }

  terminalFx(ctx, W, H, f0, t, timeMs, 'HEIST COMPLETE!', { caught: 'CAUGHT!' });
}

/* ============================================================
   RACING — Room 3
   ============================================================ */
export function renderRacing(ctx, W, H, scene) {
  const { layout, frames, cursor, timeMs } = scene;
  const N = layout.size;
  const g = gridGeom(W, H, N);
  const [f0, f1, t] = frameAt(frames, cursor);
  const walls = new Set((layout.walls || []).map(key));

  // asphalt
  ctx.fillStyle = '#101116';
  ctx.fillRect(0, 0, W, H);
  for (let r = 0; r < N; r++) {
    for (let c = 0; c < N; c++) {
      if (walls.has(`${r},${c}`)) continue;
      const [x, y] = g.tile(r, c);
      const n = hash2(r, c);
      ctx.fillStyle = n > 0.5 ? '#17181e' : '#141519';
      ctx.fillRect(x, y, g.cell, g.cell);
      // asphalt speckle + tyre rubber
      ctx.fillStyle = 'rgba(255,255,255,0.028)';
      ctx.fillRect(x + n * g.cell * 0.8, y + hash2(c, r) * g.cell * 0.8, 1.6, 1.6);
    }
  }

  // white lane lines on long straights
  ctx.save();
  ctx.strokeStyle = 'rgba(255,255,255,0.10)';
  ctx.setLineDash([g.cell * 0.28, g.cell * 0.28]);
  ctx.lineWidth = 1.6;
  for (let r = 0; r < N; r++) {
    let run = [];
    for (let c = 0; c <= N; c++) {
      if (c < N && !walls.has(`${r},${c}`)) run.push(c);
      else {
        if (run.length >= 4) {
          ctx.beginPath();
          ctx.moveTo(g.cx(run[0]) - g.cell * 0.3, g.cy(r));
          ctx.lineTo(g.cx(run[run.length - 1]) + g.cell * 0.3, g.cy(r));
          ctx.stroke();
        }
        run = [];
      }
    }
  }
  ctx.restore();

  // grass infield blocks with red/white F1 kerbs facing the track
  const isTrack = (rr, cc) => rr >= 0 && rr < N && cc >= 0 && cc < N
    && !walls.has(`${rr},${cc}`);
  const kerb = (kx, ky, kw, kh, horizontal) => {
    const seg = Math.max(5, g.cell / 8);
    const len = horizontal ? kw : kh;
    for (let i = 0; i * seg < len; i++) {
      ctx.fillStyle = i % 2 === 0 ? '#d21b2d' : '#f3f4f6';
      const s = Math.min(seg, len - i * seg);
      if (horizontal) ctx.fillRect(kx + i * seg, ky, s, kh);
      else ctx.fillRect(kx, ky + i * seg, kw, s);
    }
  };
  (layout.walls || []).forEach(([r, c]) => {
    const [x, y] = g.tile(r, c);
    ctx.save();
    // mowed grass
    ctx.fillStyle = (r + c) % 2 === 0 ? '#0d3d20' : '#0b361c';
    ctx.fillRect(x, y, g.cell, g.cell);
    ctx.globalAlpha = 0.35;
    ctx.fillStyle = '#0f4525';
    ctx.fillRect(x, y + g.cell * 0.5 * hash2(r, c), g.cell, g.cell * 0.22);
    ctx.globalAlpha = 1;
    // kerbs on every edge that touches tarmac
    const kt = Math.max(3, g.cell * 0.08);
    if (isTrack(r - 1, c)) kerb(x, y, g.cell, kt, true);
    if (isTrack(r + 1, c)) kerb(x, y + g.cell - kt, g.cell, kt, true);
    if (isTrack(r, c - 1)) kerb(x, y, kt, g.cell, false);
    if (isTrack(r, c + 1)) kerb(x + g.cell - kt, y, kt, g.cell, false);
    ctx.restore();
  });

  // oil slicks
  (layout.oil || []).forEach(([r, c]) => {
    const x = g.cx(c);
    const y = g.cy(r);
    ctx.save();
    ctx.fillStyle = 'rgba(8,8,14,0.92)';
    ctx.beginPath();
    ctx.ellipse(x, y, g.cell * 0.38, g.cell * 0.27, 0.5, 0, Math.PI * 2);
    ctx.fill();
    const sh = 0.4 + 0.3 * Math.sin(timeMs / 700 + c);
    ctx.strokeStyle = `rgba(120,190,255,${sh * 0.4})`;
    ctx.lineWidth = 1.4;
    ctx.beginPath();
    ctx.ellipse(x - 3, y - 2, g.cell * 0.16, g.cell * 0.08, 0.5, 0, Math.PI * 2);
    ctx.stroke();
    ctx.restore();
  });

  // gravel traps
  (layout.mud || []).forEach(([r, c]) => {
    const [x, y] = g.tile(r, c);
    ctx.save();
    ctx.fillStyle = 'rgba(190,160,110,0.20)';
    ctx.fillRect(x + 1.5, y + 1.5, g.cell - 3, g.cell - 3);
    ctx.fillStyle = 'rgba(214,186,130,0.55)';
    for (let i = 0; i < 16; i++) {
      const n1 = hash2(r * 7 + i, c * 5 + i);
      const n2 = hash2(c * 9 + i, r * 3 + i);
      ctx.beginPath();
      ctx.arc(x + 3 + n1 * (g.cell - 6), y + 3 + n2 * (g.cell - 6),
              0.9 + n1 * 1.1, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.restore();
  });

  // crash barriers — stacked red/white TecPro blocks
  (layout.crash || []).forEach(([r, c]) => {
    const [x, y] = g.tile(r, c);
    ctx.save();
    for (let i = 0; i < 2; i++) {
      const by = y + g.cell * 0.22 + i * g.cell * 0.30;
      ctx.fillStyle = i % 2 === 0 ? '#d21b2d' : '#f3f4f6';
      roundRect(ctx, x + 2, by, g.cell - 4, g.cell * 0.26, 3);
      ctx.fill();
      ctx.strokeStyle = 'rgba(0,0,0,0.35)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      roundRect(ctx, x + 2, by, g.cell - 4, g.cell * 0.26, 3);
      ctx.stroke();
    }
    ctx.restore();
    label(ctx, g.cx(c), y + g.cell * 0.17, '⚠', COL.amber, g.cell * 0.2);
  });

  // checkpoint timing gates (blue = ahead, green ✓ = crossed)
  const ncp = f0 ? (f0.ncp || 0) : 0;
  (layout.checkpoints || []).forEach((gate, gi) => {
    const passed = ncp > gi;
    const active = ncp === gi;
    gate.forEach(([r, c]) => {
      const [x, y] = g.tile(r, c);
      const base = passed ? COL.green : COL.cyan;
      const puls = active ? 0.5 + 0.5 * Math.sin(timeMs / 210 + gi * 2) : 0.25;
      ctx.save();
      // light band across the cell
      ctx.fillStyle = passed ? 'rgba(52,211,153,0.10)'
        : `rgba(56,189,248,${0.10 + 0.12 * puls})`;
      ctx.fillRect(x + g.cell * 0.3, y + 1.5, g.cell * 0.4, g.cell - 3);
      // gate pylons top + bottom
      ctx.fillStyle = base;
      ctx.shadowColor = base;
      ctx.shadowBlur = active ? 10 : 4;
      ctx.fillRect(x + g.cell * 0.26, y + 1.5, g.cell * 0.48, 3.2);
      ctx.fillRect(x + g.cell * 0.26, y + g.cell - 4.7, g.cell * 0.48, 3.2);
      ctx.restore();
      label(ctx, g.cx(c), g.cy(r) + 3.5, passed ? '✓' : `${gi + 1}`,
        base, g.cell * 0.3);
    });
  });

  // starting grid slot
  {
    const [sr, sc] = layout.start;
    const [x, y] = g.tile(sr, sc);
    ctx.save();
    ctx.strokeStyle = 'rgba(255,255,255,0.45)';
    ctx.lineWidth = 1.6;
    for (const oy of [0.16, 0.56]) {
      ctx.beginPath();
      ctx.moveTo(x + g.cell * 0.72, y + g.cell * oy);
      ctx.lineTo(x + g.cell * 0.28, y + g.cell * oy);
      ctx.lineTo(x + g.cell * 0.28, y + g.cell * (oy + 0.28));
      ctx.stroke();
    }
    ctx.restore();
    label(ctx, g.cx(sc), y + g.cell * 0.12, 'GRID', 'rgba(255,255,255,0.5)', 7);
  }

  // finish line — checkered, locked until enough boosters
  const open = f0 ? f0.open : false;
  {
    const [r, c] = layout.finish;
    const [x, y] = g.tile(r, c);
    const sq = (g.cell - 4) / 4;
    for (let i = 0; i < 4; i++) {
      for (let j = 0; j < 4; j++) {
        ctx.fillStyle = (i + j) % 2 === 0 ? '#e5e7eb' : '#111318';
        ctx.fillRect(x + 2 + i * sq, y + 2 + j * sq, sq, sq);
      }
    }
    if (open) {
      glow(ctx, g.cx(c), g.cy(r), g.cell * 1.15, 'rgba(52,211,153,0.85)', 0.45);
      label(ctx, g.cx(c), y - 4, 'FINISH OPEN', COL.green, 8.5);
    } else {
      ctx.fillStyle = 'rgba(10,10,16,0.55)';
      ctx.fillRect(x + 2, y + 2, g.cell - 4, g.cell - 4);
      const p = 0.5 + 0.5 * Math.sin(timeMs / 420);
      ctx.save();
      ctx.strokeStyle = `rgba(255,77,90,${0.5 + 0.4 * p})`;
      ctx.lineWidth = 2;
      ctx.strokeRect(x + 2, y + 2, g.cell - 4, g.cell - 4);
      ctx.restore();
      label(ctx, g.cx(c), g.cy(r) + 3, '🔒', COL.red, g.cell * 0.3);
    }
  }

  // cars: player (Q-Learning, red) + optional SARSA rival (violet, in races).
  // In races each car keeps its own visual lane inside the cell, so they
  // stay visible side by side on the grid and at the flag.
  const racing = !!f0?.rv;
  const laneOff = racing ? g.cell * 0.17 : 0;
  const toXY = (p) => [g.cx(p[1]), g.cy(p[0]) + laneOff];
  const toXYrv = (p) => [g.cx(p[1]), g.cy(p[0]) - laneOff];

  const drawCar = (p0, p1, dy, body, ring, name, skid) => {
    const a = p0 ? lerpPos(p0, p1, t) : null;
    if (!a) return null;
    const x = g.cx(a[1]);
    const y = g.cy(a[0]) + dy;
    let ang = 0; // canvas-space heading; sprite is drawn nose-up, +90° applied below
    if (p0 && p1 && (p0[0] !== p1[0] || p0[1] !== p1[1])) {
      ang = Math.atan2(p1[0] - p0[0], p1[1] - p0[1]);
    }
    const carW = g.cell * 0.36;
    const carL = g.cell * 0.6;
    glow(ctx, x, y, g.cell * 0.9, ring, 0.35);
    ctx.save();
    ctx.translate(x, y);
    ctx.rotate(ang + Math.PI / 2);
    // skid marks when slipping
    if (skid) {
      ctx.strokeStyle = 'rgba(30,30,36,0.8)';
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(-carW * 0.32, carL * 0.2);
      ctx.quadraticCurveTo(-carW * 0.8, carL * 0.8, -carW * 1.3, carL * 1.3);
      ctx.moveTo(carW * 0.32, carL * 0.2);
      ctx.quadraticCurveTo(carW * 0.9, carL * 0.7, carW * 1.4, carL * 1.2);
      ctx.stroke();
    }
    // wheels
    ctx.fillStyle = '#0c0d12';
    for (const sx of [-1, 1]) {
      roundRect(ctx, sx * carW * 0.52 - 2.2, -carL * 0.34, 4.4, carL * 0.24, 2);
      ctx.fill();
      roundRect(ctx, sx * carW * 0.52 - 2.2, carL * 0.12, 4.4, carL * 0.24, 2);
      ctx.fill();
    }
    // body
    ctx.fillStyle = body;
    ctx.shadowColor = body;
    ctx.shadowBlur = 12;
    roundRect(ctx, -carW / 2, -carL / 2, carW, carL, carW * 0.32);
    ctx.fill();
    ctx.shadowBlur = 0;
    // cockpit
    ctx.fillStyle = '#101423';
    roundRect(ctx, -carW * 0.3, -carL * 0.16, carW * 0.6, carL * 0.34, 3);
    ctx.fill();
    // headlights
    ctx.fillStyle = '#fff7cc';
    ctx.fillRect(-carW * 0.32, -carL / 2 + 0.5, carW * 0.2, 2.4);
    ctx.fillRect(carW * 0.12, -carL / 2 + 0.5, carW * 0.2, 2.4);
    ctx.restore();
    if (name) label(ctx, x, y - g.cell * 0.52, name, ring, 8);
    return [x, y];
  };

  if (racing) {
    const rvPts = [];
    const end = Math.floor(cursor);
    for (let i = Math.max(0, end - 8); i < end; i++) {
      const fr = frames[i];
      if (fr && fr.rv) rvPts.push(toXYrv(fr.rv));
    }
    drawTrail(ctx, rvPts, g.cell * 0.18, 'rgba(167,139,250,0.6)');
    drawCar(f0.rv, f1?.rv, -laneOff, '#7c5cd6', 'rgba(167,139,250,0.85)',
      'SARSA', false);
  }

  drawTrail(ctx, trailPoints(frames, cursor, 8, toXY), g.cell * 0.22, 'rgba(255,120,90,0.8)');
  const carXY = drawCar(f0?.p, f1?.p, laneOff, COL.red, 'rgba(255,77,90,0.85)',
    racing ? 'Q-LEARNING' : null, (f0?.ev || []).includes('slip'));
  if (carXY) floatingReward(ctx, carXY[0], carXY[1], f0?.r, t);

  // outcome — race verdicts take precedence over the plain finish banner
  const ev = f0?.ev || [];
  if (ev.includes('lost race')) {
    fullFlash(ctx, W, H, COL.amber, Math.max(0, 0.14 - t * 0.12));
    outcomeText(ctx, W, H, 'RIVAL WINS!', COL.amber, timeMs);
  } else if (ev.includes('won race')) {
    fullFlash(ctx, W, H, COL.green, Math.max(0, 0.18 - t * 0.16));
    outcomeText(ctx, W, H, 'RACE WON!', COL.green, timeMs);
  } else {
    terminalFx(ctx, W, H, f0, t, timeMs, 'RACE WON!', { crash: 'CRASHED!' });
  }
}
