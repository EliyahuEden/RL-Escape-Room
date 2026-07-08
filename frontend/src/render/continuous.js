/* ============================================================
   Canvas renderers for the two continuous rooms:

     football  — striped pitch, goal net, defenders, patrolling
                 keeper, physical ball flights (with height in
                 free-kick mode)
     crossroad — grass shoulders, multi-lane road, wrapping
                 traffic, chicken agent with live sensor ring
   ============================================================ */
import {
  COL, drawTrail, floatingReward, frameAt, fullFlash, glow, hash2, label,
  lerp, lerpPos, outcomeOf, outcomeText, roundRect, trailPoints,
} from './helpers.js';

/* ============================================================
   FOOTBALL — Room 4  (env coords: x → right, y → down-mapped)
   ============================================================ */
export function renderFootball(ctx, W, H, scene) {
  const { layout, frames, cursor, timeMs } = scene;
  const pad = 26;
  const SW = W - pad * 2;
  const SH = H - pad * 2;
  const X = (x) => pad + (x / layout.W) * SW;
  const Y = (y) => pad + (1 - y / layout.H) * SH; // env y-up → canvas y-down
  const [f0, f1, t] = frameAt(frames, cursor);
  const freekick = layout.mode === 'freekick';

  // grass with mowing stripes
  ctx.fillStyle = '#04140c';
  ctx.fillRect(0, 0, W, H);
  for (let i = 0; i < 10; i++) {
    ctx.fillStyle = i % 2 === 0 ? '#0a3320' : '#0b3a24';
    ctx.fillRect(pad + (i * SW) / 10, pad, SW / 10, SH);
  }
  // grass noise
  ctx.save();
  ctx.globalAlpha = 0.05;
  for (let i = 0; i < 140; i++) {
    const n1 = hash2(i, 7);
    const n2 = hash2(i, 13);
    ctx.fillStyle = n1 > 0.5 ? '#8dfab0' : '#04240f';
    ctx.fillRect(pad + n1 * SW, pad + n2 * SH, 2, 2);
  }
  ctx.restore();

  // pitch lines
  ctx.save();
  ctx.strokeStyle = 'rgba(255,255,255,0.55)';
  ctx.lineWidth = 2;
  ctx.strokeRect(pad, pad, SW, SH);
  // halfway line + centre circle
  ctx.beginPath();
  ctx.moveTo(X(layout.W / 2), pad);
  ctx.lineTo(X(layout.W / 2), pad + SH);
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(X(layout.W / 2), Y(layout.H / 2), SW * 0.09, 0, Math.PI * 2);
  ctx.stroke();
  // penalty box (right)
  const boxTop = Y(layout.goal_hi + 1.4);
  const boxBot = Y(layout.goal_lo - 1.4);
  ctx.strokeRect(X(layout.W - 2.2), boxTop, X(layout.W) - X(layout.W - 2.2), boxBot - boxTop);
  ctx.restore();

  // shooting zone tint (match mode)
  if (!freekick && layout.shoot_x > 0) {
    const inShoot = f0?.shoot;
    ctx.save();
    ctx.fillStyle = inShoot ? 'rgba(52,211,153,0.10)' : 'rgba(52,211,153,0.05)';
    ctx.fillRect(X(layout.shoot_x), pad, X(layout.W) - X(layout.shoot_x), SH);
    ctx.setLineDash([6, 6]);
    ctx.strokeStyle = 'rgba(52,211,153,0.5)';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(X(layout.shoot_x), pad);
    ctx.lineTo(X(layout.shoot_x), pad + SH);
    ctx.stroke();
    ctx.restore();
    label(ctx, X(layout.shoot_x) + 4, pad + 14, '⚽ SHOOTING ZONE →', 'rgba(52,211,153,0.7)', 9);
  }

  // goal mouth + net (right edge between goal_lo..goal_hi)
  {
    const gy0 = Y(layout.goal_hi);
    const gy1 = Y(layout.goal_lo);
    const gx = X(layout.W);
    glow(ctx, gx, (gy0 + gy1) / 2, 60, 'rgba(255,255,255,0.5)', 0.18);
    ctx.save();
    // posts
    ctx.fillStyle = '#f8fafc';
    ctx.fillRect(gx - 2.5, gy0 - 3, 9, 3);
    ctx.fillRect(gx - 2.5, gy1, 9, 3);
    ctx.fillRect(gx - 2.5, gy0, 3, gy1 - gy0);
    // net
    ctx.strokeStyle = 'rgba(255,255,255,0.28)';
    ctx.lineWidth = 1;
    const netW = 14;
    for (let i = 1; i <= 3; i++) {
      ctx.beginPath();
      ctx.moveTo(gx + (i * netW) / 3, gy0 - 2);
      ctx.lineTo(gx + (i * netW) / 3, gy1 + 2);
      ctx.stroke();
    }
    for (let j = 0; j <= 6; j++) {
      const yy = gy0 + ((gy1 - gy0) * j) / 6;
      ctx.beginPath();
      ctx.moveTo(gx, yy);
      ctx.lineTo(gx + netW, yy);
      ctx.stroke();
    }
    ctx.restore();
  }

  // free-kick wall line — drawn THROUGH the actual wall players, so it follows
  // the diagonal set-up when the kick is taken from the side of the pitch
  if (freekick && f0?.defs && f0.defs.length >= 2) {
    const pts = f0.defs.map((d) => [X(d[0]), Y(d[1])]);
    ctx.save();
    ctx.setLineDash([4, 8]);
    ctx.strokeStyle = 'rgba(255,255,255,0.18)';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(pts[0][0], pts[0][1]);
    for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
    ctx.stroke();
    ctx.restore();
  }

  const drawPlayer = (x, y, jersey, ring, num) => {
    const r = 11;
    glow(ctx, x, y, r * 2.6, ring, 0.35);
    ctx.save();
    // shadow
    ctx.fillStyle = 'rgba(0,0,0,0.4)';
    ctx.beginPath();
    ctx.ellipse(x, y + r * 0.85, r * 0.85, r * 0.32, 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = jersey;
    ctx.shadowColor = ring;
    ctx.shadowBlur = 9;
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fill();
    ctx.shadowBlur = 0;
    ctx.strokeStyle = 'rgba(255,255,255,0.5)';
    ctx.lineWidth = 1.4;
    ctx.stroke();
    if (num !== undefined) {
      ctx.fillStyle = '#fff';
      ctx.font = 'bold 10px "Chakra Petch", sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(num, x, y + 3.5);
    }
    ctx.restore();
  };

  // defenders / wall
  (f0?.defs || []).forEach((d, i) => {
    const q = f1?.defs?.[i];
    const p = lerpPos(d, q, t, 4);
    drawPlayer(X(p[0]), Y(p[1]), '#b91c1c', 'rgba(255,77,90,0.7)', i + 2);
  });

  // keeper
  if (f0?.keeper) {
    const kp = lerpPos(f0.keeper, f1?.keeper, t, 4);
    drawPlayer(X(kp[0]), Y(kp[1]), '#ca8a04', 'rgba(251,191,36,0.8)', 1);
  }

  // striker + trail
  const toXY = (p) => [X(p[0]), Y(p[1])];
  drawTrail(ctx, trailPoints(frames, cursor, 10, toXY), 8, 'rgba(56,189,248,0.9)');
  const pl = f0?.p ? lerpPos(f0.p, f1?.p, t, 4) : null;
  if (pl) drawPlayer(X(pl[0]), Y(pl[1]), '#0369a1', 'rgba(56,189,248,0.9)', 9);

  // ball: at the player's feet, or flying (frame.fl)
  let ballXY = null;
  let ballScale = 1;
  if (f0?.fl && f0.ball) {
    const b = lerpPos(f0.ball, f1?.fl ? f1.ball : f0.ball, t, 6);
    ballXY = [X(b[0]), Y(b[1])];
    const z = lerp(f0.z || 0, (f1?.fl ? f1.z : f0.z) || 0, t);
    ballScale = 1 + z * 0.22;
    // height shadow
    ctx.save();
    ctx.fillStyle = 'rgba(0,0,0,0.4)';
    ctx.beginPath();
    ctx.ellipse(ballXY[0], ballXY[1] + 4 + z * 6, 5 + z, 2.4, 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
    ballXY[1] -= z * 6; // lift the drawn ball with height
  } else if (pl) {
    ballXY = [X(pl[0]) + 9, Y(pl[1]) + 7];
  }
  if (ballXY) {
    const br = 5 * ballScale;
    glow(ctx, ballXY[0], ballXY[1], br * 3.2, 'rgba(255,255,255,0.8)', 0.3);
    ctx.save();
    ctx.fillStyle = '#f8fafc';
    ctx.shadowColor = '#fff';
    ctx.shadowBlur = 7;
    ctx.beginPath();
    ctx.arc(ballXY[0], ballXY[1], br, 0, Math.PI * 2);
    ctx.fill();
    ctx.shadowBlur = 0;
    // pentagon patch + seam, rotating during flight
    const rot = f0?.fl ? timeMs / 90 : 0;
    ctx.fillStyle = '#111827';
    ctx.beginPath();
    for (let i = 0; i < 5; i++) {
      const aa = rot + (i * 2 * Math.PI) / 5;
      const px = ballXY[0] + Math.cos(aa) * br * 0.42;
      const py = ballXY[1] + Math.sin(aa) * br * 0.42;
      if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    }
    ctx.closePath();
    ctx.fill();
    ctx.restore();
  }

  if (pl) floatingReward(ctx, X(pl[0]), Y(pl[1]), f0?.r, t);

  // outcome
  const oc = outcomeOf(f0);
  if (oc) {
    fullFlash(ctx, W, H, oc === 'good' ? COL.green : COL.red, Math.max(0, 0.16 - t * 0.14));
    const ev = (f0.ev || []).join(' ').toLowerCase();
    let text = oc === 'good' ? 'GOAL!' : 'NO GOAL';
    if (ev.includes('save')) text = 'SAVED!';
    else if (ev.includes('block')) text = 'BLOCKED!';
    else if (ev.includes('tackle')) text = 'TACKLED!';
    else if (ev.includes('miss')) text = 'MISSED!';
    else if (ev.includes('time')) text = 'TIME UP!';
    outcomeText(ctx, W, H, text, oc === 'good' ? COL.green : COL.red, timeMs);
  }
}

/* ============================================================
   CROSSROAD — Room 5  (x → right = crossing direction,
   y → vertical = traffic direction)
   ============================================================ */
export function renderCrossroad(ctx, W, H, scene) {
  const { layout, frames, cursor, timeMs } = scene;
  const pad = 26;
  const SW = W - pad * 2;
  const SH = H - pad * 2;
  const X = (x) => pad + (x / layout.W) * SW;
  const Y = (y) => pad + (y / layout.H) * SH;
  const SX = (m) => (m / layout.W) * SW; // metres → pixels
  const [f0, f1, t] = frameAt(frames, cursor);

  // grass background
  ctx.fillStyle = '#07160c';
  ctx.fillRect(0, 0, W, H);
  ctx.save();
  ctx.globalAlpha = 0.06;
  for (let i = 0; i < 120; i++) {
    const n1 = hash2(i, 3);
    const n2 = hash2(i, 11);
    ctx.fillStyle = n1 > 0.5 ? '#65d98a' : '#03230e';
    ctx.fillRect(pad + n1 * SW, pad + n2 * SH, 2, 2);
  }
  ctx.restore();

  // sidewalks
  const roadL = X(layout.road_x_min);
  const roadR = X(layout.road_x_max);
  ctx.fillStyle = '#252a36';
  ctx.fillRect(X(0), pad, roadL - X(0), SH);
  ctx.fillRect(roadR, pad, X(layout.W) - roadR, SH);
  // sidewalk tiles
  ctx.strokeStyle = 'rgba(255,255,255,0.05)';
  ctx.lineWidth = 1;
  for (let i = 0; i < 12; i++) {
    const yy = pad + (i * SH) / 12;
    ctx.beginPath();
    ctx.moveTo(X(0), yy);
    ctx.lineTo(roadL, yy);
    ctx.moveTo(roadR, yy);
    ctx.lineTo(X(layout.W), yy);
    ctx.stroke();
  }

  // road
  ctx.fillStyle = '#101319';
  ctx.fillRect(roadL, pad, roadR - roadL, SH);
  // road edge lines
  ctx.fillStyle = 'rgba(255,255,255,0.5)';
  ctx.fillRect(roadL, pad, 2.5, SH);
  ctx.fillRect(roadR - 2.5, pad, 2.5, SH);

  // animated lane separators between lane centres
  const lanes = layout.lane_xs || [];
  ctx.save();
  ctx.strokeStyle = 'rgba(255,210,80,0.35)';
  ctx.lineWidth = 2;
  const dashLen = 14;
  for (let i = 1; i < lanes.length; i++) {
    const lx = X((lanes[i - 1] + lanes[i]) / 2);
    // dashes drift, alternating with lane direction
    const dir = i % 2 === 0 ? 1 : -1;
    const off = ((timeMs / 24) * dir) % (dashLen * 2);
    ctx.beginPath();
    for (let yy = pad - dashLen * 2 + off; yy < pad + SH + dashLen; yy += dashLen * 2) {
      ctx.moveTo(lx, Math.max(pad, yy));
      ctx.lineTo(lx, Math.min(pad + SH, yy + dashLen));
    }
    ctx.stroke();
  }
  ctx.restore();

  // goal strip (right edge) — glowing green
  {
    const gp = 0.5 + 0.5 * Math.sin(timeMs / 500);
    const gx = X(layout.goal_x);
    ctx.save();
    ctx.fillStyle = `rgba(52,211,153,${0.10 + 0.08 * gp})`;
    ctx.fillRect(gx, pad, X(layout.W) - gx, SH);
    ctx.strokeStyle = `rgba(52,211,153,${0.5 + 0.3 * gp})`;
    ctx.setLineDash([8, 6]);
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(gx, pad);
    ctx.lineTo(gx, pad + SH);
    ctx.stroke();
    ctx.restore();
    ctx.save();
    ctx.translate(X(layout.W) - 8, H / 2);
    ctx.rotate(Math.PI / 2);
    label(ctx, 0, 0, 'SAFE ZONE', COL.green, 10, 'Chakra Petch');
    ctx.restore();
  }

  // cars (positions from frames, styling from layout.cars_meta)
  const meta = layout.cars_meta || [];
  const cars0 = f0?.cars || [];
  const cars1 = f1?.cars || cars0;
  cars0.forEach((c0, i) => {
    const m = meta[i] || { w: 0.55, h: 1.05, color: '#ef4444', dir: 1 };
    // wrap-around: don't interpolate across the wrap
    const c1 = cars1[i] || c0;
    const p = Math.abs((c1[1] ?? 0) - c0[1]) > 3 ? c0 : [lerp(c0[0], c1[0], t), lerp(c0[1], c1[1], t)];
    const cw = SX(m.w);
    const chh = SX(m.h);
    const x = X(p[0]) - cw / 2;
    const y = Y(p[1]) - chh / 2;
    if (y + chh < pad - 30 || y > pad + SH + 30) return;
    ctx.save();
    ctx.beginPath();
    ctx.rect(pad, pad, SW, SH);
    ctx.clip();
    // body
    ctx.fillStyle = m.color;
    ctx.shadowColor = m.color;
    ctx.shadowBlur = 8;
    roundRect(ctx, x, y, cw, chh, cw * 0.3);
    ctx.fill();
    ctx.shadowBlur = 0;
    // windshield sits at the leading end of the car
    ctx.fillStyle = 'rgba(12,16,26,0.85)';
    roundRect(ctx, x + cw * 0.14, m.dir > 0 ? y + chh * 0.12 : y + chh * 0.58,
              cw * 0.72, chh * 0.3, 2);
    ctx.fill();
    // headlights
    ctx.fillStyle = '#fff7cc';
    const hlY = m.dir > 0 ? y + 1 : y + chh - 3;
    ctx.fillRect(x + cw * 0.12, hlY, cw * 0.2, 2);
    ctx.fillRect(x + cw * 0.68, hlY, cw * 0.2, 2);
    ctx.restore();
  });

  // chicken agent + sensor ring
  const toXY = (p) => [X(p[0]), Y(p[1])];
  drawTrail(ctx, trailPoints(frames, cursor, 10, toXY), 7, 'rgba(255,255,255,0.8)');
  const pl = f0?.p ? lerpPos(f0.p, f1?.p, t, 4) : null;
  if (pl) {
    const ax = X(pl[0]);
    const ay = Y(pl[1]);
    // sensor ring
    const sr = SX(layout.sensor_range || 3.5);
    ctx.save();
    ctx.strokeStyle = 'rgba(56,189,248,0.35)';
    ctx.setLineDash([5, 7]);
    ctx.lineWidth = 1.4;
    ctx.beginPath();
    ctx.arc(ax, ay, sr, 0, Math.PI * 2);
    ctx.stroke();
    // radar sweep
    const sweepA = (timeMs / 900) % (Math.PI * 2);
    const grad = ctx.createConicGradient
      ? ctx.createConicGradient(sweepA, ax, ay)
      : null;
    if (grad) {
      grad.addColorStop(0, 'rgba(56,189,248,0.22)');
      grad.addColorStop(0.12, 'transparent');
      grad.addColorStop(1, 'transparent');
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.arc(ax, ay, sr, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.setLineDash([]);
    // sensor lines to cars in range
    cars0.forEach((c0, i) => {
      const dx = X(c0[0]) - ax;
      const dy = Y(c0[1]) - ay;
      const d = Math.hypot(dx, dy);
      if (d < sr) {
        const closeness = 1 - d / sr;
        ctx.strokeStyle = `rgba(${closeness > 0.55 ? '255,77,90' : '56,189,248'},${0.25 + closeness * 0.5})`;
        ctx.lineWidth = 1 + closeness * 1.4;
        ctx.beginPath();
        ctx.moveTo(ax, ay);
        ctx.lineTo(X(c0[0]), Y(c0[1]));
        ctx.stroke();
      }
    });
    ctx.restore();

    // the chicken
    glow(ctx, ax, ay, 26, 'rgba(255,255,255,0.9)', 0.3);
    ctx.save();
    // body
    ctx.fillStyle = '#f8fafc';
    ctx.shadowColor = '#fff';
    ctx.shadowBlur = 10;
    ctx.beginPath();
    ctx.arc(ax, ay, 9, 0, Math.PI * 2);
    ctx.fill();
    ctx.shadowBlur = 0;
    // wing bob
    ctx.fillStyle = '#e2e8f0';
    ctx.beginPath();
    ctx.ellipse(ax - 3, ay + 1 + Math.sin(timeMs / 180) * 0.8, 4.5, 3, -0.4, 0, Math.PI * 2);
    ctx.fill();
    // comb
    ctx.fillStyle = '#ef4444';
    for (let i = 0; i < 3; i++) {
      ctx.beginPath();
      ctx.arc(ax - 2 + i * 2.4, ay - 9.5, 1.8, 0, Math.PI * 2);
      ctx.fill();
    }
    // beak points toward the goal
    ctx.fillStyle = '#f59e0b';
    ctx.beginPath();
    ctx.moveTo(ax + 8, ay - 2.4);
    ctx.lineTo(ax + 13.5, ay);
    ctx.lineTo(ax + 8, ay + 2.4);
    ctx.closePath();
    ctx.fill();
    // eye
    ctx.fillStyle = '#111827';
    ctx.beginPath();
    ctx.arc(ax + 4.5, ay - 3, 1.5, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
    floatingReward(ctx, ax, ay, f0?.r, t);
  }

  // outcome
  const oc = outcomeOf(f0);
  if (oc) {
    fullFlash(ctx, W, H, oc === 'good' ? COL.green : COL.red, Math.max(0, 0.16 - t * 0.14));
    const ev = (f0.ev || []).join(' ').toLowerCase();
    let text = oc === 'good' ? 'CROSSED!' : 'FAILED';
    if (ev.includes('hit') || ev.includes('collision')) text = 'HIT BY TRAFFIC!';
    else if (ev.includes('road')) text = 'OFF THE ROAD!';
    else if (!oc.includes('good') && ev.includes('time')) text = 'TIME UP!';
    outcomeText(ctx, W, H, text, oc === 'good' ? COL.green : COL.red, timeMs);
  }
}
