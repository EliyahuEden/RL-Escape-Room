"""Canvas renderer for Room 1 — Pacman with smooth animation, glowing coins, ghost."""

from __future__ import annotations
from rl.envs.pacman import PacmanEnv
from ui_canvas.canvas_core import render_canvas, render_canvas_replay

DRAW_JS = """
function drawFrame(ctx, W, H, f, prev, t, fx) {
  const rows = f.rows, cols = f.cols;
  const cw = W / cols, ch = (H - 28) / rows;  // leave room for HUD
  const oY = 28; // offset for HUD bar
  t = t || 1;

  // --- background ---
  ctx.fillStyle = '#0a0a1e';
  ctx.fillRect(0, oY, W, H - oY);

  // parse positions
  function parsePos(s) {
    if (!s) return null;
    const p = s.split(','); return [parseInt(p[0]), parseInt(p[1])];
  }
  function cellXY(r, c) { return [c * cw + cw/2, oY + r * ch + ch/2]; }

  const pos = parsePos(f.pos);
  const prevPos = prev ? parsePos(prev.pos) : pos;

  // --- cells ---
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const x = c * cw, y = oY + r * ch;
      const key = r + ',' + c;

      if (f.walls.indexOf(key) >= 0) {
        // neon-border wall
        ctx.fillStyle = '#141432';
        ctx.fillRect(x+1, y+1, cw-2, ch-2);
        ctx.strokeStyle = '#2233aa';
        ctx.lineWidth = 1.8;
        // draw borders only where adjacent cell is not a wall
        const notWall = function(rr,cc) {
          return rr<0||rr>=rows||cc<0||cc>=cols|| f.walls.indexOf(rr+','+cc)<0;
        };
        if (notWall(r-1,c)) { ctx.beginPath(); ctx.moveTo(x,y); ctx.lineTo(x+cw,y); ctx.stroke(); }
        if (notWall(r+1,c)) { ctx.beginPath(); ctx.moveTo(x,y+ch); ctx.lineTo(x+cw,y+ch); ctx.stroke(); }
        if (notWall(r,c-1)) { ctx.beginPath(); ctx.moveTo(x,y); ctx.lineTo(x,y+ch); ctx.stroke(); }
        if (notWall(r,c+1)) { ctx.beginPath(); ctx.moveTo(x+cw,y); ctx.lineTo(x+cw,y+ch); ctx.stroke(); }
      } else {
        // dark floor with subtle dots
        ctx.fillStyle = '#0d0d22';
        ctx.fillRect(x, y, cw, ch);
        ctx.fillStyle = '#18183a';
        ctx.beginPath(); ctx.arc(x+cw/2, y+ch/2, 1.5, 0, Math.PI*2); ctx.fill();
      }

      // slippery
      if (f.slippery.indexOf(key) >= 0) {
        ctx.fillStyle = 'rgba(70,140,255,0.12)';
        ctx.fillRect(x, y, cw, ch);
        ctx.strokeStyle = 'rgba(100,180,255,0.35)';
        ctx.lineWidth = 0.8;
        const cx2 = x+cw/2, cy2 = y+ch/2;
        for (let a = 0; a < 6; a++) {
          const ang = a * Math.PI/3;
          ctx.beginPath(); ctx.moveTo(cx2, cy2);
          ctx.lineTo(cx2+Math.cos(ang)*cw*0.28, cy2+Math.sin(ang)*ch*0.28);
          ctx.stroke();
        }
      }
    }
  }

  // --- coins with glow ---
  const time = (f.step || 0) * 0.3;
  (f.remaining || []).forEach(function(key) {
    const p = parsePos(key);
    const [cx2, cy2] = cellXY(p[0], p[1]);
    const pulse = 0.8 + 0.2 * Math.sin(time + p[0]*2 + p[1]*3);
    // glow
    const grd = ctx.createRadialGradient(cx2, cy2, 0, cx2, cy2, cw*0.4*pulse);
    grd.addColorStop(0, 'rgba(255,215,0,0.5)');
    grd.addColorStop(1, 'rgba(255,215,0,0)');
    ctx.fillStyle = grd;
    ctx.beginPath(); ctx.arc(cx2, cy2, cw*0.4*pulse, 0, Math.PI*2); ctx.fill();
    // coin
    ctx.fillStyle = '#ffd700';
    ctx.beginPath(); ctx.arc(cx2, cy2, cw*0.17*pulse, 0, Math.PI*2); ctx.fill();
    ctx.strokeStyle = '#b8860b'; ctx.lineWidth = 1.5; ctx.stroke();
    ctx.fillStyle = '#b8860b';
    ctx.font = 'bold '+(cw*0.2)+'px sans-serif';
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText('$', cx2, cy2+1);
  });

  // --- door ---
  if (f.door) {
    const dp = parsePos(f.door);
    const [dx, dy] = cellXY(dp[0], dp[1]);
    const doorOpen = (f.remaining || []).length === 0;
    // glow when open
    if (doorOpen) {
      const grd = ctx.createRadialGradient(dx, dy, 0, dx, dy, cw*0.6);
      grd.addColorStop(0, 'rgba(76,175,80,0.4)');
      grd.addColorStop(1, 'rgba(76,175,80,0)');
      ctx.fillStyle = grd;
      ctx.beginPath(); ctx.arc(dx, dy, cw*0.6, 0, Math.PI*2); ctx.fill();
    }
    ctx.fillStyle = doorOpen ? '#4caf50' : '#6d4c41';
    ctx.fillRect(dx-cw*0.28, dy-ch*0.35, cw*0.56, ch*0.7);
    ctx.fillStyle = '#ffd54f';
    ctx.beginPath(); ctx.arc(dx+cw*0.15, dy, cw*0.04, 0, Math.PI*2); ctx.fill();
  }

  // --- ghost ---
  if (f.guard) {
    const gp = parsePos(f.guard);
    const gpPrev = prev && prev.guard ? parsePos(prev.guard) : gp;
    const ir = lerp(gpPrev[0], gp[0], t), ic = lerp(gpPrev[1], gp[1], t);
    const [gx, gy] = cellXY(ir, ic);
    const wobble = Math.sin(time*3) * 2;
    // ghost body
    ctx.fillStyle = '#e53935';
    ctx.beginPath();
    ctx.arc(gx, gy-ch*0.06, cw*0.32, Math.PI, 0);
    ctx.lineTo(gx+cw*0.32, gy+ch*0.28);
    for (let i = 4; i >= 0; i--) {
      const wx = gx-cw*0.32 + i*cw*0.16;
      ctx.lineTo(wx, gy+ch*0.28 + ((i%2===0)?-ch*0.08+wobble*0.5:wobble*0.5));
    }
    ctx.closePath(); ctx.fill();
    // eyes
    ctx.fillStyle = 'white';
    ctx.beginPath(); ctx.arc(gx-cw*0.1, gy-ch*0.1, cw*0.09, 0, Math.PI*2); ctx.fill();
    ctx.beginPath(); ctx.arc(gx+cw*0.1, gy-ch*0.1, cw*0.09, 0, Math.PI*2); ctx.fill();
    ctx.fillStyle = '#1a237e';
    ctx.beginPath(); ctx.arc(gx-cw*0.07, gy-ch*0.09, cw*0.045, 0, Math.PI*2); ctx.fill();
    ctx.beginPath(); ctx.arc(gx+cw*0.13, gy-ch*0.09, cw*0.045, 0, Math.PI*2); ctx.fill();
  }

  // --- pacman with interpolation ---
  if (pos) {
    const ir = lerp(prevPos[0], pos[0], t), ic = lerp(prevPos[1], pos[1], t);
    const [px, py] = cellXY(ir, ic);

    // trail
    if (fx && fx.trail) { fx.trail.add(px, py); fx.trail.draw(ctx); }

    // mouth animation
    const mouthAmt = 0.25 + 0.2 * Math.abs(Math.sin(time * 4));
    const action = f.action;
    let angle = 0;
    if (action === 0) angle = -Math.PI/2;
    else if (action === 1) angle = Math.PI/2;
    else if (action === 2) angle = Math.PI;

    ctx.fillStyle = '#ffd23f';
    ctx.beginPath();
    ctx.moveTo(px, py);
    ctx.arc(px, py, cw*0.38, angle+mouthAmt, angle+Math.PI*2-mouthAmt);
    ctx.closePath(); ctx.fill();
    ctx.strokeStyle = '#e6b800'; ctx.lineWidth = 1.2; ctx.stroke();
    // eye
    const eyeA = angle - 0.5;
    ctx.fillStyle = '#111';
    ctx.beginPath();
    ctx.arc(px+Math.cos(eyeA)*cw*0.16, py+Math.sin(eyeA)*cw*0.16, cw*0.05, 0, Math.PI*2);
    ctx.fill();

    // particles on coin collect
    if (fx && fx.particles && prev && f.remaining && prev.remaining &&
        f.remaining.length < prev.remaining.length) {
      fx.particles.emit(px, py, 12, '#ffd700', 80, 0.6);
    }
    // particles on escape
    if (fx && fx.particles && f.escaped) {
      fx.particles.emit(px, py, 25, '#4caf50', 120, 1.0);
    }
    // particles on caught
    if (fx && fx.particles && f.caught) {
      fx.particles.emit(px, py, 20, '#ff1744', 100, 0.8);
    }
  }

  // flash on escape/caught
  if (f.escaped) drawFlash(ctx, W, H, '#4caf50', 0.15);
  if (f.caught) drawFlash(ctx, W, H, '#ff1744', 0.2);
}
"""


def frame_to_canvas(env: PacmanEnv, frame: dict) -> dict:
    walls = [f"{r},{c}" for r, c in sorted(env.walls)]
    slippery = [f"{r},{c}" for r, c in sorted(env.slippery)]
    remaining = [f"{r},{c}" for r, c in frame.get("remaining", [])]
    pos = frame.get("pos", env.start)
    guard = frame.get("guard")
    return {
        "rows": env.rows, "cols": env.cols,
        "walls": walls, "slippery": slippery, "remaining": remaining,
        "pos": f"{pos[0]},{pos[1]}",
        "door": f"{env.door[0]},{env.door[1]}",
        "guard": f"{guard[0]},{guard[1]}" if guard else None,
        "step": frame.get("step", 0),
        "action": frame.get("action"),
        "escaped": frame.get("escaped", False),
        "caught": frame.get("caught", False),
        "cum_reward": frame.get("cum_reward"),
        "event": "Escaped!" if frame.get("escaped") else
                 "Caught!" if frame.get("caught") else
                 f"Coins left: {len(remaining)}",
    }


def draw_pacman_canvas(env: PacmanEnv, frame: dict, title: str = "") -> None:
    data = frame_to_canvas(env, frame)
    if title: data["event"] = title
    render_canvas(data, DRAW_JS, width=520, height=548, title=data["event"], bg_color="#0a0a1e")


def replay_pacman_canvas(env: PacmanEnv, frames: list, key: str = "pac_replay") -> None:
    data = [frame_to_canvas(env, f) for f in frames]
    render_canvas_replay(data, DRAW_JS, width=520, height=548, bg_color="#0a0a1e",
                         fps=8, trail_color="rgba(255,210,63,0.25)", trail_len=15, key=key)
