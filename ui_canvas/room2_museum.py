"""Canvas renderer for Room 2 — Museum Heist with stealth visuals, alarm effects."""

from __future__ import annotations
from rl.envs.museum import MuseumEnv
from ui_canvas.canvas_core import render_canvas, render_canvas_replay

DRAW_JS = """
function drawFrame(ctx, W, H, f, prev, t, fx) {
  const rows = f.rows, cols = f.cols;
  const cw = W / cols, ch = (H - 28) / rows;
  const oY = 28;
  t = t || 1;
  const alarm = f.alarm || false;
  const time = (f.step || 0) * 0.4;

  function parsePos(s) {
    if (!s) return null;
    const p = s.split(','); return [parseInt(p[0]), parseInt(p[1])];
  }
  function cellXY(r, c) { return [c * cw + cw/2, oY + r * ch + ch/2]; }

  // --- floor ---
  ctx.fillStyle = alarm ? '#2a1a1a' : '#1a1510';
  ctx.fillRect(0, oY, W, H - oY);

  // alarm pulse
  if (alarm) {
    const pulse = 0.08 + 0.06 * Math.sin(time * 6);
    drawFlash(ctx, W, H, '#ff0000', pulse);
  }

  const pos = parsePos(f.pos);
  const prevPos = prev ? parsePos(prev.pos) : pos;

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const x = c * cw, y = oY + r * ch;
      const key = r + ',' + c;

      if (f.walls.indexOf(key) >= 0) {
        ctx.fillStyle = '#3a3530';
        ctx.fillRect(x, y, cw, ch);
        ctx.fillStyle = '#4a4540';
        ctx.fillRect(x+1, y+1, cw-2, ch-2);
        // brick texture
        ctx.strokeStyle = '#2a2520';
        ctx.lineWidth = 0.5;
        ctx.beginPath();
        ctx.moveTo(x, y+ch*0.33); ctx.lineTo(x+cw, y+ch*0.33);
        ctx.moveTo(x, y+ch*0.66); ctx.lineTo(x+cw, y+ch*0.66);
        ctx.moveTo(x+cw*0.5, y); ctx.lineTo(x+cw*0.5, y+ch*0.33);
        ctx.moveTo(x+cw*0.25, y+ch*0.33); ctx.lineTo(x+cw*0.25, y+ch*0.66);
        ctx.moveTo(x+cw*0.75, y+ch*0.66); ctx.lineTo(x+cw*0.75, y+ch);
        ctx.stroke();
      } else {
        // marble checkerboard
        ctx.fillStyle = (r+c)%2===0 ? '#d4c8a8' : '#c4b898';
        ctx.fillRect(x, y, cw, ch);
        ctx.strokeStyle = 'rgba(180,170,140,0.5)';
        ctx.lineWidth = 0.3;
        ctx.strokeRect(x, y, cw, ch);
      }

      // slippery
      if (f.slippery.indexOf(key) >= 0) {
        ctx.fillStyle = 'rgba(180,220,255,0.2)';
        ctx.fillRect(x, y, cw, ch);
        ctx.fillStyle = 'rgba(255,255,255,0.25)';
        ctx.beginPath();
        ctx.ellipse(x+cw*0.3, y+ch*0.35, cw*0.18, ch*0.08, -0.4, 0, Math.PI*2);
        ctx.fill();
      }

      // camera zone
      if (f.cameras.indexOf(key) >= 0) {
        const camAlpha = alarm ? 0.35 + 0.15*Math.sin(time*8) : 0.18;
        ctx.fillStyle = 'rgba(255,0,0,' + camAlpha + ')';
        ctx.fillRect(x, y, cw, ch);
        // scan lines
        ctx.strokeStyle = 'rgba(255,50,50,' + (alarm?0.5:0.2) + ')';
        ctx.lineWidth = 0.8;
        for (let i = 1; i <= 3; i++) {
          ctx.beginPath(); ctx.moveTo(x, y+ch*i/4); ctx.lineTo(x+cw, y+ch*i/4); ctx.stroke();
        }
        ctx.fillStyle = alarm ? '#ff1744' : '#c62828';
        ctx.font = 'bold '+(cw*0.35)+'px sans-serif';
        ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.fillText('!', x+cw/2, y+ch/2);
      }

      // trap
      if (f.traps.indexOf(key) >= 0) {
        ctx.fillStyle = 'rgba(255,60,0,0.15)';
        ctx.fillRect(x, y, cw, ch);
        ctx.strokeStyle = '#ff3d00';
        ctx.lineWidth = 1.5;
        ctx.setLineDash([4,4]);
        ctx.beginPath(); ctx.moveTo(x+3, y+ch/2); ctx.lineTo(x+cw-3, y+ch/2); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(x+cw/2, y+3); ctx.lineTo(x+cw/2, y+ch-3); ctx.stroke();
        ctx.setLineDash([]);
      }
    }
  }

  // --- diamond ---
  if (f.diamond && !f.has_diamond) {
    const dp = parsePos(f.diamond);
    const [dx, dy] = cellXY(dp[0], dp[1]);
    const pulse = 0.9 + 0.1*Math.sin(time*3);
    const grd = ctx.createRadialGradient(dx, dy, 0, dx, dy, cw*0.45*pulse);
    grd.addColorStop(0, 'rgba(79,195,247,0.6)');
    grd.addColorStop(1, 'rgba(79,195,247,0)');
    ctx.fillStyle = grd;
    ctx.beginPath(); ctx.arc(dx, dy, cw*0.45*pulse, 0, Math.PI*2); ctx.fill();
    ctx.fillStyle = '#4fc3f7';
    ctx.beginPath();
    ctx.moveTo(dx, dy-ch*0.28); ctx.lineTo(dx+cw*0.22, dy);
    ctx.lineTo(dx, dy+ch*0.28); ctx.lineTo(dx-cw*0.22, dy);
    ctx.closePath(); ctx.fill();
    ctx.strokeStyle = '#0288d1'; ctx.lineWidth = 1.5; ctx.stroke();
  }

  // --- exit door ---
  if (f.exit_cell) {
    const ep = parsePos(f.exit_cell);
    const [ex, ey] = cellXY(ep[0], ep[1]);
    const open = f.has_diamond;
    if (open) {
      const grd = ctx.createRadialGradient(ex, ey, 0, ex, ey, cw*0.55);
      grd.addColorStop(0, 'rgba(76,175,80,0.45)');
      grd.addColorStop(1, 'rgba(76,175,80,0)');
      ctx.fillStyle = grd;
      ctx.beginPath(); ctx.arc(ex, ey, cw*0.55, 0, Math.PI*2); ctx.fill();
    }
    ctx.fillStyle = open ? '#4caf50' : '#5d4037';
    ctx.fillRect(ex-cw*0.25, ey-ch*0.35, cw*0.5, ch*0.7);
    ctx.fillStyle = '#ffd54f';
    ctx.beginPath(); ctx.arc(ex+cw*0.12, ey, cw*0.035, 0, Math.PI*2); ctx.fill();
  }

  // --- guards with interpolation ---
  const guards = f.guards || [];
  const prevGuards = (prev && prev.guards) || guards;
  for (let gi = 0; gi < guards.length; gi++) {
    const gp = parsePos(guards[gi]);
    const gpPrev = gi < prevGuards.length ? parsePos(prevGuards[gi]) : gp;
    const ir = lerp(gpPrev[0], gp[0], t), ic = lerp(gpPrev[1], gp[1], t);
    const [gx, gy] = cellXY(ir, ic);
    const guardColor = alarm ? '#d32f2f' : '#1565c0';
    // body
    ctx.fillStyle = guardColor;
    ctx.fillRect(gx-cw*0.2, gy-ch*0.02, cw*0.4, ch*0.32);
    // head
    ctx.fillStyle = '#ffcc80';
    ctx.beginPath(); ctx.arc(gx, gy-ch*0.14, cw*0.12, 0, Math.PI*2); ctx.fill();
    // hat
    ctx.fillStyle = guardColor;
    ctx.fillRect(gx-cw*0.15, gy-ch*0.25, cw*0.3, ch*0.06);
    // alarm beacon
    if (alarm) {
      const beaconAlpha = 0.5 + 0.5*Math.sin(time*10);
      ctx.fillStyle = 'rgba(255,23,68,'+beaconAlpha+')';
      ctx.beginPath(); ctx.arc(gx, gy-ch*0.3, cw*0.06, 0, Math.PI*2); ctx.fill();
    }
    // flashlight cone
    ctx.fillStyle = alarm ? 'rgba(255,200,50,0.12)' : 'rgba(255,255,200,0.06)';
    ctx.beginPath();
    ctx.moveTo(gx, gy);
    ctx.lineTo(gx+cw*0.8, gy-ch*0.5);
    ctx.lineTo(gx+cw*0.8, gy+ch*0.5);
    ctx.closePath(); ctx.fill();
  }

  // --- robber with interpolation ---
  if (pos) {
    const ir = lerp(prevPos[0], pos[0], t), ic = lerp(prevPos[1], pos[1], t);
    const [px, py] = cellXY(ir, ic);
    if (fx && fx.trail) { fx.trail.add(px, py); fx.trail.draw(ctx); }
    // shadow
    ctx.fillStyle = 'rgba(0,0,0,0.3)';
    ctx.beginPath(); ctx.ellipse(px, py+ch*0.25, cw*0.2, ch*0.06, 0, 0, Math.PI*2); ctx.fill();
    // body
    ctx.fillStyle = '#212121';
    ctx.fillRect(px-cw*0.2, py-ch*0.02, cw*0.4, ch*0.32);
    // head
    ctx.fillStyle = '#ffcc80';
    ctx.beginPath(); ctx.arc(px, py-ch*0.14, cw*0.12, 0, Math.PI*2); ctx.fill();
    // mask
    ctx.fillStyle = '#212121';
    ctx.fillRect(px-cw*0.14, py-ch*0.19, cw*0.28, ch*0.07);
    ctx.fillStyle = '#fff';
    ctx.beginPath(); ctx.arc(px-cw*0.05, py-ch*0.16, cw*0.025, 0, Math.PI*2); ctx.fill();
    ctx.beginPath(); ctx.arc(px+cw*0.05, py-ch*0.16, cw*0.025, 0, Math.PI*2); ctx.fill();
    // diamond indicator
    if (f.has_diamond) {
      ctx.fillStyle = '#4fc3f7';
      ctx.beginPath();
      ctx.moveTo(px+cw*0.18, py-ch*0.28);
      ctx.lineTo(px+cw*0.25, py-ch*0.2);
      ctx.lineTo(px+cw*0.18, py-ch*0.12);
      ctx.lineTo(px+cw*0.11, py-ch*0.2);
      ctx.closePath(); ctx.fill();
    }

    // particles
    if (fx && fx.particles) {
      if (f.camera) fx.particles.emit(px, py, 8, '#ff1744', 60, 0.4);
      if (f.success) fx.particles.emit(px, py, 30, '#4caf50', 120, 1.0);
      if (f.caught) fx.particles.emit(px, py, 20, '#ff1744', 100, 0.8);
    }
  }

  if (f.success) drawFlash(ctx, W, H, '#4caf50', 0.12);
  if (f.caught) drawFlash(ctx, W, H, '#ff1744', 0.18);
}
"""


def frame_to_canvas(env: MuseumEnv, frame: dict) -> dict:
    walls = [f"{r},{c}" for r, c in sorted(env.walls)]
    cameras = [f"{r},{c}" for r, c in sorted(env.cameras)]
    traps = [f"{r},{c}" for r, c in sorted(env.traps)]
    slippery = [f"{r},{c}" for r, c in sorted(env.slippery)]
    guards = [f"{g[0]},{g[1]}" for g in frame.get("guards", [])]
    pos = frame.get("pos", env.start)
    return {
        "rows": env.rows, "cols": env.cols,
        "walls": walls, "cameras": cameras, "traps": traps,
        "slippery": slippery, "guards": guards,
        "pos": f"{pos[0]},{pos[1]}",
        "diamond": f"{env.diamond[0]},{env.diamond[1]}",
        "exit_cell": f"{env.exit[0]},{env.exit[1]}",
        "has_diamond": bool(frame.get("has_diamond", 0)),
        "alarm": frame.get("alarm", False),
        "camera": frame.get("camera", False),
        "success": frame.get("success", False),
        "caught": frame.get("caught", False),
        "step": frame.get("step", 0),
        "cum_reward": frame.get("cum_reward"),
        "event": "Clean getaway!" if frame.get("success") else
                 "Caught!" if frame.get("caught") else
                 "ALARM!" if frame.get("alarm") else
                 "Got the diamond!" if frame.get("has_diamond") else
                 "Sneaking...",
    }


def draw_museum_canvas(env: MuseumEnv, frame: dict, title: str = "") -> None:
    data = frame_to_canvas(env, frame)
    if title: data["event"] = title
    render_canvas(data, DRAW_JS, width=520, height=548, title=data["event"], bg_color="#1a1510")


def replay_museum_canvas(env: MuseumEnv, frames: list, key: str = "mus_replay") -> None:
    data = [frame_to_canvas(env, f) for f in frames]
    render_canvas_replay(data, DRAW_JS, width=520, height=548, bg_color="#1a1510",
                         fps=8, trail_color="rgba(150,150,150,0.2)", trail_len=12, key=key)
