"""Canvas renderer for Room 3 — Street Race with skid marks, boost effects."""

from __future__ import annotations
from rl.envs.racing import RacingEnv
from ui_canvas.canvas_core import render_canvas, render_canvas_replay

DRAW_JS = """
function drawFrame(ctx, W, H, f, prev, t, fx) {
  const rows = f.rows, cols = f.cols;
  const cw = W / cols, ch = (H - 28) / rows;
  const oY = 28;
  t = t || 1;
  const time = (f.step || 0) * 0.3;

  function parsePos(s) {
    if (!s) return null;
    const p = s.split(','); return [parseInt(p[0]), parseInt(p[1])];
  }
  function cellXY(r, c) { return [c * cw + cw/2, oY + r * ch + ch/2]; }

  // --- asphalt ---
  ctx.fillStyle = '#2d2d2d';
  ctx.fillRect(0, oY, W, H - oY);

  const pos = parsePos(f.pos);
  const prevPos = prev ? parsePos(prev.pos) : pos;

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const x = c * cw, y = oY + r * ch;
      const key = r + ',' + c;

      if (f.walls.indexOf(key) >= 0) {
        // building with windows
        ctx.fillStyle = '#5d4e37';
        ctx.fillRect(x, y, cw, ch);
        ctx.fillStyle = '#6d5e47';
        ctx.fillRect(x+2, y+2, cw-4, ch-4);
        ctx.fillStyle = '#8ecae6';
        const ws = cw * 0.18;
        ctx.fillRect(x+cw*0.12, y+ch*0.12, ws, ws);
        ctx.fillRect(x+cw*0.62, y+ch*0.12, ws, ws);
        ctx.fillRect(x+cw*0.12, y+ch*0.58, ws, ws);
        ctx.fillRect(x+cw*0.62, y+ch*0.58, ws, ws);
      } else {
        // road
        ctx.fillStyle = '#3a3a3a';
        ctx.fillRect(x, y, cw, ch);
        // road markings
        ctx.strokeStyle = '#333';
        ctx.lineWidth = 0.3;
        ctx.strokeRect(x, y, cw, ch);
      }

      // oil with rainbow
      if (f.oil.indexOf(key) >= 0) {
        ctx.fillStyle = 'rgba(10,10,10,0.7)';
        ctx.beginPath(); ctx.ellipse(x+cw/2, y+ch/2, cw*0.38, ch*0.28, 0.2, 0, Math.PI*2); ctx.fill();
        const grd = ctx.createLinearGradient(x, y, x+cw, y+ch);
        grd.addColorStop(0, 'rgba(120,0,220,0.25)');
        grd.addColorStop(0.5, 'rgba(0,220,120,0.25)');
        grd.addColorStop(1, 'rgba(220,120,0,0.25)');
        ctx.fillStyle = grd;
        ctx.beginPath(); ctx.ellipse(x+cw/2, y+ch/2, cw*0.34, ch*0.24, 0.2, 0, Math.PI*2); ctx.fill();
      }

      // mud
      if (f.mud.indexOf(key) >= 0) {
        ctx.fillStyle = '#5d4037';
        ctx.beginPath(); ctx.ellipse(x+cw/2, y+ch/2, cw*0.4, ch*0.3, 0, 0, Math.PI*2); ctx.fill();
        ctx.fillStyle = '#4e342e';
        for (let i = 0; i < 5; i++) {
          ctx.beginPath();
          ctx.arc(x+cw*(0.15+i*0.18), y+ch*(0.3+(i%2)*0.35), cw*0.05, 0, Math.PI*2);
          ctx.fill();
        }
      }

      // crash barrier
      if (f.crash.indexOf(key) >= 0) {
        ctx.fillStyle = '#c62828';
        ctx.fillRect(x+3, y+3, cw-6, ch-6);
        ctx.strokeStyle = 'white'; ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(x+cw*0.2, y+3); ctx.lineTo(x+cw*0.8, y+ch-3);
        ctx.moveTo(x+3, y+ch*0.5); ctx.lineTo(x+cw*0.5, y+ch-3);
        ctx.moveTo(x+cw*0.5, y+3); ctx.lineTo(x+cw-3, y+ch*0.5);
        ctx.stroke();
        ctx.fillStyle = 'white';
        ctx.font = 'bold '+(cw*0.35)+'px sans-serif';
        ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.fillText('✕', x+cw/2, y+ch/2);
      }

      // booster with glow
      if (f.boosters.indexOf(key) >= 0) {
        const bPulse = 0.85 + 0.15*Math.sin(time*4 + c*2);
        const grd = ctx.createRadialGradient(x+cw/2, y+ch/2, 0, x+cw/2, y+ch/2, cw*0.45*bPulse);
        grd.addColorStop(0, 'rgba(0,229,255,0.5)');
        grd.addColorStop(1, 'rgba(0,229,255,0)');
        ctx.fillStyle = grd;
        ctx.beginPath(); ctx.arc(x+cw/2, y+ch/2, cw*0.45*bPulse, 0, Math.PI*2); ctx.fill();
        ctx.fillStyle = '#00e5ff';
        ctx.fillRect(x+cw*0.15, y+ch*0.15, cw*0.7, ch*0.7);
        // lightning bolt
        ctx.fillStyle = '#ffea00';
        ctx.beginPath();
        ctx.moveTo(x+cw*0.45, y+ch*0.18);
        ctx.lineTo(x+cw*0.58, y+ch*0.45);
        ctx.lineTo(x+cw*0.47, y+ch*0.45);
        ctx.lineTo(x+cw*0.55, y+ch*0.82);
        ctx.lineTo(x+cw*0.42, y+ch*0.55);
        ctx.lineTo(x+cw*0.53, y+ch*0.55);
        ctx.closePath(); ctx.fill();
      }
    }
  }

  // start
  if (f.start) {
    const sp = parsePos(f.start);
    const [sx, sy] = cellXY(sp[0], sp[1]);
    ctx.fillStyle = 'rgba(76,175,80,0.2)';
    ctx.fillRect(sp[1]*cw, oY+sp[0]*ch, cw, ch);
    ctx.fillStyle = '#66bb6a';
    ctx.font = (cw*0.2)+'px sans-serif';
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText('GO', sx, sy);
  }

  // finish
  if (f.finish) {
    const fp = parsePos(f.finish);
    const fx2 = fp[1]*cw, fy2 = oY+fp[0]*ch;
    const unlocked = f.finish_unlocked;
    ctx.fillStyle = unlocked ? 'rgba(76,175,80,0.3)' : 'rgba(200,150,0,0.2)';
    ctx.fillRect(fx2, fy2, cw, ch);
    const cs = cw/4;
    for (let i = 0; i < 4; i++)
      for (let j = 0; j < 4; j++)
        if ((i+j)%2===0) {
          ctx.fillStyle = unlocked ? '#222' : '#666';
          ctx.fillRect(fx2+i*cs, fy2+j*cs, cs, cs);
        }
    if (!unlocked) {
      ctx.fillStyle = '#ff6f00';
      ctx.font = 'bold '+(cw*0.25)+'px sans-serif';
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText('🔒', fx2+cw/2, fy2+ch/2);
    }
  }

  // --- car with interpolation ---
  if (pos) {
    const ir = lerp(prevPos[0], pos[0], t), ic = lerp(prevPos[1], pos[1], t);
    const [px, py] = cellXY(ir, ic);

    // trail (skid marks)
    if (fx && fx.trail) { fx.trail.add(px, py); fx.trail.draw(ctx); }

    // shadow
    ctx.fillStyle = 'rgba(0,0,0,0.3)';
    ctx.beginPath(); ctx.ellipse(px, py+ch*0.22, cw*0.28, ch*0.08, 0, 0, Math.PI*2); ctx.fill();
    // car body
    ctx.fillStyle = '#e53935';
    ctx.beginPath(); ctx.roundRect(px-cw*0.28, py-ch*0.18, cw*0.56, ch*0.36, 4); ctx.fill();
    ctx.strokeStyle = '#b71c1c'; ctx.lineWidth = 1; ctx.stroke();
    // windshield
    ctx.fillStyle = '#bbdefb';
    ctx.fillRect(px-cw*0.1, py-ch*0.15, cw*0.2, ch*0.1);
    // wheels
    ctx.fillStyle = '#111';
    ctx.beginPath(); ctx.arc(px-cw*0.2, py-ch*0.2, cw*0.06, 0, Math.PI*2); ctx.fill();
    ctx.beginPath(); ctx.arc(px+cw*0.2, py-ch*0.2, cw*0.06, 0, Math.PI*2); ctx.fill();
    ctx.beginPath(); ctx.arc(px-cw*0.2, py+ch*0.2, cw*0.06, 0, Math.PI*2); ctx.fill();
    ctx.beginPath(); ctx.arc(px+cw*0.2, py+ch*0.2, cw*0.06, 0, Math.PI*2); ctx.fill();
    // number
    ctx.fillStyle = 'white';
    ctx.font = 'bold '+(cw*0.2)+'px sans-serif';
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText('1', px, py+ch*0.04);

    // particles
    if (fx && fx.particles) {
      if (f.crash_event) fx.particles.emit(px, py, 25, '#ff6600', 120, 0.8);
      if (f.success) fx.particles.emit(px, py, 30, '#4caf50', 100, 1.0);
      if (prev && f.collected > (prev.collected||0)) fx.particles.emit(px, py, 15, '#00e5ff', 80, 0.5);
    }
  }

  if (f.crash_event) drawFlash(ctx, W, H, '#ff3300', 0.2);
  if (f.success) drawFlash(ctx, W, H, '#4caf50', 0.12);
}
"""


def frame_to_canvas(env: RacingEnv, frame: dict) -> dict:
    walls = [f"{r},{c}" for r, c in sorted(env.walls)]
    oil = [f"{r},{c}" for r, c in sorted(env.oil)]
    mud = [f"{r},{c}" for r, c in sorted(env.mud)]
    crash = [f"{r},{c}" for r, c in sorted(env.crash)]
    bmask = frame.get("bmask", 0)
    remaining = env.remaining_boosters(bmask)
    boosters = [f"{r},{c}" for r, c in remaining]
    pos = frame.get("pos", env.start)
    collected = env.collected_count(bmask)
    return {
        "rows": env.rows, "cols": env.cols,
        "walls": walls, "oil": oil, "mud": mud, "crash": crash, "boosters": boosters,
        "pos": f"{pos[0]},{pos[1]}",
        "start": f"{env.start[0]},{env.start[1]}",
        "finish": f"{env.finish[0]},{env.finish[1]}",
        "finish_unlocked": env.finish_unlocked(bmask),
        "collected": collected,
        "step": frame.get("step", 0),
        "cum_reward": frame.get("cum_reward"),
        "crash_event": frame.get("crash", False),
        "success": frame.get("success", False),
        "event": "Crashed!" if frame.get("crash") else
                 "Finished!" if frame.get("success") else
                 f"⚡ {collected}/{env.min_boosters} needed" if not env.finish_unlocked(bmask) else
                 "Finish unlocked! Go!",
    }


def draw_racing_canvas(env: RacingEnv, frame: dict, title: str = "") -> None:
    data = frame_to_canvas(env, frame)
    if title: data["event"] = title
    render_canvas(data, DRAW_JS, width=520, height=548, title=data["event"], bg_color="#2d2d2d")


def replay_racing_canvas(env: RacingEnv, frames: list, key: str = "race_replay") -> None:
    data = [frame_to_canvas(env, f) for f in frames]
    render_canvas_replay(data, DRAW_JS, width=520, height=548, bg_color="#2d2d2d",
                         fps=8, trail_color="rgba(100,100,100,0.4)", trail_len=20, key=key)
