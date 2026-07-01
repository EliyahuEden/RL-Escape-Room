"""Canvas renderer for Room 4 — Football with 3D ball arc, trail, goal flash."""

from __future__ import annotations
from rl.envs.football import FootballEnv, FreeKickEnv
from ui_canvas.canvas_core import render_canvas, render_canvas_replay

DRAW_JS = """
function drawFrame(ctx, W, H, f, prev, t, fx) {
  const pw = f.pitch_w || 10, ph = f.pitch_h || 10;
  const oY = 28;
  const pH = H - oY;
  const sx = W / pw, sy = pH / ph;
  function tx(v) { return v * sx; }
  function ty(v) { return oY + pH - v * sy; }
  t = t || 1;
  const time = (f.step || 0) * 0.3;

  // --- grass with stripes ---
  for (let i = 0; i < 12; i++) {
    ctx.fillStyle = i%2===0 ? '#2e7d32' : '#338a36';
    ctx.fillRect(0, oY + i*pH/12, W, pH/12);
  }

  // pitch markings
  ctx.strokeStyle = 'rgba(255,255,255,0.2)';
  ctx.lineWidth = 1;
  ctx.beginPath(); ctx.arc(tx(pw/2), ty(ph/2), tx(1.8), 0, Math.PI*2); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(tx(pw/2), oY); ctx.lineTo(tx(pw/2), H); ctx.stroke();

  // penalty area
  if (f.shoot_x != null) {
    ctx.fillStyle = 'rgba(255,255,255,0.04)';
    ctx.fillRect(tx(f.shoot_x), oY, tx(pw-f.shoot_x), pH);
    ctx.setLineDash([6,6]);
    ctx.strokeStyle = 'rgba(255,255,255,0.25)';
    ctx.beginPath(); ctx.moveTo(tx(f.shoot_x), oY); ctx.lineTo(tx(f.shoot_x), H); ctx.stroke();
    ctx.setLineDash([]);
  }

  // goal
  const glo = f.goal_lo || 3.5, ghi = f.goal_hi || 6.5;
  // goal posts
  ctx.fillStyle = 'white';
  ctx.fillRect(tx(pw)-3, ty(ghi), 5, ty(glo)-ty(ghi));
  // net
  ctx.strokeStyle = 'rgba(200,200,200,0.3)';
  ctx.lineWidth = 0.5;
  for (let ny = glo; ny <= ghi; ny += 0.25) {
    ctx.beginPath(); ctx.moveTo(tx(pw)-3, ty(ny)); ctx.lineTo(tx(pw)+3, ty(ny)); ctx.stroke();
  }

  // --- defenders ---
  const defs = f.defenders || [];
  const prevDefs = (prev && prev.defenders) || defs;
  for (let di = 0; di < defs.length; di++) {
    const d = defs[di];
    const pd = di < prevDefs.length ? prevDefs[di] : d;
    const dx2 = tx(lerp(pd[0],d[0],t)), dy2 = ty(lerp(pd[1],d[1],t));
    // shadow
    ctx.fillStyle = 'rgba(0,0,0,0.2)';
    ctx.beginPath(); ctx.ellipse(dx2, dy2+10, 14, 5, 0, 0, Math.PI*2); ctx.fill();
    // body
    ctx.fillStyle = '#e53935';
    ctx.beginPath(); ctx.arc(dx2, dy2, 15, 0, Math.PI*2); ctx.fill();
    ctx.strokeStyle = '#b71c1c'; ctx.lineWidth = 2; ctx.stroke();
    // shirt number
    ctx.fillStyle = 'white';
    ctx.font = 'bold 9px sans-serif';
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText(''+(di+2), dx2, dy2);
  }

  // --- keeper ---
  if (f.keeper) {
    const kp = f.keeper, kpPrev = (prev && prev.keeper) || kp;
    const kx2 = tx(lerp(kpPrev[0],kp[0],t)), ky2 = ty(lerp(kpPrev[1],kp[1],t));
    const kr = (f.keeper_reach || 0.7) * sy;
    // save zone
    ctx.fillStyle = 'rgba(255,235,59,0.1)';
    ctx.fillRect(kx2-10, ky2-kr, 20, kr*2);
    // body
    ctx.fillStyle = '#ffeb3b';
    ctx.beginPath(); ctx.arc(kx2, ky2, 15, 0, Math.PI*2); ctx.fill();
    ctx.strokeStyle = '#f9a825'; ctx.lineWidth = 2; ctx.stroke();
    ctx.fillStyle = '#333';
    ctx.font = 'bold 9px sans-serif';
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText('GK', kx2, ky2);
    // gloves
    ctx.fillStyle = '#111';
    ctx.beginPath(); ctx.arc(kx2, ky2-kr, 5, 0, Math.PI*2); ctx.fill();
    ctx.beginPath(); ctx.arc(kx2, ky2+kr, 5, 0, Math.PI*2); ctx.fill();
  }

  // --- player ---
  if (f.player) {
    const pp = f.player, ppPrev = (prev && prev.player) || pp;
    const ppx = tx(lerp(ppPrev[0],pp[0],t)), ppy = ty(lerp(ppPrev[1],pp[1],t));
    if (fx && fx.trail && !f.ball_in_flight) { fx.trail.add(ppx, ppy); fx.trail.draw(ctx); }
    // shadow
    ctx.fillStyle = 'rgba(0,0,0,0.2)';
    ctx.beginPath(); ctx.ellipse(ppx, ppy+10, 14, 5, 0, 0, Math.PI*2); ctx.fill();
    // body
    ctx.fillStyle = '#1e88e5';
    ctx.beginPath(); ctx.arc(ppx, ppy, 15, 0, Math.PI*2); ctx.fill();
    ctx.strokeStyle = 'white'; ctx.lineWidth = 2; ctx.stroke();
    // head
    ctx.fillStyle = '#ffcc80';
    ctx.beginPath(); ctx.arc(ppx, ppy-10, 6, 0, Math.PI*2); ctx.fill();
    // number
    ctx.fillStyle = 'white';
    ctx.font = 'bold 9px sans-serif';
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText('10', ppx, ppy+2);
  }

  // --- ball with 3D height ---
  if (f.ball) {
    const bp = f.ball, bpPrev = (prev && prev.ball) || bp;
    const bxp = tx(lerp(bpPrev[0],bp[0],t)), byp = ty(lerp(bpPrev[1],bp[1],t));
    const bz = f.ball_z || 0;
    const prevBz = (prev && prev.ball_z) || 0;
    const iz = lerp(prevBz, bz, t);
    const lift = iz * 20;
    const bdraw = byp - lift;

    // ball trail when in flight
    if (f.ball_in_flight && fx && fx.trail) {
      fx.trail.add(bxp, bdraw);
      fx.trail.draw(ctx);
    }

    // shadow
    if (iz > 0.1) {
      const shSize = Math.max(3, 9 - iz*1.5);
      ctx.fillStyle = 'rgba(0,0,0,0.25)';
      ctx.beginPath(); ctx.ellipse(bxp, byp, shSize, shSize*0.4, 0, 0, Math.PI*2); ctx.fill();
      // connector
      if (lift > 4) {
        ctx.strokeStyle = 'rgba(180,180,180,0.3)';
        ctx.lineWidth = 0.8;
        ctx.setLineDash([2,3]);
        ctx.beginPath(); ctx.moveTo(bxp, byp); ctx.lineTo(bxp, bdraw+8); ctx.stroke();
        ctx.setLineDash([]);
      }
    }
    // ball
    ctx.fillStyle = 'white';
    ctx.beginPath(); ctx.arc(bxp, bdraw, 8, 0, Math.PI*2); ctx.fill();
    ctx.strokeStyle = '#444'; ctx.lineWidth = 1.2; ctx.stroke();
    // pentagon
    ctx.fillStyle = '#333';
    ctx.beginPath();
    for (let i = 0; i < 5; i++) {
      const a = i*Math.PI*2/5 - Math.PI/2;
      const mx = bxp+Math.cos(a)*3.5, my = bdraw+Math.sin(a)*3.5;
      if (i===0) ctx.moveTo(mx,my); else ctx.lineTo(mx,my);
    }
    ctx.closePath(); ctx.fill();
    // height label
    if (iz > 0.3) {
      ctx.fillStyle = '#fff';
      ctx.font = 'bold 10px sans-serif';
      ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
      ctx.fillText(iz.toFixed(1)+'m', bxp+12, bdraw);
    }
  }

  // particles
  if (fx && fx.particles) {
    if (f.event === 'GOAL!' && prev && prev.event !== 'GOAL!') {
      const gx = tx(pw), gy = ty((glo+ghi)/2);
      fx.particles.emit(gx, gy, 40, '#ffd700', 150, 1.2);
      fx.particles.emit(gx, gy, 20, '#fff', 100, 0.8);
    }
  }

  if (f.event === 'GOAL!') drawFlash(ctx, W, H, '#ffd700', 0.1 + 0.05*Math.sin(time*8));
}
"""


def frame_to_canvas(env, frame: dict) -> dict:
    is_fk = isinstance(env, FreeKickEnv)
    px, py = frame.get("x", 0), frame.get("y", 0)
    ball = frame.get("ball", (px, py))
    keeper = frame.get("keeper")
    defenders = frame.get("defenders", [])
    return {
        "pitch_w": env.W, "pitch_h": env.H,
        "goal_lo": env.goal_lo, "goal_hi": env.goal_hi,
        "player": [px, py],
        "vx": frame.get("vx", 0), "vy": frame.get("vy", 0),
        "ball": list(ball), "ball_z": frame.get("ball_z", 0.0),
        "ball_in_flight": frame.get("ball_in_flight", False),
        "keeper": list(keeper) if keeper else None,
        "keeper_reach": getattr(env, "keeper_reach", 0.7),
        "defenders": [list(d) if isinstance(d, (list, tuple)) else d for d in defenders],
        "shoot_x": getattr(env, "shoot_x", None) if not is_fk else None,
        "step": frame.get("step", 0),
        "cum_reward": frame.get("cum_reward"),
        "event": frame.get("event", "In play"),
    }


def _expand_flight_frames(frames: list) -> list:
    out = []
    for f in frames:
        flight = f.get("flight")
        if flight:
            last = len(flight) - 1
            for i, scene in enumerate(flight):
                out.append({
                    "x": f["x"], "y": f["y"], "vx": 0, "vy": 0,
                    "ball": list(scene["ball"]), "ball_z": scene.get("ball_z", 0.0),
                    "ball_in_flight": True,
                    "keeper": list(scene["keeper"]) if scene.get("keeper") else None,
                    "defenders": f.get("defenders", []),
                    "event": f.get("event", "shot!") if i == last else "Ball in flight!",
                    "step": f.get("step"), "cum_reward": f.get("cum_reward"),
                })
        else:
            out.append(f)
    return out


def draw_football_canvas(env, frame: dict, title: str = "") -> None:
    data = frame_to_canvas(env, frame)
    if title: data["event"] = title
    render_canvas(data, DRAW_JS, width=540, height=568, title=data["event"], bg_color="#2e7d32")


def replay_football_canvas(env, frames: list, key: str = "fb_replay",
                           expand_flight: bool = True) -> None:
    if expand_flight:
        frames = _expand_flight_frames(frames)
    data = [frame_to_canvas(env, f) for f in frames]
    render_canvas_replay(data, DRAW_JS, width=540, height=568, bg_color="#2e7d32",
                         fps=12, interpolation_steps=3,
                         trail_color="rgba(30,136,229,0.3)", trail_len=20, key=key)
