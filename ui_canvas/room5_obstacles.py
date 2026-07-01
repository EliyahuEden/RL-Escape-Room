"""Canvas renderer for Room 5 — Chicken crossing with headlights, sensor glow."""

from __future__ import annotations
import math
from rl.envs.obstacles import ObstacleEnv
from ui_canvas.canvas_core import render_canvas, render_canvas_replay

DRAW_JS = """
function drawFrame(ctx, W, H, f, prev, t, fx) {
  const pw = f.pitch_w, ph = f.pitch_h;
  const oY = 28;
  const pH = H - oY;
  const sx = W / pw, sy = pH / ph;
  function tx(v) { return v * sx; }
  function ty(v) { return oY + pH - v * sy; }
  t = t || 1;
  const time = (f.step || 0) * 0.15;

  // --- road background ---
  const roadMin = tx(f.road_x_min), roadMax = tx(f.road_x_max);
  // left sidewalk
  ctx.fillStyle = '#d4a017';
  ctx.fillRect(0, oY, roadMin, pH);
  // sidewalk texture
  ctx.strokeStyle = 'rgba(180,140,20,0.3)';
  ctx.lineWidth = 0.5;
  for (let sy2 = oY; sy2 < H; sy2 += 20) {
    ctx.beginPath(); ctx.moveTo(0, sy2); ctx.lineTo(roadMin, sy2); ctx.stroke();
  }
  // right sidewalk (goal)
  ctx.fillStyle = '#66bb6a';
  ctx.fillRect(tx(f.goal_x), oY, W - tx(f.goal_x), pH);
  // road
  ctx.fillStyle = '#37474f';
  ctx.fillRect(roadMin, oY, roadMax - roadMin, pH);

  // lane markings
  const lanes = f.lane_xs || [];
  for (let i = 1; i < lanes.length; i++) {
    const midX = tx((lanes[i] + lanes[i-1]) / 2);
    const isCenter = i === Math.floor(lanes.length / 2);
    ctx.strokeStyle = isCenter ? '#fdd835' : 'rgba(255,255,255,0.6)';
    ctx.lineWidth = isCenter ? 3 : 1.5;
    ctx.setLineDash(isCenter ? [] : [14, 10]);
    ctx.beginPath(); ctx.moveTo(midX, oY); ctx.lineTo(midX, H); ctx.stroke();
    ctx.setLineDash([]);
  }

  // labels
  ctx.save();
  ctx.fillStyle = '#7c2d12'; ctx.font = 'bold 11px sans-serif'; ctx.textAlign = 'center';
  ctx.translate(tx(f.road_x_min/2), oY + pH/2);
  ctx.rotate(-Math.PI/2); ctx.fillText('START', 0, 0); ctx.restore();
  ctx.save();
  ctx.fillStyle = '#1b5e20'; ctx.font = 'bold 11px sans-serif'; ctx.textAlign = 'center';
  ctx.translate(tx(f.goal_x + (pw - f.goal_x)/2), oY + pH/2);
  ctx.rotate(-Math.PI/2); ctx.fillText('GOAL', 0, 0); ctx.restore();

  // --- sensor range circle ---
  const pp = f.player;
  const ppPrev = (prev && prev.player) || pp;
  if (pp) {
    const pcx = tx(lerp(ppPrev[0],pp[0],t)), pcy = ty(lerp(ppPrev[1],pp[1],t));
    const sr = f.sensor_range * sx;
    // glow
    const grd = ctx.createRadialGradient(pcx, pcy, sr*0.7, pcx, pcy, sr);
    grd.addColorStop(0, 'rgba(56,189,248,0)');
    grd.addColorStop(1, 'rgba(56,189,248,0.12)');
    ctx.fillStyle = grd;
    ctx.beginPath(); ctx.arc(pcx, pcy, sr, 0, Math.PI*2); ctx.fill();
    ctx.strokeStyle = 'rgba(56,189,248,0.5)';
    ctx.lineWidth = 1.8;
    ctx.setLineDash([6,6]);
    ctx.beginPath(); ctx.arc(pcx, pcy, sr, 0, Math.PI*2); ctx.stroke();
    ctx.setLineDash([]);
  }

  // --- cars ---
  (f.cars || []).forEach(function(car) {
    const cpPrev = car; // no per-car interp for simplicity
    const cx2 = tx(car.x), cy2 = ty(car.y);
    const cw2 = car.width * sx, ch2 = car.height * sy;
    const hl = car.highlighted;
    const color = hl ? '#fb923c' : (car.color || '#ef4444');
    const dir = car.direction >= 0 ? -1 : 1;
    // shadow
    ctx.fillStyle = 'rgba(0,0,0,0.15)';
    ctx.beginPath(); ctx.roundRect(cx2-cw2/2+3, cy2-ch2/2+3, cw2, ch2, 5); ctx.fill();
    // body
    ctx.fillStyle = color;
    ctx.beginPath(); ctx.roundRect(cx2-cw2/2, cy2-ch2/2, cw2, ch2, 5); ctx.fill();
    ctx.strokeStyle = hl ? '#fde68a' : '#263238';
    ctx.lineWidth = hl ? 2.5 : 1;
    ctx.stroke();
    // windshield
    ctx.fillStyle = '#b3e5fc';
    ctx.fillRect(cx2-cw2*0.28, cy2+dir*ch2*0.15, cw2*0.56, ch2*0.18*dir);
    // headlights
    ctx.fillStyle = '#fff59d';
    const hlY = cy2 + dir*ch2*0.45;
    ctx.beginPath(); ctx.arc(cx2-cw2*0.3, hlY, 3, 0, Math.PI*2); ctx.fill();
    ctx.beginPath(); ctx.arc(cx2+cw2*0.3, hlY, 3, 0, Math.PI*2); ctx.fill();
    // headlight beam
    if (hl) {
      ctx.fillStyle = 'rgba(255,245,157,0.08)';
      ctx.beginPath();
      ctx.moveTo(cx2-cw2*0.3, hlY);
      ctx.lineTo(cx2-cw2*0.8, hlY+dir*ch2*0.8);
      ctx.lineTo(cx2+cw2*0.8, hlY+dir*ch2*0.8);
      ctx.lineTo(cx2+cw2*0.3, hlY);
      ctx.closePath(); ctx.fill();
    }
    // wheels
    ctx.fillStyle = '#111';
    const wxo = cw2*0.42, wyo = ch2*0.38;
    ctx.beginPath(); ctx.arc(cx2-wxo, cy2-wyo, 3.5, 0, Math.PI*2); ctx.fill();
    ctx.beginPath(); ctx.arc(cx2+wxo, cy2-wyo, 3.5, 0, Math.PI*2); ctx.fill();
    ctx.beginPath(); ctx.arc(cx2-wxo, cy2+wyo, 3.5, 0, Math.PI*2); ctx.fill();
    ctx.beginPath(); ctx.arc(cx2+wxo, cy2+wyo, 3.5, 0, Math.PI*2); ctx.fill();
  });

  // --- chicken with interpolation ---
  if (pp) {
    const pcx = tx(lerp(ppPrev[0],pp[0],t)), pcy = ty(lerp(ppPrev[1],pp[1],t));
    if (fx && fx.trail) { fx.trail.add(pcx, pcy); fx.trail.draw(ctx); }
    const facing = (f.vx || 0) < 0 ? -1 : 1;
    // body
    ctx.fillStyle = '#fff7ed';
    ctx.beginPath(); ctx.ellipse(pcx-2*facing, pcy, 17, 13, 0.12*facing, 0, Math.PI*2); ctx.fill();
    ctx.strokeStyle = '#9a6a38'; ctx.lineWidth = 1.2; ctx.stroke();
    // head
    ctx.fillStyle = '#fff7ed';
    ctx.beginPath(); ctx.arc(pcx+11*facing, pcy-5, 8, 0, Math.PI*2); ctx.fill();
    ctx.strokeStyle = '#9a6a38'; ctx.lineWidth = 0.8; ctx.stroke();
    // beak
    ctx.fillStyle = '#f97316';
    ctx.beginPath();
    ctx.moveTo(pcx+18*facing, pcy-5);
    ctx.lineTo(pcx+25*facing, pcy-2);
    ctx.lineTo(pcx+25*facing, pcy-8);
    ctx.closePath(); ctx.fill();
    // comb
    ctx.fillStyle = '#ef4444';
    ctx.beginPath();
    ctx.moveTo(pcx+7*facing, pcy-12);
    ctx.lineTo(pcx+11*facing, pcy-20);
    ctx.lineTo(pcx+15*facing, pcy-11);
    ctx.closePath(); ctx.fill();
    // eye
    ctx.fillStyle = '#111';
    ctx.beginPath(); ctx.arc(pcx+13*facing, pcy-6, 2, 0, Math.PI*2); ctx.fill();
    // legs
    ctx.strokeStyle = '#f97316'; ctx.lineWidth = 1.8;
    const legPhase = Math.sin(time*8) * 3;
    ctx.beginPath();
    ctx.moveTo(pcx-3, pcy+12); ctx.lineTo(pcx-5+legPhase, pcy+20);
    ctx.moveTo(pcx-5+legPhase, pcy+20); ctx.lineTo(pcx-12+legPhase, pcy+20);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(pcx+5, pcy+12); ctx.lineTo(pcx+3-legPhase, pcy+19);
    ctx.moveTo(pcx+3-legPhase, pcy+19); ctx.lineTo(pcx-4-legPhase, pcy+19);
    ctx.stroke();

    // particles
    if (fx && fx.particles) {
      if (f.event === 'Crossed!' || f.event === 'Safe!') {
        fx.particles.emit(pcx, pcy, 30, '#66bb6a', 120, 1.0);
      }
      if (f.event && f.event.indexOf('hit') >= 0) {
        fx.particles.emit(pcx, pcy, 25, '#ff1744', 100, 0.8);
      }
    }
  }

  if (f.event && f.event.indexOf('Crossed') >= 0) drawFlash(ctx, W, H, '#4caf50', 0.1);
  if (f.event && f.event.indexOf('hit') >= 0) drawFlash(ctx, W, H, '#ff1744', 0.15);
}
"""


def _rect_distance(px, py, x, y, w, h):
    cx = min(max(px, x - w/2), x + w/2)
    cy = min(max(py, y - h/2), y + h/2)
    return math.hypot(px - cx, py - cy)


def frame_to_canvas(env: ObstacleEnv, frame: dict) -> dict:
    px, py = frame.get("x", 0), frame.get("y", 0)
    rng = frame.get("sensor_range", getattr(env, "sensor_range", 3.0))
    cars = frame.get("cars")
    if cars is None:
        cars = [
            {"x": ox, "y": oy, "direction": 1, "speed": 0.0,
             "width": 0.55, "height": 1.05, "color": "#ef4444"}
            for ox, oy in frame.get("obstacles", [])
        ]
    car_data = []
    for car in cars:
        dist = _rect_distance(px, py, car["x"], car["y"],
                              car.get("width", 0.55), car.get("height", 1.05))
        car_data.append({
            "x": car["x"], "y": car["y"],
            "width": car.get("width", 0.55), "height": car.get("height", 1.05),
            "direction": car.get("direction", 1),
            "color": car.get("color", "#ef4444"),
            "highlighted": dist <= rng,
        })
    return {
        "pitch_w": env.W, "pitch_h": env.H,
        "player": [px, py],
        "vx": frame.get("vx", 0), "vy": frame.get("vy", 0),
        "sensor_range": rng,
        "road_x_min": frame.get("road_x_min", getattr(env, "road_x_min", 1.0)),
        "road_x_max": frame.get("road_x_max", getattr(env, "road_x_max", 9.0)),
        "goal_x": frame.get("goal_x", getattr(env, "goal_x", 9.0)),
        "lane_xs": frame.get("lane_xs", getattr(env, "lane_xs", [])),
        "cars": car_data,
        "event": frame.get("event", "Crossing..."),
        "step": frame.get("step", 0),
        "cum_reward": frame.get("cum_reward"),
    }


def draw_chicken_canvas(env: ObstacleEnv, frame: dict, title: str = "") -> None:
    data = frame_to_canvas(env, frame)
    if title: data["event"] = title
    render_canvas(data, DRAW_JS, width=540, height=568, title=data["event"], bg_color="#37474f")


def replay_chicken_canvas(env: ObstacleEnv, frames: list, key: str = "chk_replay") -> None:
    data = [frame_to_canvas(env, f) for f in frames]
    render_canvas_replay(data, DRAW_JS, width=540, height=568, bg_color="#37474f",
                         fps=12, interpolation_steps=3,
                         trail_color="rgba(255,247,237,0.25)", trail_len=15, key=key)
