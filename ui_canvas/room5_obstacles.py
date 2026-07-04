"""Canvas renderer for Room 5 — Chicken road crossing with rich car visuals."""

from __future__ import annotations
import math
from rl.envs.obstacles import ObstacleEnv
from ui_canvas.canvas_core import render_canvas, render_canvas_replay

DRAW_JS = r"""
function drawFrame(ctx, W, H, f, prev, t, fx) {
  const pw=f.pitch_w, ph=f.pitch_h;
  const oY=30;
  const pH=H-oY;
  const sx=W/pw, sy=pH/ph;
  t=(t==null)?1:t;
  const time=performance.now()*0.001;

  function tx(v){return v*sx;}
  function ty(v){return oY+pH-v*sy;}

  const roadMin=tx(f.road_x_min), roadMax=tx(f.road_x_max);
  const goalX=tx(f.goal_x);

  // ── Left sidewalk (START) ──────────────────────────────────────────────
  const sg=ctx.createLinearGradient(0,oY,roadMin,oY);
  sg.addColorStop(0,'#c8860a');sg.addColorStop(1,'#d4a017');
  ctx.fillStyle=sg;ctx.fillRect(0,oY,roadMin,pH);
  // sidewalk texture
  ctx.strokeStyle='rgba(160,120,10,0.2)';ctx.lineWidth=0.6;
  for(let sy2=oY+12;sy2<H;sy2+=18){
    ctx.beginPath();ctx.moveTo(0,sy2);ctx.lineTo(roadMin,sy2);ctx.stroke();
  }
  ctx.fillStyle='rgba(0,0,0,0.1)';ctx.fillRect(roadMin-2,oY,2,pH);

  // ── Right sidewalk (GOAL) ──────────────────────────────────────────────
  const gg2=ctx.createLinearGradient(goalX,oY,W,oY);
  gg2.addColorStop(0,'#55a85e');gg2.addColorStop(1,'#66bb6a');
  ctx.fillStyle=gg2;ctx.fillRect(goalX,oY,W-goalX,pH);
  // grass texture
  ctx.strokeStyle='rgba(40,120,50,0.2)';ctx.lineWidth=0.6;
  for(let sy2=oY+12;sy2<H;sy2+=18){
    ctx.beginPath();ctx.moveTo(goalX,sy2);ctx.lineTo(W,sy2);ctx.stroke();
  }
  ctx.fillStyle='rgba(0,0,0,0.1)';ctx.fillRect(goalX,oY,2,pH);

  // ── Road ──────────────────────────────────────────────────────────────
  const rg=ctx.createLinearGradient(roadMin,oY,roadMax,oY);
  rg.addColorStop(0,'#2a2f35');rg.addColorStop(0.5,'#37474f');rg.addColorStop(1,'#2a2f35');
  ctx.fillStyle=rg;ctx.fillRect(roadMin,oY,roadMax-roadMin,pH);
  // road grain
  ctx.fillStyle='rgba(255,255,255,0.008)';
  for(let i=0;i<40;i++){
    ctx.fillRect(roadMin+Math.random()*(roadMax-roadMin),oY+Math.random()*pH,1,Math.random()*8+2);
  }

  // ── Lane markings ─────────────────────────────────────────────────────
  const lanes=f.lane_xs||[];
  for(let i=1;i<lanes.length;i++){
    const midX=tx((lanes[i]+lanes[i-1])/2);
    const isCenter=(i===Math.floor(lanes.length/2));
    if(isCenter){
      // solid yellow center line
      ctx.strokeStyle='#fdd835';ctx.lineWidth=3;
      ctx.setLineDash([]);
      ctx.beginPath();ctx.moveTo(midX,oY);ctx.lineTo(midX,H);ctx.stroke();
    } else {
      // dashed white
      ctx.strokeStyle='rgba(255,255,255,0.6)';ctx.lineWidth=1.5;
      ctx.setLineDash([16,12]);
      ctx.beginPath();ctx.moveTo(midX,oY);ctx.lineTo(midX,H);ctx.stroke();
      ctx.setLineDash([]);
    }
  }

  // ── START / GOAL labels ────────────────────────────────────────────────
  ctx.save();
  ctx.fillStyle='rgba(100,60,0,0.8)';ctx.font='bold 11px "Segoe UI",sans-serif';
  ctx.textAlign='center';
  ctx.translate(roadMin/2,oY+pH/2);ctx.rotate(-Math.PI/2);
  ctx.fillText('START',0,0);ctx.restore();

  ctx.save();
  ctx.fillStyle='rgba(20,80,30,0.9)';ctx.font='bold 11px "Segoe UI",sans-serif';
  ctx.textAlign='center';
  ctx.translate(goalX+(W-goalX)/2,oY+pH/2);ctx.rotate(-Math.PI/2);
  ctx.fillText('GOAL',0,0);ctx.restore();

  // ── Sensor range ──────────────────────────────────────────────────────
  const pp=f.player, ppP=(prev&&prev.player)||pp;
  if(pp){
    const pcx=tx(lerp(ppP[0],pp[0],t)), pcy=ty(lerp(ppP[1],pp[1],t));
    const sr=f.sensor_range*sx;
    const srGrad=ctx.createRadialGradient(pcx,pcy,sr*0.6,pcx,pcy,sr);
    srGrad.addColorStop(0,'rgba(56,189,248,0)');srGrad.addColorStop(1,'rgba(56,189,248,0.1)');
    ctx.fillStyle=srGrad;ctx.beginPath();ctx.arc(pcx,pcy,sr,0,Math.PI*2);ctx.fill();
    ctx.strokeStyle='rgba(56,189,248,0.55)';ctx.lineWidth=1.8;
    ctx.setLineDash([7,6]);ctx.beginPath();ctx.arc(pcx,pcy,sr,0,Math.PI*2);ctx.stroke();
    ctx.setLineDash([]);
  }

  // ── Cars ──────────────────────────────────────────────────────────────
  (f.cars||[]).forEach(car=>{
    const cx2=tx(car.x), cy2=ty(car.y);
    const cw2=car.width*sx, ch2=car.height*sy;
    const hl=car.highlighted;
    const dir=car.direction>=0?-1:1;
    const color=hl?'#fb923c':(car.color||'#ef4444');
    // shadow
    ctx.fillStyle='rgba(0,0,0,0.2)';
    ctx.beginPath();ctx.roundRect(cx2-cw2/2+4,cy2-ch2/2+4,cw2,ch2,6);ctx.fill();
    // body with gradient
    const cBodyG=ctx.createLinearGradient(cx2-cw2/2,cy2,cx2+cw2/2,cy2);
    cBodyG.addColorStop(0,hl?'#fb923c':color);
    cBodyG.addColorStop(0.4,hl?'#fed7aa':color);
    cBodyG.addColorStop(1,hl?'#f97316':color);
    ctx.fillStyle=cBodyG;
    ctx.beginPath();ctx.roundRect(cx2-cw2/2,cy2-ch2/2,cw2,ch2,6);ctx.fill();
    ctx.strokeStyle=hl?'#fde68a':'#263238';ctx.lineWidth=hl?2:1.2;ctx.stroke();
    // windshield
    const wsY=cy2+dir*ch2*0.14, wsH=dir*ch2*0.2;
    ctx.fillStyle='rgba(179,229,252,0.82)';
    ctx.beginPath();ctx.roundRect(cx2-cw2*0.27,Math.min(wsY,wsY+wsH),cw2*0.54,Math.abs(wsH),2);ctx.fill();
    ctx.strokeStyle='rgba(3,105,161,0.5)';ctx.lineWidth=0.7;ctx.stroke();
    // roof section
    ctx.fillStyle=hl?'rgba(250,150,50,0.4)':'rgba(0,0,0,0.2)';
    ctx.beginPath();ctx.roundRect(cx2-cw2*0.2,Math.min(wsY,wsY+wsH)+wsH,cw2*0.4,wsH*(-0.6),2);ctx.fill();
    // headlights
    const hlY=cy2+dir*ch2*0.44;
    ctx.fillStyle='#fff9c4';
    ctx.beginPath();ctx.arc(cx2-cw2*0.28,hlY,3.5,0,Math.PI*2);ctx.fill();
    ctx.beginPath();ctx.arc(cx2+cw2*0.28,hlY,3.5,0,Math.PI*2);ctx.fill();
    // headlight glow
    if(hl){
      const hlGrad=ctx.createRadialGradient(cx2-cw2*0.28,hlY,0,cx2-cw2*0.28,hlY,20);
      hlGrad.addColorStop(0,'rgba(255,249,196,0.4)');hlGrad.addColorStop(1,'rgba(255,249,196,0)');
      ctx.fillStyle=hlGrad;ctx.beginPath();ctx.arc(cx2-cw2*0.28,hlY,20,0,Math.PI*2);ctx.fill();
      const hlGrad2=ctx.createRadialGradient(cx2+cw2*0.28,hlY,0,cx2+cw2*0.28,hlY,20);
      hlGrad2.addColorStop(0,'rgba(255,249,196,0.4)');hlGrad2.addColorStop(1,'rgba(255,249,196,0)');
      ctx.fillStyle=hlGrad2;ctx.beginPath();ctx.arc(cx2+cw2*0.28,hlY,20,0,Math.PI*2);ctx.fill();
      // beam cone
      ctx.fillStyle='rgba(255,245,157,0.07)';
      ctx.beginPath();
      ctx.moveTo(cx2-cw2*0.28,hlY);ctx.lineTo(cx2-cw2*1.0,hlY+dir*ch2*0.9);
      ctx.lineTo(cx2+cw2*1.0,hlY+dir*ch2*0.9);ctx.lineTo(cx2+cw2*0.28,hlY);
      ctx.closePath();ctx.fill();
    }
    // wheels
    ctx.fillStyle='#1a1a1a';
    const wxo=cw2*0.38, wyo=ch2*0.36, wr=Math.min(cw2,ch2)*0.11;
    [[cx2-wxo,cy2-wyo],[cx2+wxo,cy2-wyo],[cx2-wxo,cy2+wyo],[cx2+wxo,cy2+wyo]].forEach(([wx,wy])=>{
      ctx.beginPath();ctx.arc(wx,wy,wr,0,Math.PI*2);ctx.fill();
      ctx.fillStyle='#444';ctx.beginPath();ctx.arc(wx,wy,wr*0.5,0,Math.PI*2);ctx.fill();
      ctx.fillStyle='#1a1a1a';
    });
  });

  // ── Chicken ────────────────────────────────────────────────────────────
  if(pp){
    const pcx=tx(lerp(ppP[0],pp[0],t)), pcy=ty(lerp(ppP[1],pp[1],t));
    if(fx&&fx.trail){fx.trail.add(pcx,pcy);fx.trail.draw(ctx);}
    const facing=(f.vx||0)<0?-1:1;
    const walk=Math.sin(time*7)*3;
    // shadow
    ctx.fillStyle='rgba(0,0,0,0.2)';
    ctx.beginPath();ctx.ellipse(pcx,pcy+17,16,5,0,0,Math.PI*2);ctx.fill();
    // body
    const bGrad=ctx.createRadialGradient(pcx-5*facing,pcy-3,0,pcx,pcy,18);
    bGrad.addColorStop(0,'#fffdf5');bGrad.addColorStop(0.5,'#fff7ed');bGrad.addColorStop(1,'#f5e6d0');
    ctx.fillStyle=bGrad;
    ctx.beginPath();ctx.ellipse(pcx-2*facing,pcy,18,14,0.1*facing,0,Math.PI*2);ctx.fill();
    ctx.strokeStyle='#c4956a';ctx.lineWidth=1.3;ctx.stroke();
    // wing detail
    ctx.strokeStyle='rgba(180,140,90,0.4)';ctx.lineWidth=1;
    ctx.beginPath();ctx.ellipse(pcx-4*facing,pcy+2,10,7,0.2*facing,0,Math.PI*2);ctx.stroke();
    // neck
    ctx.fillStyle='#fff7ed';
    ctx.beginPath();ctx.ellipse(pcx+8*facing,pcy-8,7,11,0.2*facing,0,Math.PI*2);ctx.fill();
    ctx.strokeStyle='#c4956a';ctx.lineWidth=1;ctx.stroke();
    // head
    const hGrad=ctx.createRadialGradient(pcx+9*facing-2,pcy-15,0,pcx+9*facing,pcy-16,9);
    hGrad.addColorStop(0,'#fffdf5');hGrad.addColorStop(1,'#f5e6d0');
    ctx.fillStyle=hGrad;
    ctx.beginPath();ctx.arc(pcx+9*facing,pcy-16,9,0,Math.PI*2);ctx.fill();
    ctx.strokeStyle='#c4956a';ctx.lineWidth=1;ctx.stroke();
    // beak
    ctx.fillStyle='#f97316';
    ctx.beginPath();
    ctx.moveTo(pcx+17*facing,pcy-17);ctx.lineTo(pcx+25*facing,pcy-14);
    ctx.lineTo(pcx+25*facing,pcy-20);ctx.closePath();ctx.fill();
    ctx.strokeStyle='#c2410c';ctx.lineWidth=0.7;ctx.stroke();
    // nostril
    ctx.fillStyle='#c2410c';ctx.beginPath();ctx.arc(pcx+19*facing,pcy-17,1.2,0,Math.PI*2);ctx.fill();
    // comb
    ctx.fillStyle='#ef4444';
    ctx.beginPath();
    ctx.moveTo(pcx+6*facing,pcy-24);ctx.lineTo(pcx+9*facing,pcy-32);
    ctx.lineTo(pcx+12*facing,pcy-24);ctx.closePath();ctx.fill();
    ctx.beginPath();
    ctx.moveTo(pcx+3*facing,pcy-23);ctx.lineTo(pcx+6*facing,pcy-29);
    ctx.lineTo(pcx+9*facing,pcy-23);ctx.closePath();ctx.fill();
    // eye
    ctx.fillStyle='#1a1a1a';ctx.beginPath();ctx.arc(pcx+13*facing,pcy-17,2.5,0,Math.PI*2);ctx.fill();
    ctx.fillStyle='rgba(255,255,255,0.6)';ctx.beginPath();ctx.arc(pcx+14*facing,pcy-18,1,0,Math.PI*2);ctx.fill();
    // wattle
    ctx.fillStyle='#dc2626';ctx.beginPath();ctx.arc(pcx+16*facing,pcy-13,3,0,Math.PI);ctx.fill();
    // legs with walk animation
    ctx.strokeStyle='#f97316';ctx.lineWidth=2;ctx.lineCap='round';
    ctx.beginPath();ctx.moveTo(pcx-4,pcy+13);ctx.lineTo(pcx-5+walk,pcy+22);ctx.stroke();
    ctx.beginPath();ctx.moveTo(pcx-5+walk,pcy+22);ctx.lineTo(pcx-13+walk,pcy+22);ctx.stroke();
    ctx.beginPath();ctx.moveTo(pcx+4,pcy+13);ctx.lineTo(pcx+4-walk,pcy+22);ctx.stroke();
    ctx.beginPath();ctx.moveTo(pcx+4-walk,pcy+22);ctx.lineTo(pcx-4-walk,pcy+22);ctx.stroke();

    // particles
    if(fx&&fx.particles){
      if(f.event&&(f.event.indexOf('Crossed')>=0||f.event.indexOf('Safe')>=0))
        fx.particles.emit(pcx,pcy,32,'#66bb6a',130,1.1);
      if(f.event&&f.event.indexOf('hit')>=0)
        fx.particles.emit(pcx,pcy,26,'#ff1744',110,0.9);
    }
  }

  if(f.event&&f.event.indexOf('Crossed')>=0) drawFlash(ctx,W,H,'#4caf50',0.12);
  if(f.event&&f.event.indexOf('hit')>=0)     drawFlash(ctx,W,H,'#ff1744',0.18);
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
                         fps=12, trail_color="rgba(255,247,237,0.3)", trail_len=18, key=key)
