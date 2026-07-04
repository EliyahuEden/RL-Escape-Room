"""Canvas renderer for Room 4 — Football. Rich pitch, 3D ball arc, goal celebrations."""

from __future__ import annotations
from rl.envs.football import FootballEnv, FreeKickEnv
from ui_canvas.canvas_core import render_canvas, render_canvas_replay

DRAW_JS = r"""
function drawFrame(ctx, W, H, f, prev, t, fx) {
  const pw = f.pitch_w||10, ph = f.pitch_h||10;
  const oY = 30;
  const pH = H - oY;
  const sx = W/pw, sy = pH/ph;
  t = (t==null)?1:t;
  const time = performance.now()*0.001;

  function tx(v){return v*sx;}
  function ty(v){return oY+pH-v*sy;}

  // ── Grass with stripes ─────────────────────────────────────────────────
  const stripes = 14;
  for(let i=0;i<stripes;i++){
    const g=ctx.createLinearGradient(0,oY+i*pH/stripes,0,oY+(i+1)*pH/stripes);
    if(i%2===0){g.addColorStop(0,'#2e7d32');g.addColorStop(1,'#277030');}
    else{g.addColorStop(0,'#338a36');g.addColorStop(1,'#2e8032');}
    ctx.fillStyle=g;ctx.fillRect(0,oY+i*pH/stripes,W,pH/stripes+1);
  }

  // ── Pitch markings ─────────────────────────────────────────────────────
  ctx.strokeStyle='rgba(255,255,255,0.35)';ctx.lineWidth=1.5;
  // halfway line
  ctx.beginPath();ctx.moveTo(tx(pw/2),oY);ctx.lineTo(tx(pw/2),H);ctx.stroke();
  // center circle
  ctx.beginPath();ctx.arc(tx(pw/2),ty(ph/2),tx(2.0),0,Math.PI*2);ctx.stroke();
  ctx.fillStyle='rgba(255,255,255,0.08)';
  ctx.beginPath();ctx.arc(tx(pw/2),ty(ph/2),tx(2.0),0,Math.PI*2);ctx.fill();
  // center spot
  ctx.fillStyle='rgba(255,255,255,0.6)';
  ctx.beginPath();ctx.arc(tx(pw/2),ty(ph/2),3,0,Math.PI*2);ctx.fill();
  // penalty area
  if(f.shoot_x!=null){
    ctx.fillStyle='rgba(255,255,255,0.04)';
    ctx.fillRect(tx(f.shoot_x),oY,W-tx(f.shoot_x),pH);
    ctx.strokeStyle='rgba(255,255,255,0.3)';ctx.lineWidth=1.2;
    ctx.setLineDash([7,6]);
    ctx.beginPath();ctx.moveTo(tx(f.shoot_x),oY);ctx.lineTo(tx(f.shoot_x),H);ctx.stroke();
    ctx.setLineDash([]);
  }

  // ── Goal ───────────────────────────────────────────────────────────────
  const glo=f.goal_lo||3.5, ghi=f.goal_hi||6.5;
  const gTop=ty(ghi), gBot=ty(glo), gH=gBot-gTop;
  // goal net backing
  ctx.fillStyle='rgba(220,220,220,0.08)';
  ctx.fillRect(tx(pw)-2,gTop,14,gH);
  // net lines
  ctx.strokeStyle='rgba(200,200,200,0.25)';ctx.lineWidth=0.6;
  for(let ny=0;ny<=gH;ny+=7){
    ctx.beginPath();ctx.moveTo(tx(pw)-2,gTop+ny);ctx.lineTo(tx(pw)+12,gTop+ny);ctx.stroke();
  }
  for(let nx=0;nx<=12;nx+=7){
    ctx.beginPath();ctx.moveTo(tx(pw)-2+nx,gTop);ctx.lineTo(tx(pw)-2+nx,gBot);ctx.stroke();
  }
  // goal posts
  ctx.strokeStyle='#e0e0e0';ctx.lineWidth=4;ctx.lineCap='round';
  ctx.beginPath();ctx.moveTo(tx(pw),gTop);ctx.lineTo(tx(pw),gBot);ctx.stroke();
  ctx.strokeStyle='white';ctx.lineWidth=3;
  ctx.beginPath();ctx.moveTo(tx(pw),gTop);ctx.lineTo(tx(pw)+8,gTop);ctx.stroke();
  ctx.beginPath();ctx.moveTo(tx(pw),gBot);ctx.lineTo(tx(pw)+8,gBot);ctx.stroke();
  // goal flash on score
  if(f.event==='GOAL!'){
    const gPulse=0.12+0.08*Math.sin(time*10);
    ctx.fillStyle='rgba(255,215,0,'+gPulse+')';ctx.fillRect(tx(pw)-4,gTop,20,gH);
  }

  // ── Defenders ──────────────────────────────────────────────────────────
  const defs=f.defenders||[], prevDefs=(prev&&prev.defenders)||defs;
  for(let di=0;di<defs.length;di++){
    const d=defs[di], pd=di<prevDefs.length?prevDefs[di]:d;
    const dx2=tx(lerp(pd[0],d[0],t)), dy2=ty(lerp(pd[1],d[1],t));
    // shadow
    ctx.fillStyle='rgba(0,0,0,0.25)';
    ctx.beginPath();ctx.ellipse(dx2,dy2+13,14,5,0,0,Math.PI*2);ctx.fill();
    // legs
    ctx.strokeStyle='#880e4f';ctx.lineWidth=5;ctx.lineCap='round';
    ctx.beginPath();ctx.moveTo(dx2-5,dy2+2);ctx.lineTo(dx2-5,dy2+18);ctx.stroke();
    ctx.beginPath();ctx.moveTo(dx2+5,dy2+2);ctx.lineTo(dx2+5,dy2+18);ctx.stroke();
    // body (jersey)
    const jg=ctx.createRadialGradient(dx2-4,dy2-4,0,dx2,dy2,16);
    jg.addColorStop(0,'#f44336');jg.addColorStop(1,'#b71c1c');
    ctx.fillStyle=jg;ctx.beginPath();ctx.arc(dx2,dy2,14,0,Math.PI*2);ctx.fill();
    ctx.strokeStyle='#7f0000';ctx.lineWidth=1.5;ctx.stroke();
    // head
    ctx.fillStyle='#ffcc80';ctx.beginPath();ctx.arc(dx2,dy2-16,7,0,Math.PI*2);ctx.fill();
    // number
    ctx.fillStyle='white';ctx.font='bold 9px sans-serif';
    ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText(''+(di+2),dx2,dy2+1);
  }

  // ── Goalkeeper ─────────────────────────────────────────────────────────
  if(f.keeper){
    const kp=f.keeper, kpP=(prev&&prev.keeper)||kp;
    const kx2=tx(lerp(kpP[0],kp[0],t)), ky2=ty(lerp(kpP[1],kp[1],t));
    const kr=(f.keeper_reach||0.7)*sy;
    // save zone indicator
    ctx.fillStyle='rgba(255,235,59,0.08)';
    ctx.fillRect(kx2-12,ky2-kr,24,kr*2);
    ctx.strokeStyle='rgba(255,235,59,0.3)';ctx.lineWidth=1;
    ctx.setLineDash([4,4]);
    ctx.strokeRect(kx2-12,ky2-kr,24,kr*2);
    ctx.setLineDash([]);
    // shadow
    ctx.fillStyle='rgba(0,0,0,0.25)';
    ctx.beginPath();ctx.ellipse(kx2,ky2+13,14,5,0,0,Math.PI*2);ctx.fill();
    // body
    const kg=ctx.createRadialGradient(kx2-4,ky2-4,0,kx2,ky2,16);
    kg.addColorStop(0,'#fff176');kg.addColorStop(1,'#f9a825');
    ctx.fillStyle=kg;ctx.beginPath();ctx.arc(kx2,ky2,14,0,Math.PI*2);ctx.fill();
    ctx.strokeStyle='#e65100';ctx.lineWidth=1.5;ctx.stroke();
    // head
    ctx.fillStyle='#ffcc80';ctx.beginPath();ctx.arc(kx2,ky2-16,7,0,Math.PI*2);ctx.fill();
    // gloves
    ctx.fillStyle='#333';
    ctx.beginPath();ctx.arc(kx2,ky2-kr,6,0,Math.PI*2);ctx.fill();
    ctx.beginPath();ctx.arc(kx2,ky2+kr,6,0,Math.PI*2);ctx.fill();
    ctx.fillStyle='#777';
    ctx.beginPath();ctx.arc(kx2,ky2-kr,3.5,0,Math.PI*2);ctx.fill();
    ctx.beginPath();ctx.arc(kx2,ky2+kr,3.5,0,Math.PI*2);ctx.fill();
    ctx.fillStyle='#333';ctx.font='bold 8px sans-serif';
    ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText('GK',kx2,ky2+1);
  }

  // ── Player ─────────────────────────────────────────────────────────────
  if(f.player){
    const pp=f.player, ppP=(prev&&prev.player)||pp;
    const ppx=tx(lerp(ppP[0],pp[0],t)), ppy=ty(lerp(ppP[1],pp[1],t));
    if(fx&&fx.trail&&!f.ball_in_flight){fx.trail.add(ppx,ppy);fx.trail.draw(ctx);}
    // shadow
    ctx.fillStyle='rgba(0,0,0,0.25)';
    ctx.beginPath();ctx.ellipse(ppx,ppy+13,14,5,0,0,Math.PI*2);ctx.fill();
    // legs
    ctx.strokeStyle='#003d7a';ctx.lineWidth=5;ctx.lineCap='round';
    ctx.beginPath();ctx.moveTo(ppx-5,ppy+2);ctx.lineTo(ppx-5,ppy+18);ctx.stroke();
    ctx.beginPath();ctx.moveTo(ppx+5,ppy+2);ctx.lineTo(ppx+5,ppy+18);ctx.stroke();
    // body
    const pg=ctx.createRadialGradient(ppx-4,ppy-4,0,ppx,ppy,16);
    pg.addColorStop(0,'#42a5f5');pg.addColorStop(1,'#1565c0');
    ctx.fillStyle=pg;ctx.beginPath();ctx.arc(ppx,ppy,14,0,Math.PI*2);ctx.fill();
    ctx.strokeStyle='white';ctx.lineWidth=1.5;ctx.stroke();
    // head
    ctx.fillStyle='#ffcc80';ctx.beginPath();ctx.arc(ppx,ppy-16,7,0,Math.PI*2);ctx.fill();
    // hair
    ctx.fillStyle='#5d4037';ctx.beginPath();ctx.arc(ppx,ppy-19,5,Math.PI,0);ctx.fill();
    // number
    ctx.fillStyle='white';ctx.font='bold 9px sans-serif';
    ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText('10',ppx,ppy+1);
  }

  // ── Ball with 3D height ─────────────────────────────────────────────────
  if(f.ball){
    const bp=f.ball, bpP=(prev&&prev.ball)||bp;
    const bxp=tx(lerp(bpP[0],bp[0],t)), byp=ty(lerp(bpP[1],bp[1],t));
    const bz=f.ball_z||0, prevBz=(prev&&prev.ball_z)||0;
    const iz=lerp(prevBz,bz,t);
    const lift=iz*22;
    const bdraw=byp-lift;

    if(f.ball_in_flight&&fx&&fx.trail){fx.trail.add(bxp,bdraw);fx.trail.draw(ctx);}

    // ground shadow (shrinks as ball rises)
    if(iz>0.05){
      const shScale=Math.max(0.2,1-iz*0.07);
      ctx.fillStyle='rgba(0,0,0,'+Math.min(0.35,0.12+iz*0.015)+')';
      ctx.beginPath();ctx.ellipse(bxp,byp,11*shScale,4*shScale,0,0,Math.PI*2);ctx.fill();
      // dotted height connector
      if(lift>6){
        ctx.strokeStyle='rgba(255,255,255,0.3)';ctx.lineWidth=0.8;
        ctx.setLineDash([2,3]);
        ctx.beginPath();ctx.moveTo(bxp,byp-1);ctx.lineTo(bxp,bdraw+9);ctx.stroke();
        ctx.setLineDash([]);
      }
    }
    // ball body
    const bGrad=ctx.createRadialGradient(bxp-3,bdraw-3,0,bxp,bdraw,9);
    bGrad.addColorStop(0,'#ffffff');bGrad.addColorStop(0.5,'#f5f5f5');bGrad.addColorStop(1,'#e0e0e0');
    ctx.fillStyle=bGrad;ctx.beginPath();ctx.arc(bxp,bdraw,9,0,Math.PI*2);ctx.fill();
    ctx.strokeStyle='#555';ctx.lineWidth=1;ctx.stroke();
    // hexagon panels
    ctx.fillStyle='rgba(50,50,50,0.6)';
    for(let i=0;i<5;i++){
      const a=i*Math.PI*2/5-Math.PI/2+time*(f.ball_in_flight?3:0);
      const mx=bxp+Math.cos(a)*3.8, my=bdraw+Math.sin(a)*3.8;
      ctx.beginPath();ctx.arc(mx,my,1.8,0,Math.PI*2);ctx.fill();
    }
    // center panel
    ctx.beginPath();ctx.arc(bxp,bdraw,1.8,0,Math.PI*2);ctx.fill();
    // height label
    if(iz>0.4){
      ctx.fillStyle='rgba(255,255,255,0.9)';
      ctx.font='bold 9px sans-serif';
      ctx.textAlign='left';ctx.textBaseline='middle';
      ctx.fillText(iz.toFixed(1)+'m',bxp+13,bdraw);
    }
  }

  // ── Goal celebration ───────────────────────────────────────────────────
  if(fx&&fx.particles&&f.event==='GOAL!'&&prev&&prev.event!=='GOAL!'){
    const gx=tx(pw), gy=ty((glo+ghi)/2);
    fx.particles.emit(gx,gy,45,'#ffd700',170,1.3);
    fx.particles.emit(gx,gy,20,'#fff',110,0.9);
    fx.particles.emit(gx,gy,15,'#ff5722',130,1.0);
  }
  if(f.event==='GOAL!'){
    drawFlash(ctx,W,H,'#ffd700',0.08+0.05*Math.sin(time*10));
  }
  if(f.event&&f.event.indexOf('SAVE')>=0) drawFlash(ctx,W,H,'#ff1744',0.12);
}
"""


def frame_to_canvas(env, frame: dict) -> dict:
    is_fk  = isinstance(env, FreeKickEnv)
    px, py = frame.get("x", 0), frame.get("y", 0)
    ball   = frame.get("ball", (px, py))
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
                    "event": f.get("event", "Shot!") if i == last else "Ball in flight!",
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
                         fps=12, trail_color="rgba(30,136,229,0.35)", trail_len=20, key=key)
