"""Canvas renderer for Room 2 — Museum Heist. Rich tile textures, alarm glow."""

from __future__ import annotations
from rl.envs.museum import MuseumEnv
from ui_canvas.canvas_core import render_canvas, render_canvas_replay

DRAW_JS = r"""
function drawFrame(ctx, W, H, f, prev, t, fx) {
  const rows = f.rows, cols = f.cols;
  const cw = W / cols, ch = (H - 30) / rows;
  const oY = 30;
  t = (t == null) ? 1 : t;
  const alarm = f.alarm || false;
  const time = performance.now() * 0.001;

  const wallSet   = new Set(f.walls||[]);
  const camSet    = new Set(f.cameras||[]);
  const trapSet   = new Set(f.traps||[]);
  const slipSet   = new Set(f.slippery||[]);

  function parsePos(s) {
    if (!s) return null;
    const p=s.split(','); return [parseInt(p[0]),parseInt(p[1])];
  }
  function cellXY(r,c) { return [c*cw+cw/2, oY+r*ch+ch/2]; }

  // ── Background ─────────────────────────────────────────────────────────
  const bg = ctx.createLinearGradient(0,oY,0,H);
  if (alarm) { bg.addColorStop(0,'#1a0505'); bg.addColorStop(1,'#0d0000'); }
  else       { bg.addColorStop(0,'#12100c'); bg.addColorStop(1,'#0a0806'); }
  ctx.fillStyle=bg; ctx.fillRect(0,oY,W,H-oY);

  // ── Alarm pulse overlay ────────────────────────────────────────────────
  if (alarm) {
    const pulse = 0.07 + 0.05*Math.sin(time*7);
    drawFlash(ctx,W,H,'#cc0000',pulse);
  }

  const pos     = parsePos(f.pos);
  const prevPos = prev ? parsePos(prev.pos) : pos;

  // ── Tiles ──────────────────────────────────────────────────────────────
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const x=c*cw, y=oY+r*ch;
      const key=r+','+c;

      if (wallSet.has(key)) {
        // Stone brick wall
        const wg = ctx.createLinearGradient(x,y,x+cw,y+ch);
        wg.addColorStop(0,'#4a4035'); wg.addColorStop(1,'#2e2820');
        ctx.fillStyle=wg; ctx.fillRect(x,y,cw,ch);
        // mortar lines
        ctx.strokeStyle='rgba(20,15,10,0.7)'; ctx.lineWidth=1;
        ctx.beginPath();
        ctx.moveTo(x,y+ch*0.5); ctx.lineTo(x+cw,y+ch*0.5);
        ctx.moveTo(x+cw*0.5,y); ctx.lineTo(x+cw*0.5,y+ch*0.5);
        ctx.moveTo(x+cw*0.25,y+ch*0.5); ctx.lineTo(x+cw*0.25,y+ch);
        ctx.moveTo(x+cw*0.75,y+ch*0.5); ctx.lineTo(x+cw*0.75,y+ch);
        ctx.stroke();
        // top highlight
        ctx.fillStyle='rgba(255,255,220,0.06)';
        ctx.fillRect(x,y,cw,3);
      } else {
        // Marble floor
        const even = (r+c)%2===0;
        const mg = ctx.createLinearGradient(x,y,x+cw,y+ch);
        if (even) { mg.addColorStop(0,'#ddd5b5'); mg.addColorStop(1,'#c8bc98'); }
        else      { mg.addColorStop(0,'#c5b990'); mg.addColorStop(1,'#b0a678'); }
        ctx.fillStyle=mg; ctx.fillRect(x,y,cw,ch);
        // grout
        ctx.strokeStyle='rgba(140,128,100,0.5)';ctx.lineWidth=0.5;
        ctx.strokeRect(x+0.5,y+0.5,cw-1,ch-1);
        // marble vein
        ctx.strokeStyle='rgba(255,255,240,0.12)';ctx.lineWidth=0.8;
        ctx.beginPath();
        ctx.moveTo(x+cw*0.1,y+ch*0.7);ctx.bezierCurveTo(x+cw*0.4,y+ch*0.3,x+cw*0.6,y+ch*0.6,x+cw*0.9,y+ch*0.2);
        ctx.stroke();
      }

      // Slippery (ice)
      if (slipSet.has(key)) {
        ctx.fillStyle='rgba(180,230,255,0.18)';ctx.fillRect(x,y,cw,ch);
        ctx.strokeStyle='rgba(200,240,255,0.4)';ctx.lineWidth=0.8;
        ctx.beginPath();ctx.moveTo(x+cw*0.2,y+ch*0.5);ctx.lineTo(x+cw*0.8,y+ch*0.5);ctx.stroke();
        ctx.beginPath();ctx.moveTo(x+cw*0.5,y+ch*0.2);ctx.lineTo(x+cw*0.5,y+ch*0.8);ctx.stroke();
      }

      // Camera zone
      if (camSet.has(key)) {
        const ca = alarm ? 0.35+0.12*Math.sin(time*9) : 0.15;
        ctx.fillStyle='rgba(255,0,0,'+ca+')'; ctx.fillRect(x,y,cw,ch);
        // scan lines
        ctx.strokeStyle='rgba(255,80,80,'+(alarm?0.6:0.25)+')';ctx.lineWidth=0.7;
        for (let i=1;i<=4;i++){
          ctx.beginPath();ctx.moveTo(x,y+ch*i/5);ctx.lineTo(x+cw,y+ch*i/5);ctx.stroke();
        }
        // camera icon
        ctx.fillStyle= alarm?'#ff1744':'#ef9a9a';
        ctx.font='bold '+(cw*0.38)+'px sans-serif';
        ctx.textAlign='center';ctx.textBaseline='middle';
        ctx.fillText('📷',x+cw/2,y+ch*0.55);
      }

      // Laser trap
      if (trapSet.has(key)) {
        ctx.fillStyle='rgba(255,80,0,0.1)';ctx.fillRect(x,y,cw,ch);
        const beamAlpha = 0.6+0.3*Math.sin(time*5+c);
        ctx.strokeStyle='rgba(255,60,0,'+beamAlpha+')';ctx.lineWidth=1.5;
        ctx.setLineDash([5,3]);
        ctx.beginPath();ctx.moveTo(x+4,y+ch/2);ctx.lineTo(x+cw-4,y+ch/2);ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle='#ff3d00';ctx.font='bold '+(cw*0.3)+'px sans-serif';
        ctx.textAlign='center';ctx.textBaseline='middle';
        ctx.fillText('⚡',x+cw/2,y+ch/2);
      }
    }
  }

  // ── Start / Exit special tiles ─────────────────────────────────────────
  if (f.start_cell) {
    const sp=parsePos(f.start_cell); const [sx,sy]=cellXY(sp[0],sp[1]);
    ctx.fillStyle='rgba(76,175,80,0.18)';ctx.fillRect(sp[1]*cw,oY+sp[0]*ch,cw,ch);
    ctx.fillStyle='#81c784';ctx.font='bold '+(cw*0.25)+'px sans-serif';
    ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText('🚪',sx,sy);
  }

  // ── Diamond ─────────────────────────────────────────────────────────────
  if (f.diamond && !f.has_diamond) {
    const dp=parsePos(f.diamond); const [dx,dy]=cellXY(dp[0],dp[1]);
    const pulse=0.88+0.12*Math.sin(time*3);
    const dg=ctx.createRadialGradient(dx,dy,0,dx,dy,cw*0.5*pulse);
    dg.addColorStop(0,'rgba(100,220,255,0.55)'); dg.addColorStop(1,'rgba(30,150,255,0)');
    ctx.fillStyle=dg;ctx.beginPath();ctx.arc(dx,dy,cw*0.5*pulse,0,Math.PI*2);ctx.fill();
    // diamond shape
    ctx.fillStyle='#4fc3f7';
    ctx.beginPath();
    ctx.moveTo(dx,dy-ch*0.3); ctx.lineTo(dx+cw*0.22,dy-ch*0.05);
    ctx.lineTo(dx+cw*0.22,dy+ch*0.05); ctx.lineTo(dx,dy+ch*0.3);
    ctx.lineTo(dx-cw*0.22,dy+ch*0.05); ctx.lineTo(dx-cw*0.22,dy-ch*0.05);
    ctx.closePath();
    const dFill=ctx.createLinearGradient(dx-cw*0.2,dy-ch*0.3,dx+cw*0.2,dy+ch*0.3);
    dFill.addColorStop(0,'#e1f5fe'); dFill.addColorStop(0.4,'#4fc3f7'); dFill.addColorStop(1,'#0277bd');
    ctx.fillStyle=dFill; ctx.fill();
    ctx.strokeStyle='#01579b';ctx.lineWidth=1.5;ctx.stroke();
  }

  // ── Exit door ───────────────────────────────────────────────────────────
  if (f.exit_cell) {
    const ep=parsePos(f.exit_cell); const [ex,ey]=cellXY(ep[0],ep[1]);
    const open=f.has_diamond;
    if (open){
      const eg=ctx.createRadialGradient(ex,ey,0,ex,ey,cw*0.6);
      eg.addColorStop(0,'rgba(100,255,120,0.5)');eg.addColorStop(1,'rgba(50,200,70,0)');
      ctx.fillStyle=eg;ctx.beginPath();ctx.arc(ex,ey,cw*0.65,0,Math.PI*2);ctx.fill();
    }
    ctx.fillStyle=open?'#2e7d32':'#4e342e';
    ctx.beginPath();ctx.roundRect(ex-cw*0.28,ey-ch*0.38,cw*0.56,ch*0.76,4);ctx.fill();
    ctx.fillStyle=open?'#66bb6a':'#8d6e63';
    ctx.beginPath();ctx.roundRect(ex-cw*0.22,ey-ch*0.32,cw*0.44,ch*0.64,3);ctx.fill();
    ctx.fillStyle='#ffd54f';
    ctx.beginPath();ctx.arc(ex+cw*0.1,ey,cw*0.04,0,Math.PI*2);ctx.fill();
  }

  // ── Guards ─────────────────────────────────────────────────────────────
  const guards=f.guards||[], prevGuards=(prev&&prev.guards)||guards;
  for (let gi=0;gi<guards.length;gi++){
    const gp=parsePos(guards[gi]);
    const gpP=gi<prevGuards.length?parsePos(prevGuards[gi]):gp;
    const ir=lerp(gpP[0],gp[0],t), ic=lerp(gpP[1],gp[1],t);
    const [gx,gy]=cellXY(ir,ic);
    const gc2=alarm?'#c62828':'#1565c0';
    // flashlight cone
    ctx.fillStyle=alarm?'rgba(255,180,0,0.1)':'rgba(255,255,180,0.07)';
    ctx.beginPath();ctx.moveTo(gx,gy);
    ctx.lineTo(gx+cw*1.1,gy-ch*0.7);ctx.lineTo(gx+cw*1.1,gy+ch*0.7);
    ctx.closePath();ctx.fill();
    // legs
    ctx.strokeStyle='#333';ctx.lineWidth=cw*0.06;
    ctx.beginPath();ctx.moveTo(gx-cw*0.06,gy+ch*0.2);ctx.lineTo(gx-cw*0.06,gy+ch*0.38);ctx.stroke();
    ctx.beginPath();ctx.moveTo(gx+cw*0.06,gy+ch*0.2);ctx.lineTo(gx+cw*0.06,gy+ch*0.38);ctx.stroke();
    // body
    ctx.fillStyle=gc2;
    ctx.beginPath();ctx.roundRect(gx-cw*0.18,gy-ch*0.05,cw*0.36,ch*0.28,3);ctx.fill();
    // head
    ctx.fillStyle='#ffcc80';ctx.beginPath();ctx.arc(gx,gy-ch*0.15,cw*0.13,0,Math.PI*2);ctx.fill();
    // hat brim + crown
    ctx.fillStyle=gc2;
    ctx.fillRect(gx-cw*0.18,gy-ch*0.25,cw*0.36,ch*0.06);
    ctx.fillRect(gx-cw*0.11,gy-ch*0.38,cw*0.22,ch*0.13);
    // alarm beacon
    if (alarm){
      const ba=0.5+0.5*Math.sin(time*10);
      ctx.fillStyle='rgba(255,23,68,'+ba+')';
      ctx.beginPath();ctx.arc(gx,gy-ch*0.42,cw*0.07,0,Math.PI*2);ctx.fill();
    }
  }

  // ── Robber ─────────────────────────────────────────────────────────────
  if (pos) {
    const ir=lerp(prevPos[0],pos[0],t), ic=lerp(prevPos[1],pos[1],t);
    const [px,py]=cellXY(ir,ic);
    if (fx&&fx.trail){fx.trail.add(px,py);fx.trail.draw(ctx);}
    // shadow
    ctx.fillStyle='rgba(0,0,0,0.35)';
    ctx.beginPath();ctx.ellipse(px,py+ch*0.28,cw*0.22,ch*0.07,0,0,Math.PI*2);ctx.fill();
    // legs
    ctx.strokeStyle='#111';ctx.lineWidth=cw*0.07;
    ctx.beginPath();ctx.moveTo(px-cw*0.07,py+ch*0.22);ctx.lineTo(px-cw*0.07,py+ch*0.38);ctx.stroke();
    ctx.beginPath();ctx.moveTo(px+cw*0.07,py+ch*0.22);ctx.lineTo(px+cw*0.07,py+ch*0.38);ctx.stroke();
    // body
    ctx.fillStyle='#212121';
    ctx.beginPath();ctx.roundRect(px-cw*0.19,py-ch*0.05,cw*0.38,ch*0.28,3);ctx.fill();
    // arms
    ctx.strokeStyle='#212121';ctx.lineWidth=cw*0.07;
    ctx.beginPath();ctx.moveTo(px-cw*0.19,py+ch*0.02);ctx.lineTo(px-cw*0.3,py+ch*0.12);ctx.stroke();
    ctx.beginPath();ctx.moveTo(px+cw*0.19,py+ch*0.02);ctx.lineTo(px+cw*0.3,py+ch*0.12);ctx.stroke();
    // head
    ctx.fillStyle='#ffcc80';ctx.beginPath();ctx.arc(px,py-ch*0.15,cw*0.13,0,Math.PI*2);ctx.fill();
    // mask
    ctx.fillStyle='#1a1a1a';ctx.fillRect(px-cw*0.14,py-ch*0.2,cw*0.28,ch*0.08);
    ctx.fillStyle='white';
    ctx.beginPath();ctx.arc(px-cw*0.055,py-ch*0.165,cw*0.028,0,Math.PI*2);ctx.fill();
    ctx.beginPath();ctx.arc(px+cw*0.055,py-ch*0.165,cw*0.028,0,Math.PI*2);ctx.fill();
    // bag (with diamond when carrying)
    if (f.has_diamond){
      ctx.fillStyle='#0d47a1';
      ctx.beginPath();ctx.arc(px+cw*0.22,py-ch*0.22,cw*0.12,0,Math.PI*2);ctx.fill();
      ctx.fillStyle='#4fc3f7';ctx.font='bold '+(cw*0.15)+'px sans-serif';
      ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText('💎',px+cw*0.22,py-ch*0.22);
    }
    // particles
    if (fx&&fx.particles){
      if (f.camera)  fx.particles.emit(px,py,10,'#ff1744',70,0.5);
      if (f.success) fx.particles.emit(px,py,30,'#4caf50',130,1.0);
      if (f.caught)  fx.particles.emit(px,py,22,'#ff1744',110,0.8);
    }
  }

  if (f.success) drawFlash(ctx,W,H,'#4caf50',0.13);
  if (f.caught)  drawFlash(ctx,W,H,'#ff1744',0.2);
}
"""


def frame_to_canvas(env: MuseumEnv, frame: dict) -> dict:
    walls    = [f"{r},{c}" for r, c in sorted(env.walls)]
    cameras  = [f"{r},{c}" for r, c in sorted(env.cameras)]
    traps    = [f"{r},{c}" for r, c in sorted(env.traps)]
    slippery = [f"{r},{c}" for r, c in sorted(env.slippery)]
    guards   = [f"{g[0]},{g[1]}" for g in frame.get("guards", [])]
    pos      = frame.get("pos", env.start)
    return {
        "rows": env.rows, "cols": env.cols,
        "walls": walls, "cameras": cameras, "traps": traps, "slippery": slippery,
        "guards": guards,
        "pos": f"{pos[0]},{pos[1]}",
        "start_cell": f"{env.start[0]},{env.start[1]}",
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
                 "🚨 ALARM!" if frame.get("alarm") else
                 "Got the diamond!" if frame.get("has_diamond") else "Sneaking...",
    }


def draw_museum_canvas(env: MuseumEnv, frame: dict, title: str = "") -> None:
    data = frame_to_canvas(env, frame)
    if title: data["event"] = title
    render_canvas(data, DRAW_JS, width=520, height=548, title=data["event"], bg_color="#12100c")


def replay_museum_canvas(env: MuseumEnv, frames: list, key: str = "mus_replay") -> None:
    data = [frame_to_canvas(env, f) for f in frames]
    render_canvas_replay(data, DRAW_JS, width=520, height=548, bg_color="#12100c",
                         fps=8, trail_color="rgba(150,140,100,0.25)", trail_len=14, key=key)
