"""Canvas renderer for Room 3 — Street Race. City-street visuals, protest crowds."""

from __future__ import annotations
from rl.envs.racing import RacingEnv
from ui_canvas.canvas_core import render_canvas, render_canvas_replay

DRAW_JS = r"""
function drawFrame(ctx, W, H, f, prev, t, fx) {
  const rows = f.rows, cols = f.cols;
  const cw = W / cols, ch = (H - 30) / rows;
  const oY = 30;
  t = (t == null) ? 1 : t;
  const time = performance.now() * 0.001;

  const wallSet    = new Set(f.walls||[]);
  const oilSet     = new Set(f.oil||[]);
  const mudSet     = new Set(f.mud||[]);
  const crashSet   = new Set(f.crash||[]);
  const protestSet = new Set(f.protest||[]);
  const boostSet   = new Set(f.boosters||[]);

  function parsePos(s){
    if(!s)return null; const p=s.split(','); return[parseInt(p[0]),parseInt(p[1])];
  }
  function cellXY(r,c){return[c*cw+cw/2, oY+r*ch+ch/2];}

  // ── Background (night sky between buildings) ───────────────────────────
  const bg=ctx.createLinearGradient(0,oY,0,H);
  bg.addColorStop(0,'#1a1a2e'); bg.addColorStop(1,'#0d0d1a');
  ctx.fillStyle=bg; ctx.fillRect(0,oY,W,H-oY);

  const pos     = parsePos(f.pos);
  const prevPos = prev ? parsePos(prev.pos) : pos;

  // ── Draw orthodox Jewish protester (animated) ──────────────────────────
  function drawProtester(px2, py2, scale, seed) {
    const s = scale;
    const wobble = Math.sin(time*2.2 + seed*1.9) * s * 0.03;
    const legSwing = Math.sin(time*3.5 + seed) * s * 0.04;
    // long black coat (kaftan)
    const coatG = ctx.createLinearGradient(px2-s*0.15,py2-s*0.2,px2+s*0.15,py2+s*0.45);
    coatG.addColorStop(0,'#1a1a1a'); coatG.addColorStop(1,'#0a0a0a');
    ctx.fillStyle=coatG;
    ctx.beginPath();
    ctx.roundRect(px2-s*0.14,py2-s*0.2,s*0.28,s*0.48,s*0.03);
    ctx.fill();
    // white shirt collar
    ctx.fillStyle='#e8e8e8';
    ctx.beginPath();
    ctx.moveTo(px2-s*0.06,py2-s*0.2);
    ctx.lineTo(px2,py2-s*0.1);
    ctx.lineTo(px2+s*0.06,py2-s*0.2);
    ctx.closePath(); ctx.fill();
    // face
    ctx.fillStyle='#d4956a';
    ctx.beginPath();ctx.arc(px2,py2-s*0.26,s*0.11,0,Math.PI*2);ctx.fill();
    // eyes
    ctx.fillStyle='#2a1a0a';
    ctx.beginPath();ctx.arc(px2-s*0.04,py2-s*0.28,s*0.018,0,Math.PI*2);ctx.fill();
    ctx.beginPath();ctx.arc(px2+s*0.04,py2-s*0.28,s*0.018,0,Math.PI*2);ctx.fill();
    // beard
    ctx.fillStyle='#2d1f10';
    ctx.beginPath();
    ctx.ellipse(px2,py2-s*0.17,s*0.09,s*0.1,0,0,Math.PI);
    ctx.fill();
    // black hat — wide brim fedora-style
    ctx.fillStyle='#111';
    ctx.beginPath();ctx.ellipse(px2,py2-s*0.35,s*0.19,s*0.05,0,0,Math.PI*2);ctx.fill();
    ctx.fillRect(px2-s*0.1,py2-s*0.5+wobble,s*0.2,s*0.17);
    // arms raised
    ctx.strokeStyle='#111';ctx.lineWidth=s*0.05;ctx.lineCap='round';
    ctx.beginPath();
    ctx.moveTo(px2-s*0.14,py2-s*0.1+wobble*0.5);
    ctx.lineTo(px2-s*0.26,py2-s*0.25+wobble);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(px2+s*0.14,py2-s*0.1+wobble*0.5);
    ctx.lineTo(px2+s*0.26,py2-s*0.25+wobble);
    ctx.stroke();
    // protest sign
    ctx.fillStyle='#fff8dc';
    ctx.fillRect(px2+s*0.22,py2-s*0.38+wobble,s*0.22,s*0.16);
    ctx.strokeStyle='#999';ctx.lineWidth=0.5;
    ctx.strokeRect(px2+s*0.22,py2-s*0.38+wobble,s*0.22,s*0.16);
    ctx.fillStyle='#c0392b';ctx.font='bold '+(s*0.08)+'px sans-serif';
    ctx.textAlign='center';ctx.textBaseline='middle';
    ctx.fillText('NO!',px2+s*0.33,py2-s*0.3+wobble);
    // legs with animated walk
    ctx.strokeStyle='#111';ctx.lineWidth=s*0.06;
    ctx.beginPath();ctx.moveTo(px2-s*0.06,py2+s*0.28);ctx.lineTo(px2-s*0.06+legSwing,py2+s*0.44);ctx.stroke();
    ctx.beginPath();ctx.moveTo(px2+s*0.06,py2+s*0.28);ctx.lineTo(px2+s*0.06-legSwing,py2+s*0.44);ctx.stroke();
  }

  // ── Tiles ──────────────────────────────────────────────────────────────
  for (let r=0; r<rows; r++) {
    for (let c=0; c<cols; c++) {
      const x=c*cw, y=oY+r*ch;
      const key=r+','+c;

      if (wallSet.has(key)) {
        // Building block
        const bg2=ctx.createLinearGradient(x,y,x+cw,y+ch);
        bg2.addColorStop(0,'#3d3530'); bg2.addColorStop(1,'#252018');
        ctx.fillStyle=bg2; ctx.fillRect(x,y,cw,ch);
        // brick texture
        ctx.strokeStyle='rgba(0,0,0,0.5)';ctx.lineWidth=0.8;
        ctx.beginPath();
        ctx.moveTo(x,y+ch*0.33);ctx.lineTo(x+cw,y+ch*0.33);
        ctx.moveTo(x,y+ch*0.66);ctx.lineTo(x+cw,y+ch*0.66);
        ctx.moveTo(x+cw*0.5,y);ctx.lineTo(x+cw*0.5,y+ch*0.33);
        ctx.moveTo(x+cw*0.25,y+ch*0.33);ctx.lineTo(x+cw*0.25,y+ch*0.66);
        ctx.moveTo(x+cw*0.75,y+ch*0.66);ctx.lineTo(x+cw*0.75,y+ch);
        ctx.stroke();
        // lit windows
        const wx=cw*0.17, wy=ch*0.17;
        [[0.12,0.1],[0.62,0.1],[0.12,0.55],[0.62,0.55]].forEach(([fx2,fy2])=>{
          const lit=Math.random()>0.4;
          ctx.fillStyle=lit?'rgba(255,240,180,0.85)':'rgba(60,60,80,0.6)';
          ctx.fillRect(x+cw*fx2,y+ch*fy2,wx,wy);
          if(lit){
            ctx.fillStyle='rgba(255,255,200,0.2)';
            ctx.fillRect(x+cw*fx2-2,y+ch*fy2-2,wx+4,wy+4);
          }
        });
        // roof edge
        ctx.fillStyle='rgba(255,255,200,0.06)';ctx.fillRect(x,y,cw,3);
      } else {
        // Asphalt road
        const ag=ctx.createLinearGradient(x,y,x,y+ch);
        ag.addColorStop(0,'#2a2a2a'); ag.addColorStop(1,'#1e1e1e');
        ctx.fillStyle=ag; ctx.fillRect(x,y,cw,ch);
        // road grain texture
        ctx.fillStyle='rgba(255,255,255,0.012)';
        for(let i=0;i<3;i++){
          ctx.fillRect(x+Math.random()*cw,y+Math.random()*ch,1,1);
        }
      }

      // ── Road lane markings ─────────────────────────────────────────
      // vertical dashes between horizontal road cells
      if (!wallSet.has(key)) {
        // check if it's in a road row (even rows by layout pattern)
        if (r%2===0 && c>0 && !wallSet.has(r+','+(c-1)) && r!==0 && r!==rows-1) {
          ctx.strokeStyle='rgba(255,255,255,0.25)';ctx.lineWidth=1;
          ctx.setLineDash([ch*0.3,ch*0.3]);
          ctx.beginPath();ctx.moveTo(x,y+ch*0.5);ctx.lineTo(x,y+ch*0.5);
          ctx.stroke();ctx.setLineDash([]);
        }
      }

      // Oil spill (iridescent)
      if (oilSet.has(key)) {
        ctx.fillStyle='rgba(5,5,5,0.75)';
        ctx.beginPath();ctx.ellipse(x+cw/2,y+ch/2,cw*0.4,ch*0.3,0.3,0,Math.PI*2);ctx.fill();
        const og=ctx.createLinearGradient(x,y,x+cw,y+ch);
        og.addColorStop(0,'rgba(100,0,200,0.3)');
        og.addColorStop(0.33,'rgba(0,200,100,0.3)');
        og.addColorStop(0.66,'rgba(200,100,0,0.3)');
        og.addColorStop(1,'rgba(0,100,200,0.3)');
        ctx.fillStyle=og;
        ctx.beginPath();ctx.ellipse(x+cw/2,y+ch/2,cw*0.35,ch*0.25,0.3,0,Math.PI*2);ctx.fill();
        ctx.fillStyle='rgba(255,255,255,0.15)';
        ctx.beginPath();ctx.ellipse(x+cw*0.38,y+ch*0.4,cw*0.08,ch*0.05,-0.5,0,Math.PI*2);ctx.fill();
      }

      // Mud patch
      if (mudSet.has(key)) {
        ctx.fillStyle='#4a3728';
        ctx.beginPath();ctx.ellipse(x+cw/2,y+ch/2,cw*0.42,ch*0.32,0,0,Math.PI*2);ctx.fill();
        ctx.fillStyle='#5d4437';
        for(let i=0;i<6;i++){
          ctx.beginPath();
          ctx.arc(x+cw*(0.15+i*0.15),y+ch*(0.25+(i%2)*0.4),cw*0.055,0,Math.PI*2);
          ctx.fill();
        }
      }

      // Protest crowd — Orthodox Jewish protesters blocking road
      if (protestSet.has(key)) {
        // Dark crowd tint
        ctx.fillStyle='rgba(20,8,5,0.6)';ctx.fillRect(x,y,cw,ch);
        // 6 protesters in a 3×2 arrangement
        const positions=[[0.2,0.28],[0.5,0.28],[0.8,0.28],[0.2,0.7],[0.5,0.7],[0.8,0.7]];
        const sc=Math.min(cw,ch)*0.36;
        positions.forEach(([fx2,fy2],i)=>{
          drawProtester(x+cw*fx2,y+ch*fy2,sc,r*13+c*7+i*3);
        });
        // warning bar
        ctx.fillStyle='rgba(200,30,0,0.75)';
        ctx.fillRect(x+1,y+1,cw-2,10);
        ctx.fillStyle='white';ctx.font='bold 7px sans-serif';
        ctx.textAlign='center';ctx.textBaseline='middle';
        ctx.fillText('⚠ ROAD BLOCKED',x+cw/2,y+6);
      } else if (crashSet.has(key)) {
        // Generic crash barrier
        ctx.fillStyle='#b71c1c';ctx.fillRect(x+3,y+3,cw-6,ch-6);
        ctx.strokeStyle='white';ctx.lineWidth=2.5;
        ctx.beginPath();
        ctx.moveTo(x+cw*0.15,y+4);ctx.lineTo(x+cw*0.85,y+ch-4);
        ctx.moveTo(x+4,y+ch*0.5);ctx.lineTo(x+cw*0.5,y+ch-4);
        ctx.moveTo(x+cw*0.5,y+4);ctx.lineTo(x+cw-4,y+ch*0.5);
        ctx.stroke();
        ctx.fillStyle='white';ctx.font='bold '+(cw*0.38)+'px sans-serif';
        ctx.textAlign='center';ctx.textBaseline='middle';
        ctx.fillText('✕',x+cw/2,y+ch/2);
      }

      // Booster pad (glowing electric)
      if (boostSet.has(key)) {
        const bp=0.82+0.18*Math.sin(time*5+c*2.1);
        const bGrad=ctx.createRadialGradient(x+cw/2,y+ch/2,0,x+cw/2,y+ch/2,cw*0.48*bp);
        bGrad.addColorStop(0,'rgba(0,240,255,0.6)');bGrad.addColorStop(1,'rgba(0,240,255,0)');
        ctx.fillStyle=bGrad;ctx.beginPath();ctx.arc(x+cw/2,y+ch/2,cw*0.48*bp,0,Math.PI*2);ctx.fill();
        ctx.fillStyle='#00bcd4';
        ctx.beginPath();ctx.roundRect(x+cw*0.12,y+ch*0.12,cw*0.76,ch*0.76,4);ctx.fill();
        ctx.strokeStyle='#00e5ff';ctx.lineWidth=1.5;ctx.stroke();
        // lightning bolt
        ctx.fillStyle='#ffea00';
        ctx.beginPath();
        ctx.moveTo(x+cw*0.48,y+ch*0.15);ctx.lineTo(x+cw*0.62,y+ch*0.45);
        ctx.lineTo(x+cw*0.5,y+ch*0.45);ctx.lineTo(x+cw*0.58,y+ch*0.85);
        ctx.lineTo(x+cw*0.38,y+ch*0.55);ctx.lineTo(x+cw*0.5,y+ch*0.55);
        ctx.closePath();ctx.fill();
      }
    }
  }

  // ── Start marker ───────────────────────────────────────────────────────
  if (f.start) {
    const sp=parsePos(f.start); const [sx,sy]=cellXY(sp[0],sp[1]);
    ctx.fillStyle='rgba(76,175,80,0.22)';ctx.fillRect(sp[1]*cw,oY+sp[0]*ch,cw,ch);
    ctx.fillStyle='#81c784';ctx.font='bold '+(cw*0.22)+'px sans-serif';
    ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText('START',sx,sy);
  }

  // ── Finish line ────────────────────────────────────────────────────────
  if (f.finish) {
    const fp=parsePos(f.finish);
    const fx2=fp[1]*cw, fy2=oY+fp[0]*ch;
    const unlocked=f.finish_unlocked;
    const cs=Math.min(cw,ch)/4;
    if(unlocked){
      const fg=ctx.createRadialGradient(fx2+cw/2,fy2+ch/2,0,fx2+cw/2,fy2+ch/2,cw*0.6);
      fg.addColorStop(0,'rgba(100,255,100,0.35)');fg.addColorStop(1,'rgba(50,200,50,0)');
      ctx.fillStyle=fg;ctx.fillRect(fx2,fy2,cw,ch);
    }
    const cols2=Math.ceil(cw/cs), rows2=Math.ceil(ch/cs);
    for(let i=0;i<cols2;i++) for(let j=0;j<rows2;j++){
      if((i+j)%2===0){
        ctx.fillStyle=unlocked?'rgba(255,255,255,0.85)':'rgba(160,160,160,0.5)';
        ctx.fillRect(fx2+i*cs,fy2+j*cs,cs,cs);
      } else {
        ctx.fillStyle=unlocked?'rgba(0,0,0,0.85)':'rgba(60,60,60,0.5)';
        ctx.fillRect(fx2+i*cs,fy2+j*cs,cs,cs);
      }
    }
    if(!unlocked){
      ctx.fillStyle='rgba(0,0,0,0.4)';ctx.fillRect(fx2,fy2,cw,ch);
      ctx.fillStyle='#ff8f00';ctx.font='bold '+(cw*0.28)+'px sans-serif';
      ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText('🔒',fx2+cw/2,fy2+ch/2);
    }
  }

  // ── Race car ───────────────────────────────────────────────────────────
  if (pos) {
    const ir=lerp(prevPos[0],pos[0],t), ic=lerp(prevPos[1],pos[1],t);
    const [px,py]=cellXY(ir,ic);
    if(fx&&fx.trail){fx.trail.add(px,py);fx.trail.draw(ctx);}

    // shadow
    ctx.fillStyle='rgba(0,0,0,0.4)';
    ctx.beginPath();ctx.ellipse(px,py+ch*0.24,cw*0.3,ch*0.08,0,0,Math.PI*2);ctx.fill();

    // car body gradient
    const cg=ctx.createLinearGradient(px-cw*0.3,py-ch*0.2,px+cw*0.3,py+ch*0.2);
    cg.addColorStop(0,'#ef5350');cg.addColorStop(0.4,'#e53935');cg.addColorStop(1,'#b71c1c');
    ctx.fillStyle=cg;
    ctx.beginPath();ctx.roundRect(px-cw*0.3,py-ch*0.19,cw*0.6,ch*0.38,5);ctx.fill();
    ctx.strokeStyle='#b71c1c';ctx.lineWidth=1.2;ctx.stroke();

    // windshield
    ctx.fillStyle='rgba(186,230,253,0.85)';
    ctx.beginPath();ctx.roundRect(px-cw*0.13,py-ch*0.16,cw*0.26,ch*0.12,2);ctx.fill();
    ctx.strokeStyle='#0369a1';ctx.lineWidth=0.8;ctx.stroke();
    // windshield reflection
    ctx.fillStyle='rgba(255,255,255,0.3)';
    ctx.beginPath();ctx.moveTo(px-cw*0.1,py-ch*0.16);ctx.lineTo(px-cw*0.01,py-ch*0.04);
    ctx.lineTo(px-cw*0.12,py-ch*0.04);ctx.closePath();ctx.fill();

    // wheels
    [[px-cw*0.22,py-ch*0.18],[px+cw*0.22,py-ch*0.18],[px-cw*0.22,py+ch*0.18],[px+cw*0.22,py+ch*0.18]].forEach(([wx,wy])=>{
      ctx.fillStyle='#222';ctx.beginPath();ctx.arc(wx,wy,cw*0.075,0,Math.PI*2);ctx.fill();
      ctx.fillStyle='#555';ctx.beginPath();ctx.arc(wx,wy,cw*0.04,0,Math.PI*2);ctx.fill();
    });
    // racing number
    ctx.fillStyle='white';ctx.font='bold '+(cw*0.22)+'px sans-serif';
    ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText('1',px,py+ch*0.04);

    // particles
    if(fx&&fx.particles){
      if(f.crash_event) fx.particles.emit(px,py,28,'#ff6600',130,0.85);
      if(f.success) fx.particles.emit(px,py,32,'#4caf50',110,1.0);
      if(prev&&f.collected>(prev.collected||0)) fx.particles.emit(px,py,16,'#00e5ff',90,0.55);
    }
  }

  if(f.crash_event) drawFlash(ctx,W,H,'#ff3300',0.22);
  if(f.success) drawFlash(ctx,W,H,'#4caf50',0.14);
}
"""


def frame_to_canvas(env: RacingEnv, frame: dict) -> dict:
    walls   = [f"{r},{c}" for r, c in sorted(env.walls)]
    oil     = [f"{r},{c}" for r, c in sorted(env.oil)]
    mud     = [f"{r},{c}" for r, c in sorted(env.mud)]
    protest = [f"{r},{c}" for r, c in sorted(getattr(env, "protest", set()))]
    crash_only = [f"{r},{c}" for r, c in sorted(env.crash - getattr(env, "protest", set()))]
    bmask   = frame.get("bmask", 0)
    boosters = [f"{r},{c}" for r, c in env.remaining_boosters(bmask)]
    pos     = frame.get("pos", env.start)
    collected = env.collected_count(bmask)
    return {
        "rows": env.rows, "cols": env.cols,
        "walls": walls, "oil": oil, "mud": mud,
        "crash": crash_only, "protest": protest, "boosters": boosters,
        "pos": f"{pos[0]},{pos[1]}",
        "start": f"{env.start[0]},{env.start[1]}",
        "finish": f"{env.finish[0]},{env.finish[1]}",
        "finish_unlocked": env.finish_unlocked(bmask),
        "collected": collected,
        "step": frame.get("step", 0),
        "cum_reward": frame.get("cum_reward"),
        "crash_event": frame.get("crash", False),
        "success": frame.get("success", False),
        "event": "Crashed into protesters!" if frame.get("crash") else
                 "Finished!" if frame.get("success") else
                 f"⚡ {collected}/{env.min_boosters} needed" if not env.finish_unlocked(bmask) else
                 "Finish unlocked! Go!",
    }


def draw_racing_canvas(env: RacingEnv, frame: dict, title: str = "") -> None:
    data = frame_to_canvas(env, frame)
    if title: data["event"] = title
    render_canvas(data, DRAW_JS, width=520, height=548, title=data["event"], bg_color="#1a1a2e")


def replay_racing_canvas(env: RacingEnv, frames: list, key: str = "race_replay") -> None:
    data = [frame_to_canvas(env, f) for f in frames]
    render_canvas_replay(data, DRAW_JS, width=520, height=548, bg_color="#1a1a2e",
                         fps=8, trail_color="rgba(100,100,100,0.45)", trail_len=22, key=key)
