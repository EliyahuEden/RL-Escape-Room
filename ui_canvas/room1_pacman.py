"""Canvas renderer for Room 1 — Pacman. Dark dungeon aesthetic, smooth animation."""

from __future__ import annotations
from rl.envs.pacman import PacmanEnv
from ui_canvas.canvas_core import render_canvas, render_canvas_replay

DRAW_JS = r"""
function drawFrame(ctx, W, H, f, prev, t, fx) {
  const rows = f.rows, cols = f.cols;
  const cw = W / cols, ch = (H - 30) / rows;
  const oY = 30;
  t = (t == null) ? 1 : t;
  const time = performance.now() * 0.001;

  // Pre-build sets for O(1) lookup
  const wallSet = new Set(f.walls);
  const slipSet = new Set(f.slippery||[]);
  const remSet  = new Set(f.remaining||[]);

  function parsePos(s) {
    if (!s) return null;
    const p = s.split(','); return [parseInt(p[0]), parseInt(p[1])];
  }
  function cellXY(r, c) { return [c*cw + cw/2, oY + r*ch + ch/2]; }

  // ── Background gradient ────────────────────────────────────────────────
  const bg = ctx.createLinearGradient(0, oY, 0, H);
  bg.addColorStop(0, '#050510');
  bg.addColorStop(1, '#0a0820');
  ctx.fillStyle = bg; ctx.fillRect(0, oY, W, H-oY);

  const pos     = parsePos(f.pos);
  const prevPos = prev ? parsePos(prev.pos) : pos;

  // ── Grid cells ─────────────────────────────────────────────────────────
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const x = c*cw, y = oY + r*ch;
      const key = r+','+c;

      if (wallSet.has(key)) {
        // Stone-block wall with bevel
        const g = ctx.createLinearGradient(x, y, x+cw, y+ch);
        g.addColorStop(0, '#1e2a4a'); g.addColorStop(1, '#0d1630');
        ctx.fillStyle = g; ctx.fillRect(x, y, cw, ch);
        // inner bevel
        ctx.fillStyle = '#2a3860';
        ctx.fillRect(x+1, y+1, cw-2, 3);
        ctx.fillRect(x+1, y+1, 3, ch-2);
        ctx.fillStyle = '#060c20';
        ctx.fillRect(x+1, y+ch-4, cw-2, 3);
        ctx.fillRect(x+cw-4, y+1, 3, ch-2);
        // neon edge on non-wall neighbors
        ctx.strokeStyle = '#3355cc';
        ctx.lineWidth = 1.5;
        const nw = (rr,cc)=> rr<0||rr>=rows||cc<0||cc>=cols||!wallSet.has(rr+','+cc);
        if (nw(r-1,c)){ctx.beginPath();ctx.moveTo(x,y);ctx.lineTo(x+cw,y);ctx.stroke();}
        if (nw(r+1,c)){ctx.beginPath();ctx.moveTo(x,y+ch);ctx.lineTo(x+cw,y+ch);ctx.stroke();}
        if (nw(r,c-1)){ctx.beginPath();ctx.moveTo(x,y);ctx.lineTo(x,y+ch);ctx.stroke();}
        if (nw(r,c+1)){ctx.beginPath();ctx.moveTo(x+cw,y);ctx.lineTo(x+cw,y+ch);ctx.stroke();}
      } else {
        // Floor tile
        ctx.fillStyle = (r+c)%2===0 ? '#09091e' : '#0c0c22';
        ctx.fillRect(x, y, cw, ch);
        // subtle corner dots
        ctx.fillStyle = '#1a1a35';
        ctx.fillRect(x, y, 2, 2); ctx.fillRect(x+cw-2, y, 2, 2);
        ctx.fillRect(x, y+ch-2, 2, 2); ctx.fillRect(x+cw-2, y+ch-2, 2, 2);
      }

      // Icy tile
      if (slipSet.has(key)) {
        const ig = ctx.createRadialGradient(x+cw/2,y+ch/2,0,x+cw/2,y+ch/2,cw*0.5);
        ig.addColorStop(0,'rgba(130,200,255,0.18)'); ig.addColorStop(1,'rgba(70,150,255,0)');
        ctx.fillStyle = ig; ctx.fillRect(x,y,cw,ch);
        ctx.strokeStyle='rgba(160,220,255,0.4)';ctx.lineWidth=0.8;
        ctx.beginPath();ctx.moveTo(x+cw*0.25,y+ch*0.4);ctx.lineTo(x+cw*0.75,y+ch*0.4);ctx.stroke();
        ctx.beginPath();ctx.moveTo(x+cw*0.5,y+ch*0.25);ctx.lineTo(x+cw*0.5,y+ch*0.7);ctx.stroke();
      }
    }
  }

  // ── Coins ──────────────────────────────────────────────────────────────
  for (const key of remSet) {
    const p = parsePos(key);
    const [cx2, cy2] = cellXY(p[0], p[1]);
    const pulse = 0.78 + 0.22*Math.sin(time*2.5 + p[1]*1.3 + p[0]*0.9);
    const r2 = cw*0.18*pulse;
    // outer glow
    const cg = ctx.createRadialGradient(cx2,cy2,0,cx2,cy2,cw*0.42*pulse);
    cg.addColorStop(0,'rgba(255,210,0,0.45)'); cg.addColorStop(1,'rgba(255,180,0,0)');
    ctx.fillStyle=cg; ctx.beginPath(); ctx.arc(cx2,cy2,cw*0.42*pulse,0,Math.PI*2); ctx.fill();
    // coin body
    const cBody = ctx.createRadialGradient(cx2-r2*0.25,cy2-r2*0.25,0,cx2,cy2,r2);
    cBody.addColorStop(0,'#fff9c4'); cBody.addColorStop(0.5,'#ffd700'); cBody.addColorStop(1,'#b8860b');
    ctx.fillStyle=cBody; ctx.beginPath(); ctx.arc(cx2,cy2,r2,0,Math.PI*2); ctx.fill();
    ctx.strokeStyle='#996600';ctx.lineWidth=1;ctx.stroke();
    ctx.fillStyle='#8b6914';ctx.font='bold '+(r2*1.1)+'px sans-serif';
    ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText('$',cx2,cy2+0.5);
  }

  // ── Door (exit) ─────────────────────────────────────────────────────────
  if (f.door) {
    const dp = parsePos(f.door);
    const [dx,dy] = cellXY(dp[0],dp[1]);
    const open = remSet.size === 0;
    if (open) {
      const dg = ctx.createRadialGradient(dx,dy,0,dx,dy,cw*0.6);
      dg.addColorStop(0,'rgba(100,255,120,0.5)'); dg.addColorStop(1,'rgba(50,200,70,0)');
      ctx.fillStyle=dg; ctx.beginPath(); ctx.arc(dx,dy,cw*0.7,0,Math.PI*2); ctx.fill();
    }
    // door frame
    ctx.fillStyle = open ? '#1b5e20' : '#3e2723';
    ctx.beginPath(); ctx.roundRect(dx-cw*0.3,dy-ch*0.38,cw*0.6,ch*0.76,4); ctx.fill();
    // door face
    ctx.fillStyle = open ? '#4caf50' : '#6d4c41';
    ctx.beginPath(); ctx.roundRect(dx-cw*0.25,dy-ch*0.33,cw*0.5,ch*0.66,3); ctx.fill();
    // knob
    ctx.fillStyle='#ffd54f';
    ctx.beginPath();ctx.arc(dx+cw*0.12,dy+ch*0.05,cw*0.045,0,Math.PI*2);ctx.fill();
    if (open) {
      ctx.fillStyle='#a5d6a7';ctx.font='bold '+(cw*0.22)+'px sans-serif';
      ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText('EXIT',dx,dy-ch*0.15);
    }
  }

  // ── Guard / Ghost ──────────────────────────────────────────────────────
  if (f.guard) {
    const gp = parsePos(f.guard);
    const gpP = prev && prev.guard ? parsePos(prev.guard) : gp;
    const ir = lerp(gpP[0],gp[0],t), ic = lerp(gpP[1],gp[1],t);
    const [gx,gy] = cellXY(ir,ic);
    const bob = Math.sin(time*3)*ch*0.04;
    // glow aura
    const gg = ctx.createRadialGradient(gx,gy,0,gx,gy,cw*0.5);
    gg.addColorStop(0,'rgba(220,30,30,0.25)'); gg.addColorStop(1,'rgba(220,30,30,0)');
    ctx.fillStyle=gg; ctx.beginPath(); ctx.arc(gx,gy,cw*0.5,0,Math.PI*2); ctx.fill();
    // skirt wavy bottom
    ctx.fillStyle='#c62828';
    ctx.beginPath();
    ctx.arc(gx, gy-ch*0.05+bob, cw*0.32, Math.PI, 0);
    ctx.lineTo(gx+cw*0.32, gy+ch*0.28+bob);
    const segs = 5;
    for (let i = segs; i >= 0; i--) {
      const wx = gx - cw*0.32 + i*(cw*0.64/segs);
      const wave = (i%2===0 ? -1 : 1)*ch*0.07 + bob*0.6;
      ctx.lineTo(wx, gy+ch*0.28+wave);
    }
    ctx.closePath(); ctx.fill();
    // eyes
    ctx.fillStyle='white';
    ctx.beginPath();ctx.arc(gx-cw*0.1,gy-ch*0.1+bob,cw*0.09,0,Math.PI*2);ctx.fill();
    ctx.beginPath();ctx.arc(gx+cw*0.1,gy-ch*0.1+bob,cw*0.09,0,Math.PI*2);ctx.fill();
    ctx.fillStyle='#1a237e';
    ctx.beginPath();ctx.arc(gx-cw*0.07,gy-ch*0.09+bob,cw*0.05,0,Math.PI*2);ctx.fill();
    ctx.beginPath();ctx.arc(gx+cw*0.13,gy-ch*0.09+bob,cw*0.05,0,Math.PI*2);ctx.fill();
  }

  // ── Pacman ─────────────────────────────────────────────────────────────
  if (pos) {
    const ir = lerp(prevPos[0],pos[0],t), ic = lerp(prevPos[1],pos[1],t);
    const [px,py] = cellXY(ir,ic);
    if (fx && fx.trail) { fx.trail.add(px,py); fx.trail.draw(ctx); }

    const mouthAmt = 0.18 + 0.18*Math.abs(Math.sin(time*7));
    let angle = 0;
    const act = f.action;
    if (act===0) angle=-Math.PI/2; else if (act===1) angle=Math.PI/2;
    else if (act===2) angle=Math.PI;

    // body glow
    const pg = ctx.createRadialGradient(px,py,0,px,py,cw*0.5);
    pg.addColorStop(0,'rgba(255,210,0,0.3)'); pg.addColorStop(1,'rgba(255,210,0,0)');
    ctx.fillStyle=pg; ctx.beginPath(); ctx.arc(px,py,cw*0.5,0,Math.PI*2); ctx.fill();

    // body
    const pBody = ctx.createRadialGradient(px-cw*0.12,py-cw*0.12,0,px,py,cw*0.38);
    pBody.addColorStop(0,'#fff9c4'); pBody.addColorStop(0.4,'#ffd23f'); pBody.addColorStop(1,'#e6b800');
    ctx.fillStyle=pBody;
    ctx.beginPath();
    ctx.moveTo(px,py);
    ctx.arc(px,py,cw*0.38,angle+mouthAmt,angle+Math.PI*2-mouthAmt);
    ctx.closePath(); ctx.fill();
    ctx.strokeStyle='#b8860b';ctx.lineWidth=1.2;ctx.stroke();

    // eye
    const eyeA = angle-0.55;
    ctx.fillStyle='#111';
    ctx.beginPath();ctx.arc(px+Math.cos(eyeA)*cw*0.17,py+Math.sin(eyeA)*cw*0.17,cw*0.055,0,Math.PI*2);ctx.fill();

    // particles
    if (fx && fx.particles) {
      if (prev && f.remaining && prev.remaining && f.remaining.length<prev.remaining.length)
        fx.particles.emit(px,py,14,'#ffd700',90,0.7);
      if (f.escaped) fx.particles.emit(px,py,30,'#4caf50',140,1.0);
      if (f.caught)  fx.particles.emit(px,py,22,'#ff1744',110,0.9);
    }
  }

  if (f.escaped) drawFlash(ctx,W,H,'#4caf50',0.15);
  if (f.caught)  drawFlash(ctx,W,H,'#ff1744',0.20);
}
"""


def frame_to_canvas(env: PacmanEnv, frame: dict) -> dict:
    walls     = [f"{r},{c}" for r, c in sorted(env.walls)]
    slippery  = [f"{r},{c}" for r, c in sorted(env.slippery)]
    remaining = [f"{r},{c}" for r, c in frame.get("remaining", [])]
    pos   = frame.get("pos", env.start)
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
    render_canvas(data, DRAW_JS, width=520, height=548, title=data["event"], bg_color="#050510")


def replay_pacman_canvas(env: PacmanEnv, frames: list, key: str = "pac_replay") -> None:
    data = [frame_to_canvas(env, f) for f in frames]
    render_canvas_replay(data, DRAW_JS, width=520, height=548, bg_color="#050510",
                         fps=8, trail_color="rgba(255,210,63,0.3)", trail_len=18, key=key)
