"""HTML5 Canvas rendering engine — smooth rAF loop with delta-time interpolation."""

from __future__ import annotations

import json

import streamlit.components.v1 as components

_SHARED_JS = r"""
// ── Math helpers ───────────────────────────────────────────────────────────
function lerp(a, b, t) { return a + (b - a) * t; }

// ── Particle system ────────────────────────────────────────────────────────
class Particles {
  constructor() { this.list = []; }
  emit(x, y, count, color, speed, life) {
    for (let i = 0; i < count; i++) {
      const a = Math.random() * Math.PI * 2;
      const s = speed * (0.4 + Math.random() * 0.8);
      this.list.push({ x, y, vx: Math.cos(a)*s, vy: Math.sin(a)*s,
        life: life*(0.5+Math.random()*0.5), maxLife: life,
        color, size: 2 + Math.random()*3 });
    }
  }
  update(dt) {
    for (const p of this.list) {
      p.x += p.vx*dt; p.y += p.vy*dt; p.vy += 80*dt; p.life -= dt;
    }
    this.list = this.list.filter(p => p.life > 0);
  }
  draw(ctx) {
    for (const p of this.list) {
      const a = Math.max(0, p.life/p.maxLife);
      ctx.globalAlpha = a;
      ctx.fillStyle = p.color;
      ctx.beginPath(); ctx.arc(p.x, p.y, p.size*a, 0, Math.PI*2); ctx.fill();
    }
    ctx.globalAlpha = 1;
  }
}

// ── Trail system ───────────────────────────────────────────────────────────
class Trail {
  constructor(maxLen, color, width) {
    this.pts = []; this.maxLen = maxLen||20;
    this.color = color||'rgba(255,255,255,0.3)'; this.width = width||2;
  }
  add(x, y) { this.pts.push([x,y]); if (this.pts.length>this.maxLen) this.pts.shift(); }
  draw(ctx) {
    if (this.pts.length < 2) return;
    for (let i = 1; i < this.pts.length; i++) {
      const a = (i/this.pts.length)*0.55;
      ctx.strokeStyle = this.color; ctx.lineWidth = this.width;
      ctx.globalAlpha = a; ctx.lineCap = 'round';
      ctx.beginPath(); ctx.moveTo(this.pts[i-1][0],this.pts[i-1][1]);
      ctx.lineTo(this.pts[i][0],this.pts[i][1]); ctx.stroke();
    }
    ctx.globalAlpha = 1;
  }
  clear() { this.pts = []; }
}

// ── HUD ────────────────────────────────────────────────────────────────────
function drawHUD(ctx, W, H, frame, extra) {
  ctx.fillStyle = 'rgba(0,0,0,0.6)';
  ctx.fillRect(0, 0, W, 28);
  ctx.font = 'bold 12px "Segoe UI",sans-serif';
  ctx.textBaseline = 'middle';
  ctx.fillStyle = '#fff'; ctx.textAlign = 'left';
  ctx.fillText('Step: '+(frame.step!=null?frame.step:'—'), 8, 14);
  if (frame.cum_reward != null) {
    ctx.fillStyle = frame.cum_reward >= 0 ? '#69f0ae' : '#ff5252';
    ctx.textAlign = 'center';
    ctx.fillText('Reward: '+frame.cum_reward.toFixed(1), W/2, 14);
  }
  ctx.fillStyle = '#ccc'; ctx.textAlign = 'right';
  ctx.fillText(extra||'', W-8, 14);
}

// ── Screen flash ───────────────────────────────────────────────────────────
function drawFlash(ctx, W, H, color, alpha) {
  ctx.globalAlpha = alpha; ctx.fillStyle = color;
  ctx.fillRect(0, 0, W, H); ctx.globalAlpha = 1;
}
"""


def render_canvas(
    frame_data: dict,
    draw_js: str,
    width: int = 520,
    height: int = 520,
    title: str = "",
    bg_color: str = "#222",
) -> None:
    frame_json = json.dumps(frame_data)
    html = f"""
    <div style="text-align:center;font-family:'Segoe UI',sans-serif;">
      <div style="font-size:14px;font-weight:bold;margin-bottom:4px;color:#eee;">{title}</div>
      <canvas id="gc" width="{width}" height="{height}"
              style="border:2px solid #555;border-radius:10px;background:{bg_color};
                     box-shadow:0 4px 24px rgba(0,0,0,0.5);display:block;margin:0 auto;"></canvas>
    </div>
    <script>
    (function(){{
      const canvas=document.getElementById('gc');
      const ctx=canvas.getContext('2d');
      const W={width},H={height};
      const frame={frame_json};
      {_SHARED_JS}
      {draw_js}
      drawFrame(ctx,W,H,frame,null,1,null);
      drawHUD(ctx,W,H,frame,frame.event||'');
    }})();
    </script>
    """
    components.html(html, height=height + 50, scrolling=False)


def render_canvas_replay(
    frames_data: list,
    draw_js: str,
    width: int = 520,
    height: int = 520,
    title: str = "",
    bg_color: str = "#222",
    fps: int = 8,
    interpolation_steps: int = 1,   # kept for API compat; smoothing is now delta-time based
    trail_color: str = "rgba(255,255,255,0.3)",
    trail_len: int = 20,
    key: str = "replay",
) -> None:
    """Smooth Canvas replay using proper delta-time rAF loop (no busy-poll jank)."""
    frames_json = json.dumps(frames_data)
    frame_ms = int(1000 / fps)   # ms per logical frame

    html = f"""
    <div style="text-align:center;font-family:'Segoe UI',sans-serif;" id="wrap_{key}">
      <canvas id="gc_{key}" width="{width}" height="{height}"
              style="border:2px solid #555;border-radius:10px;background:{bg_color};
                     box-shadow:0 4px 24px rgba(0,0,0,0.5);display:block;margin:0 auto;"></canvas>
      <div style="margin-top:8px;display:flex;align-items:center;justify-content:center;gap:8px;flex-wrap:wrap;">
        <button id="play_{key}"  style="padding:5px 20px;border:none;border-radius:6px;background:#4CAF50;color:#fff;font-weight:bold;cursor:pointer;font-size:13px;">▶ Play</button>
        <button id="pause_{key}" style="padding:5px 18px;border:none;border-radius:6px;background:#ff9800;color:#fff;font-weight:bold;cursor:pointer;font-size:13px;">⏸</button>
        <button id="reset_{key}" style="padding:5px 14px;border:none;border-radius:6px;background:#607d8b;color:#fff;font-weight:bold;cursor:pointer;font-size:13px;">⏮</button>
        <select id="speed_{key}" style="padding:4px 8px;border-radius:6px;border:1px solid #555;background:#333;color:#eee;font-size:12px;">
          <option value="0.5">0.5×</option>
          <option value="1" selected>1×</option>
          <option value="2">2×</option>
          <option value="4">4×</option>
        </select>
        <span id="info_{key}" style="color:#aaa;font-size:12px;min-width:110px;text-align:left;"></span>
      </div>
      <input id="slider_{key}" type="range" min="0" max="{len(frames_data)-1}" value="0"
             style="width:{width-20}px;margin-top:6px;accent-color:#4CAF50;">
    </div>
    <script>
    (function(){{
      const canvas = document.getElementById('gc_{key}');
      const ctx = canvas.getContext('2d');
      const W={width}, H={height};
      const FRAMES = {frames_json};
      const N = FRAMES.length;
      const FRAME_MS = {frame_ms};

      {_SHARED_JS}
      {draw_js}

      const particles = new Particles();
      const trail = new Trail({trail_len}, '{trail_color}', 2.5);
      const fx = {{particles, trail}};

      let playing = false, raf = null;
      // floatingIdx: real-valued frame index (0 to N-1), fractional part = interp t
      let floatIdx = 0;
      let lastTs = null;

      function getSpeed() {{
        return parseFloat(document.getElementById('speed_{key}').value)||1;
      }}

      function renderAt(floatI) {{
        const i = Math.min(Math.floor(floatI), N-1);
        const t = floatI - Math.floor(floatI);
        const f = FRAMES[i];
        const prev = i > 0 ? FRAMES[i-1] : f;

        ctx.clearRect(0,0,W,H);
        ctx.fillStyle = '{bg_color}'; ctx.fillRect(0,0,W,H);

        drawFrame(ctx,W,H,f,prev,t,fx);
        particles.draw(ctx);
        drawHUD(ctx,W,H,f,f.event||'');

        document.getElementById('slider_{key}').value = i;
        document.getElementById('info_{key}').textContent =
          'Frame '+(f.step!=null?f.step:i)+' / '+(N-1);
      }}

      function tick(ts) {{
        if (!playing) return;
        if (lastTs === null) lastTs = ts;
        const dt = Math.min(ts - lastTs, 100); // cap at 100ms to prevent spiral after tab switch
        lastTs = ts;

        const speed = getSpeed();
        floatIdx += (dt / FRAME_MS) * speed;
        particles.update(dt/1000);

        if (floatIdx >= N-1) {{
          floatIdx = N-1;
          renderAt(floatIdx);
          playing = false;
          return;
        }}
        renderAt(floatIdx);
        raf = requestAnimationFrame(tick);
      }}

      function startPlay() {{
        if (playing) return;
        if (floatIdx >= N-1) {{ floatIdx = 0; trail.clear(); particles.list=[]; }}
        playing = true;
        lastTs = null;
        raf = requestAnimationFrame(tick);
      }}

      function pausePlay() {{
        playing = false;
        if (raf) {{ cancelAnimationFrame(raf); raf=null; }}
      }}

      function resetPlay() {{
        pausePlay();
        floatIdx = 0;
        trail.clear(); particles.list = [];
        renderAt(0);
      }}

      document.getElementById('play_{key}').onclick  = startPlay;
      document.getElementById('pause_{key}').onclick = pausePlay;
      document.getElementById('reset_{key}').onclick = resetPlay;
      document.getElementById('slider_{key}').oninput = function() {{
        pausePlay();
        trail.clear(); particles.list = [];
        floatIdx = parseInt(this.value);
        renderAt(floatIdx);
      }};

      renderAt(0);
    }})();
    </script>
    """
    components.html(html, height=height + 90, scrolling=False)
