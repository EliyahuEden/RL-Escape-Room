"""Advanced HTML5 Canvas rendering engine for all rooms.

Generates self-contained HTML pages with embedded JavaScript featuring:
- Smooth position interpolation between frames
- Trail effects for agent movement
- Particle system for events (goal flash, crash sparks, alarm)
- HUD overlay with step counter, reward, and status
- Play/pause/speed controls with frame slider
"""

from __future__ import annotations

import json
from typing import Optional

import streamlit.components.v1 as components

# Shared JS utilities injected into every canvas page
_SHARED_JS = """
// ── Interpolation ──────────────────────────────────────────────────────
function lerp(a, b, t) { return a + (b - a) * t; }
function lerpPos(a, b, t) { return [lerp(a[0],b[0],t), lerp(a[1],b[1],t)]; }

// ── Particle system ────────────────────────────────────────────────────
class Particles {
  constructor() { this.list = []; }
  emit(x, y, count, color, speed, life) {
    for (let i = 0; i < count; i++) {
      const a = Math.random() * Math.PI * 2;
      const s = speed * (0.5 + Math.random());
      this.list.push({
        x, y, vx: Math.cos(a)*s, vy: Math.sin(a)*s,
        life: life * (0.5 + Math.random()*0.5), maxLife: life,
        color, size: 2 + Math.random()*3
      });
    }
  }
  update(dt) {
    for (let p of this.list) {
      p.x += p.vx * dt; p.y += p.vy * dt;
      p.vy += 30 * dt; // gravity
      p.life -= dt;
    }
    this.list = this.list.filter(p => p.life > 0);
  }
  draw(ctx) {
    for (let p of this.list) {
      const alpha = Math.max(0, p.life / p.maxLife);
      ctx.globalAlpha = alpha;
      ctx.fillStyle = p.color;
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.size * alpha, 0, Math.PI*2);
      ctx.fill();
    }
    ctx.globalAlpha = 1;
  }
}

// ── Trail system ───────────────────────────────────────────────────────
class Trail {
  constructor(maxLen, color, width) {
    this.points = []; this.maxLen = maxLen || 20;
    this.color = color || 'rgba(255,255,255,0.3)'; this.width = width || 2;
  }
  add(x, y) {
    this.points.push([x, y]);
    if (this.points.length > this.maxLen) this.points.shift();
  }
  draw(ctx) {
    if (this.points.length < 2) return;
    ctx.strokeStyle = this.color;
    ctx.lineWidth = this.width;
    ctx.lineCap = 'round';
    ctx.beginPath();
    ctx.moveTo(this.points[0][0], this.points[0][1]);
    for (let i = 1; i < this.points.length; i++) {
      ctx.globalAlpha = i / this.points.length * 0.6;
      ctx.lineTo(this.points[i][0], this.points[i][1]);
    }
    ctx.stroke();
    ctx.globalAlpha = 1;
  }
  clear() { this.points = []; }
}

// ── HUD overlay ────────────────────────────────────────────────────────
function drawHUD(ctx, W, H, frame, extra) {
  // top bar
  ctx.fillStyle = 'rgba(0,0,0,0.55)';
  ctx.fillRect(0, 0, W, 28);
  ctx.fillStyle = '#fff';
  ctx.font = 'bold 12px "Segoe UI", sans-serif';
  ctx.textBaseline = 'middle';
  ctx.textAlign = 'left';
  const step = frame.step != null ? frame.step : '—';
  ctx.fillText('Step: ' + step, 8, 14);
  if (frame.cum_reward != null) {
    const r = frame.cum_reward;
    ctx.fillStyle = r >= 0 ? '#69f0ae' : '#ff5252';
    ctx.textAlign = 'center';
    ctx.fillText('Reward: ' + r.toFixed(1), W/2, 14);
  }
  ctx.fillStyle = '#fff';
  ctx.textAlign = 'right';
  ctx.fillText(extra || '', W - 8, 14);
}

// ── Screen flash ───────────────────────────────────────────────────────
function drawFlash(ctx, W, H, color, alpha) {
  ctx.fillStyle = color;
  ctx.globalAlpha = alpha;
  ctx.fillRect(0, 0, W, H);
  ctx.globalAlpha = 1;
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
              style="border:2px solid #444;border-radius:8px;background:{bg_color};
                     box-shadow:0 4px 20px rgba(0,0,0,0.4);"></canvas>
    </div>
    <script>
    (function() {{
      const canvas = document.getElementById('gc');
      const ctx = canvas.getContext('2d');
      const W = {width}, H = {height};
      const frame = {frame_json};
      {_SHARED_JS}
      {draw_js}
      drawFrame(ctx, W, H, frame, null, null, null);
    }})();
    </script>
    """
    components.html(html, height=height + 40, scrolling=False)


def render_canvas_replay(
    frames_data: list,
    draw_js: str,
    width: int = 520,
    height: int = 520,
    title: str = "",
    bg_color: str = "#222",
    fps: int = 10,
    interpolation_steps: int = 4,
    trail_color: str = "rgba(255,255,255,0.3)",
    trail_len: int = 25,
    key: str = "replay",
) -> None:
    """Full-featured Canvas replay with interpolation, trails, particles, HUD."""
    frames_json = json.dumps(frames_data)
    html = f"""
    <div style="text-align:center;font-family:'Segoe UI',sans-serif;" id="wrap_{key}">
      <canvas id="gc_{key}" width="{width}" height="{height}"
              style="border:2px solid #444;border-radius:8px;background:{bg_color};
                     box-shadow:0 4px 20px rgba(0,0,0,0.4);"></canvas>
      <div style="margin-top:8px;display:flex;align-items:center;justify-content:center;gap:6px;">
        <button id="play_{key}" style="padding:5px 18px;border:none;border-radius:6px;
                background:#4CAF50;color:white;font-weight:bold;cursor:pointer;font-size:13px;">
          ▶ Play</button>
        <button id="pause_{key}" style="padding:5px 18px;border:none;border-radius:6px;
                background:#ff9800;color:white;font-weight:bold;cursor:pointer;font-size:13px;">
          ⏸</button>
        <button id="reset_{key}" style="padding:5px 14px;border:none;border-radius:6px;
                background:#607d8b;color:white;font-weight:bold;cursor:pointer;font-size:13px;">
          ⏮</button>
        <select id="speed_{key}" style="padding:4px 8px;border-radius:6px;border:1px solid #555;
                background:#333;color:#eee;font-size:12px;">
          <option value="0.5">0.5×</option>
          <option value="1" selected>1×</option>
          <option value="2">2×</option>
          <option value="4">4×</option>
        </select>
        <span id="info_{key}" style="color:#aaa;font-size:12px;min-width:120px;"></span>
      </div>
      <input id="slider_{key}" type="range" min="0" max="{len(frames_data)-1}" value="0"
             style="width:{width-20}px;margin-top:6px;accent-color:#4CAF50;">
    </div>
    <script>
    (function() {{
      const canvas = document.getElementById('gc_{key}');
      const ctx = canvas.getContext('2d');
      const W = {width}, H = {height};
      const frames = {frames_json};
      const baseFps = {fps};
      const interpSteps = {interpolation_steps};
      let idx = 0, playing = false, raf = null;
      let lastTime = 0;

      {_SHARED_JS}

      const particles = new Particles();
      const trail = new Trail({trail_len}, '{trail_color}', 2.5);

      {draw_js}

      function getSpeed() {{
        return parseFloat(document.getElementById('speed_{key}').value) || 1;
      }}

      function showFrame(i, subT) {{
        idx = Math.min(Math.max(i, 0), frames.length - 1);
        const f = frames[idx];
        const prev = idx > 0 ? frames[idx-1] : f;
        const t = subT != null ? subT : 1;

        ctx.clearRect(0, 0, W, H);
        ctx.fillStyle = '{bg_color}';
        ctx.fillRect(0, 0, W, H);

        drawFrame(ctx, W, H, f, prev, t, {{ particles, trail }});

        particles.draw(ctx);

        const ev = f.event || f.title || '';
        drawHUD(ctx, W, H, f, ev);

        document.getElementById('slider_{key}').value = idx;
        document.getElementById('info_{key}').textContent =
          'Step ' + (f.step != null ? f.step : idx) + ' / ' + (frames.length - 1);
      }}

      let subFrame = 0;
      function animate(ts) {{
        if (!playing) return;
        const speed = getSpeed();
        const interval = 1000 / (baseFps * speed * interpSteps);
        if (ts - lastTime >= interval) {{
          lastTime = ts;
          subFrame++;
          if (subFrame >= interpSteps) {{
            subFrame = 0;
            idx++;
            if (idx >= frames.length) {{
              idx = frames.length - 1;
              playing = false;
              showFrame(idx, 1);
              return;
            }}
          }}
          showFrame(idx, subFrame / interpSteps);
        }}
        raf = requestAnimationFrame(animate);
      }}

      document.getElementById('play_{key}').onclick = function() {{
        if (playing) return;
        playing = true;
        if (idx >= frames.length - 1) {{ idx = 0; trail.clear(); }}
        subFrame = 0;
        lastTime = performance.now();
        raf = requestAnimationFrame(animate);
      }};
      document.getElementById('pause_{key}').onclick = function() {{
        playing = false;
        if (raf) cancelAnimationFrame(raf);
      }};
      document.getElementById('reset_{key}').onclick = function() {{
        playing = false;
        if (raf) cancelAnimationFrame(raf);
        idx = 0; subFrame = 0;
        trail.clear();
        particles.list = [];
        showFrame(0, 1);
      }};
      document.getElementById('slider_{key}').oninput = function() {{
        playing = false;
        if (raf) cancelAnimationFrame(raf);
        trail.clear();
        particles.list = [];
        showFrame(parseInt(this.value), 1);
      }};

      showFrame(0, 1);
    }})();
    </script>
    """
    components.html(html, height=height + 80, scrolling=False)
