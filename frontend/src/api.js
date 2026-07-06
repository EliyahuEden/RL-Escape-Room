// Thin API client. In dev, vite proxies /api to the FastAPI backend (:8000).
const BASE = '/api';

async function get(path) {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
  return res.json();
}

async function post(path, body) {
  const res = await fetch(`${BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body || {}),
  });
  if (!res.ok) {
    let detail = `${res.status}`;
    try { detail = (await res.json()).detail || detail; } catch { /* noop */ }
    throw new Error(detail);
  }
  return res.json();
}

export const api = {
  rooms: () => get('/rooms'),
  room: (id) => get(`/rooms/${id}`),
  train: (id, params) => post(`/train/${id}`, { params }),
  status: (id) => get(`/train/${id}/status`),
  stop: (id) => post(`/train/${id}/stop`),
  evaluate: (id, episodes = 10) => post(`/evaluate/${id}`, { episodes }),
  metrics: (id) => get(`/metrics/${id}`),
  policy: (id) => get(`/policy/${id}`),
  replays: (id) => get(`/replay/${id}`),
  replay: (id, rid) => get(`/replay/${id}/${rid}`),
  getConfig: (id) => get(`/config/${id}`),
  saveConfig: (id, values) => post(`/config/${id}`, { values }),
  summary: () => get('/results/summary'),
  health: () => get('/health'),
};

// events the replay viewer highlights in green
export const GOOD_EVENTS = new Set([
  'coin', 'diamond', 'boost', 'escaped', 'finished', 'goal', 'goal!',
  'crossed', 'dodged', 'shooting zone', 'shortcut', 'success',
]);
