import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Dev server proxies /api to the FastAPI backend (port 8000), so
// `npm run dev` + `python -m backend.api.main` just work together.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': 'http://127.0.0.1:8000',
    },
  },
  build: {
    chunkSizeWarningLimit: 1200,
  },
});
