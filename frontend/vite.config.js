import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig(({ mode }) => ({
  plugins: [react()],
  server: {
    proxy: {
      '/api': {
        target: 'http://backend:3000',
        changeOrigin: true,
        secure: mode === 'production',
      },
      '/rails/active_storage': {
        target: 'http://backend:3000',
        changeOrigin: true,
        secure: mode === 'production',
      },
    },
  },
}));
