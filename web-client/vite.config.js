import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    // Disable proxy when using ngrok URL to avoid conflicts
    proxy: process.env.USE_PROXY === 'true' ? {
      '/api': {
        target: 'http://localhost:5000',
        changeOrigin: true
      }
    } : {}
  }
})