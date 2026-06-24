// vite.config.ts
// Vite build + dev configuration for the vector-system / bir3d WebGPU demos.
// Responsibilities:
//   - Serve all demos in dev on a fixed port
//   - Set the GitHub Pages base path so built asset URLs resolve under /bir3d/
//   - Declare every HTML entry point as a multi-page build input
import { defineConfig } from "vite";
import { resolve } from "node:path";

// Base path is overridable via VITE_BASE so the same config works locally
// (default "/") and on GitHub Pages (project site served under /bir3d/).
const base = process.env.VITE_BASE ?? "/";

export default defineConfig({
  base,
  server: { port: 5173 },
  build: {
    rollupOptions: {
      input: {
        main: resolve(__dirname, "index.html"),
        bird: resolve(__dirname, "index-bird.html"),
        fluid: resolve(__dirname, "index-fluid.html"),
      },
    },
  },
});
