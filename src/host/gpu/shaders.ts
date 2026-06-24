/// <reference types="vite/client" />
// shaders.ts — build-time WGSL loader.
// Responsibilities:
//   - Inline every src/host/shaders/**/*.wgsl as a raw string at BUILD time
//     (Vite ?raw glob), so production builds need no runtime fetch of source files.
//   - Resolve a shader by the absolute /src/host/... path the call sites use.
// Why: the Vite dev server serves /src/host/shaders/*.wgsl live, but `vite build`
// does not emit those source files. On a static host (e.g. GitHub Pages) a runtime
// fetch then 404s, and WebGPU compiles the returned HTML error page → invalid
// shader module → invalid pipeline → uncaptured validation errors. Bundling the
// shaders as strings makes dev and prod identical.
const SOURCES = import.meta.glob("../shaders/**/*.wgsl", {
  query: "?raw",
  import: "default",
  eager: true,
}) as Record<string, string>;

/** Resolve a bundled shader by its absolute source path, e.g. "/src/host/shaders/wind.wgsl". */
export function loadShader(absPath: string): string {
  // glob keys are relative to THIS file: /src/host/shaders/x.wgsl -> ../shaders/x.wgsl
  const key = absPath.replace("/src/host", "..");
  const src = SOURCES[key];
  if (src === undefined) {
    throw new Error(`shader not bundled: ${absPath} (looked for key ${key})`);
  }
  return src;
}
