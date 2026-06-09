// device.ts — acquire the GPUDevice and detect engine-critical features.
// Responsibilities: adapter/device acquisition; timestamp-query feature gate; install an
// uncaptured-error logger so validation errors are never silent.

export interface GpuContext {
  adapter: GPUAdapter;
  device: GPUDevice;
  hasTimestampQuery: boolean;
}

export async function acquireDevice(): Promise<GpuContext> {
  if (!("gpu" in navigator)) throw new Error("WebGPU unavailable: navigator.gpu missing");
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("WebGPU unavailable: no adapter");
  const hasTimestampQuery = adapter.features.has("timestamp-query");
  const requiredFeatures: GPUFeatureName[] = hasTimestampQuery ? ["timestamp-query"] : [];
  const device = await adapter.requestDevice({ requiredFeatures });
  device.addEventListener("uncapturederror", (e) =>
    console.error("[WebGPU uncaptured]", (e as GPUUncapturedErrorEvent).error)
  );
  return { adapter, device, hasTimestampQuery };
}
