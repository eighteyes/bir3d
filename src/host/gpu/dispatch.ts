// dispatch.ts — encode/submit compute passes over a 1D workload.
// Responsibilities: build a pipeline from WGSL; record a compute pass that maps bindings[i] -> @binding(i)
// and dispatches ceil(n/workgroupSize). Two entry points, by who owns the encoder:
//   - encodeComputePass: records into a CALLER-OWNED encoder (the frame loop batches many passes, one submit)
//   - dispatchCompute:   standalone convenience (own encoder, immediate submit) for tests / one-offs

export function makeComputePipeline(device: GPUDevice, code: string, entryPoint = "main"): GPUComputePipeline {
  const module = device.createShaderModule({ code });
  return device.createComputePipeline({ layout: "auto", compute: { module, entryPoint } });
}

export function encodeComputePass(
  device: GPUDevice,
  encoder: GPUCommandEncoder,
  pipeline: GPUComputePipeline,
  bindings: GPUBuffer[],
  workItems: number,
  workgroupSize = 64,
  timestampWrites?: GPUComputePassTimestampWrites
): void {
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: bindings.map((buffer, binding) => ({ binding, resource: { buffer } })),
  });
  const pass = encoder.beginComputePass(timestampWrites ? { timestampWrites } : undefined);
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(workItems / workgroupSize));
  pass.end();
}

export function dispatchCompute(
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  bindings: GPUBuffer[],
  workItems: number,
  workgroupSize = 64
): void {
  const encoder = device.createCommandEncoder();
  encodeComputePass(device, encoder, pipeline, bindings, workItems, workgroupSize);
  device.queue.submit([encoder.finish()]);
}
