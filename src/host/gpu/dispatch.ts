// dispatch.ts — encode and submit a single compute pass over a 1D workload.
// Responsibilities: build a pipeline from WGSL, bind group from buffers, dispatch ceil(n/wg).

export function makeComputePipeline(device: GPUDevice, code: string, entryPoint = "main"): GPUComputePipeline {
  const module = device.createShaderModule({ code });
  return device.createComputePipeline({ layout: "auto", compute: { module, entryPoint } });
}

export function runComputePass(
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  bindings: GPUBuffer[],
  workItems: number,
  workgroupSize = 64,
  encoder = device.createCommandEncoder(),
  submit = true
): GPUCommandEncoder {
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: bindings.map((buffer, binding) => ({ binding, resource: { buffer } })),
  });
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(workItems / workgroupSize));
  pass.end();
  if (submit) device.queue.submit([encoder.finish()]);
  return encoder;
}
