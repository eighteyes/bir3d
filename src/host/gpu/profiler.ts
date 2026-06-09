// profiler.ts — per-pass GPU timing via timestamp-query. The instrument the perf-LOD
// controller (blueprint §7.3) and the fluid feasibility spike (§8.1) are built on.
// Responsibilities: allocate a querySet + resolve buffer; wrap a compute pass with begin/end
// timestamps; resolve to milliseconds. Degrades to NaN when timestamp-query is unsupported.

export class GpuProfiler {
  private querySet?: GPUQuerySet;
  private resolveBuf?: GPUBuffer;
  private readBuf?: GPUBuffer;
  readonly enabled: boolean;

  constructor(private device: GPUDevice, enabled: boolean) {
    this.enabled = enabled;
    if (!enabled) return;
    this.querySet = device.createQuerySet({ type: "timestamp", count: 2 });
    this.resolveBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
    this.readBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  }

  /** Timestamp-wrapped compute pass descriptor; pass null when disabled. */
  timestampWrites(): GPUComputePassTimestampWrites | undefined {
    if (!this.enabled) return undefined;
    return { querySet: this.querySet!, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 };
  }

  /** Encode resolve+copy after the pass; call once per measured frame. */
  resolve(encoder: GPUCommandEncoder): void {
    if (!this.enabled) return;
    encoder.resolveQuerySet(this.querySet!, 0, 2, this.resolveBuf!, 0);
    encoder.copyBufferToBuffer(this.resolveBuf!, 0, this.readBuf!, 0, 16);
  }

  /** Await the mapped timestamps and return elapsed ms (NaN if disabled). Not for the hot loop. */
  async readMs(): Promise<number> {
    if (!this.enabled) return NaN;
    await this.readBuf!.mapAsync(GPUMapMode.READ);
    const raw = this.readBuf!.getMappedRange().slice(0);
    this.readBuf!.unmap();
    const t = new BigUint64Array(raw);
    return Number(t[1]! - t[0]!) / 1e6; // ns → ms
  }
}
