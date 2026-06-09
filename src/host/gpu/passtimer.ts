// passtimer.ts — multi-stage GPU timestamp timer for the fluid budget spike (§8.1).
// Generalizes the single-pass GpuProfiler: instead of one begin/end pair, it hands out a
// fresh (begin, end) timestamp pair per timed compute pass and buckets each pass's elapsed
// time under a caller-supplied STAGE label. Stages in GpuFluid.step are interleaved and
// repeated (divergence ×2, jacobi ×2×iters, advect ×3, set_bnd scattered), so per-stage ms
// is the SUM of all passes tagged that stage — NOT a single contiguous bracket.
// Responsibilities:
//   - Allocate a timestamp querySet of count = 2*capacity (begin/end per timed pass) + the
//     resolve/read buffers; size capacity to the max passes a frame will record.
//   - timestampWrites(stage): reserve the next pair, remember its stage, return the descriptor
//     to splice into a compute pass (begin = 2k, end = 2k+1). Returns undefined when disabled.
//   - resolve(encoder): after the frame's passes, copy the used query range to a read buffer.
//   - readStages(): await the map, return { perStage ms (summed by label), total ms
//     (max end − min begin over the frame), passes count }. NOT for a production hot loop;
//     this is a measurement instrument (8 bytes × K, GPU-side timings) used in the spike test.
//   - reset(): zero the pair counter + clear the per-pass stage tags for the next frame.
// Degrades to a no-op (all NaN) when `enabled` is false — caller passes `hasTimestampQuery`.

export interface StageTimings {
  /** Summed elapsed ms per stage label (sum over all passes tagged that stage). */
  perStage: Record<string, number>;
  /** Whole-frame ms: max(end) − min(begin) across every timed pass this frame. */
  total: number;
  /** Number of timed passes recorded this frame. */
  passes: number;
}

export class PassTimer {
  private querySet?: GPUQuerySet;
  private resolveBuf?: GPUBuffer;
  private readBuf?: GPUBuffer;
  readonly enabled: boolean;
  readonly capacity: number;

  // Per-frame state, reset() each frame.
  private k = 0; // next pair index (pass count this frame)
  private stages: string[] = []; // stages[k] = label of the k-th timed pass

  /**
   * @param capacity max number of timed passes a single frame will record. The querySet holds
   *   2*capacity timestamps (begin/end per pass). Compute it from the worst-case pass count
   *   (a function of iters); do NOT hardcode.
   */
  constructor(device: GPUDevice, enabled: boolean, capacity: number) {
    this.enabled = enabled;
    this.capacity = capacity;
    if (!enabled) return;
    const count = 2 * capacity;
    const bytes = count * 8; // 8 bytes per u64 timestamp
    this.querySet = device.createQuerySet({ type: "timestamp", count });
    this.resolveBuf = device.createBuffer({
      size: bytes,
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
    });
    this.readBuf = device.createBuffer({
      size: bytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
  }

  /** Clear the per-frame pair counter + stage tags. Call once at the start of each timed frame. */
  reset(): void {
    this.k = 0;
    this.stages.length = 0;
  }

  /**
   * Reserve the next timestamp pair for a pass and tag it with `stage`. Splice the returned
   * descriptor into `encoder.beginComputePass({ timestampWrites })`. Returns undefined when
   * disabled (the pass runs untimed). Throws if the frame exceeds `capacity`.
   */
  timestampWrites(stage: string): GPUComputePassTimestampWrites | undefined {
    if (!this.enabled) return undefined;
    if (this.k >= this.capacity) {
      throw new Error(`PassTimer capacity ${this.capacity} exceeded (pass ${this.k}); size it larger`);
    }
    const begin = 2 * this.k;
    this.stages[this.k] = stage;
    this.k += 1;
    return { querySet: this.querySet!, beginningOfPassWriteIndex: begin, endOfPassWriteIndex: begin + 1 };
  }

  /** Encode resolve+copy for the pairs used THIS frame. Call after the frame's passes, before submit. */
  resolve(encoder: GPUCommandEncoder): void {
    if (!this.enabled || this.k === 0) return;
    const used = 2 * this.k;
    encoder.resolveQuerySet(this.querySet!, 0, used, this.resolveBuf!, 0);
    encoder.copyBufferToBuffer(this.resolveBuf!, 0, this.readBuf!, 0, used * 8);
  }

  /**
   * Await the mapped timestamps and aggregate this frame's pairs into per-stage sums + total.
   * Bucketing handles interleaved/repeated stages. NOT a hot-loop call (it maps + awaits).
   * Returns all-NaN when disabled or no passes were recorded.
   */
  async readStages(): Promise<StageTimings> {
    if (!this.enabled || this.k === 0) {
      return { perStage: {}, total: NaN, passes: 0 };
    }
    const used = 2 * this.k;
    await this.readBuf!.mapAsync(GPUMapMode.READ, 0, used * 8);
    const raw = this.readBuf!.getMappedRange(0, used * 8).slice(0);
    this.readBuf!.unmap();
    const t = new BigUint64Array(raw);

    const perStage: Record<string, number> = {};
    let minBegin = t[0]!;
    let maxEnd = t[1]!;
    for (let p = 0; p < this.k; p++) {
      const b = t[2 * p]!;
      const e = t[2 * p + 1]!;
      if (b < minBegin) minBegin = b;
      if (e > maxEnd) maxEnd = e;
      const ms = Number(e - b) / 1e6;
      const stage = this.stages[p]!;
      perStage[stage] = (perStage[stage] ?? 0) + ms;
    }
    const total = Number(maxEnd - minBegin) / 1e6;
    return { perStage, total, passes: this.k };
  }

  /** Release GPU resources. */
  destroy(): void {
    this.querySet?.destroy();
    this.resolveBuf?.destroy();
    this.readBuf?.destroy();
  }
}
