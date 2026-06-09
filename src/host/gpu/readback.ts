// readback.ts — async GPU→CPU aggregate readback that NEVER blocks the frame.
// Responsibilities: round-robin staging buffers (triple by default); kick mapAsync without
// awaiting in-frame; hand the last RESOLVED result to the CPU (2–3 frames stale, by design).
// RingIndex is split out so the index policy is unit-testable without a GPU.

export class RingIndex {
  private i = 0;
  constructor(private readonly size: number) {}
  acquire(): number { return this.i; }
  advance(): void { this.i = (this.i + 1) % this.size; }
}

export class ReadbackRing {
  private readonly ring: RingIndex;
  private readonly staging: GPUBuffer[];
  private readonly inflight: boolean[];
  private latest: Float32Array | null = null;

  /** `byteLength` must EXACTLY match the src buffer size; reconstruct the ring if the layout changes. */
  constructor(private device: GPUDevice, private byteLength: number, size = 3) {
    this.ring = new RingIndex(size);
    this.staging = Array.from({ length: size }, () =>
      device.createBuffer({ size: byteLength, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ })
    );
    this.inflight = Array.from({ length: size }, () => false);
  }

  /** Encode the copy from `src` into this frame's slot, then kick a non-awaited map. */
  enqueue(encoder: GPUCommandEncoder, src: GPUBuffer): void {
    const slot = this.ring.acquire();
    if (!this.inflight[slot]) {
      encoder.copyBufferToBuffer(src, 0, this.staging[slot]!, 0, this.byteLength);
      this.inflight[slot] = true;
      // fire-and-forget: resolves a few frames later. Maps may resolve out of order under load;
      // `latest` is intentionally last-writer-wins — fine for the stale-aggregate contract.
      this.staging[slot]!.mapAsync(GPUMapMode.READ)
        .then(() => {
          this.latest = new Float32Array(this.staging[slot]!.getMappedRange().slice(0));
          this.staging[slot]!.unmap();
          this.inflight[slot] = false;
        })
        .catch((err: unknown) => {
          // device lost / validation error — release the slot so the ring doesn't starve
          this.inflight[slot] = false;
          console.error(`ReadbackRing slot ${slot} map failed:`, err);
        });
    }
    this.ring.advance();
  }

  /** Last resolved aggregate (or null until the first map resolves). Never blocks. */
  read(): Float32Array | null { return this.latest; }
}
