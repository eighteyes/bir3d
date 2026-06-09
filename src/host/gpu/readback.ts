// readback.ts — async GPU→CPU aggregate readback that NEVER blocks the frame.
// Responsibilities: round-robin staging buffers (triple by default). The map MUST be kicked
// AFTER the caller submits — a buffer that is map-pending at submit time is a validation error
// that drops the whole command buffer. So the frame contract is two calls:
//   1. enqueue(encoder, src)  — record the copy into the caller's encoder, reserve the slot
//   2. <caller submits>
//   3. afterSubmit()          — kick the non-awaited map for the slot copied this frame
// read() hands back the last RESOLVED aggregate (2–3 frames stale, by design). One enqueue +
// one afterSubmit per frame. RingIndex is split out so the index policy is unit-testable.

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
  private pendingMap: number | null = null; // slot copied this frame, awaiting afterSubmit()

  /** `byteLength` must EXACTLY match the src buffer size; reconstruct the ring if the layout changes. */
  constructor(private device: GPUDevice, private byteLength: number, size = 3) {
    this.ring = new RingIndex(size);
    this.staging = Array.from({ length: size }, () =>
      device.createBuffer({ size: byteLength, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ })
    );
    this.inflight = Array.from({ length: size }, () => false);
  }

  /** Record the copy from `src` into this frame's slot. Does NOT map — call afterSubmit() post-submit. */
  enqueue(encoder: GPUCommandEncoder, src: GPUBuffer): void {
    const slot = this.ring.acquire();
    if (!this.inflight[slot]) {
      encoder.copyBufferToBuffer(src, 0, this.staging[slot]!, 0, this.byteLength);
      this.inflight[slot] = true;
      this.pendingMap = slot;
    } else {
      this.pendingMap = null; // slot still mapping from an earlier frame; skip this frame
    }
    this.ring.advance();
  }

  /** Call AFTER device.queue.submit(). Kicks the non-awaited map for the slot copied this frame. */
  afterSubmit(): void {
    const slot = this.pendingMap;
    if (slot === null) return;
    this.pendingMap = null;
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

  /** Last resolved aggregate (or null until the first map resolves). Never blocks. */
  read(): Float32Array | null { return this.latest; }
}
