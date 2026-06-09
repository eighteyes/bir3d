// frameloop.ts — requestAnimationFrame driver with the blueprint's frame-timing hooks.
// Responsibilities: call onFrame(realDtSeconds) each rAF; expose start/stop; track a CPU-side
// dt so later the budget model frame_ms = max(GPU, CPU) + derate can be evaluated.

export class FrameLoop {
  private raf = 0;
  private last = 0;
  private running = false;
  constructor(private onFrame: (dtSeconds: number) => void) {}
  start(): void {
    if (this.running) return;
    this.running = true;
    const tick = (t: number) => {
      if (!this.running) return;
      const dt = this.last ? (t - this.last) / 1000 : 1 / 60;
      this.last = t;
      this.onFrame(dt);
      this.raf = requestAnimationFrame(tick);
    };
    this.raf = requestAnimationFrame(tick);
  }
  stop(): void { this.running = false; cancelAnimationFrame(this.raf); }
}
