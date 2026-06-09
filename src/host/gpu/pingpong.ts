// pingpong.ts — a read/write pair for double-buffered GPU state (fluid grids, particle SoA).
// Responsibilities: hold two like resources; expose stable current()/next(); swap each step.
// Generic over T so it unit-tests without a GPU (T can be GPUBuffer at runtime).

export class PingPong<T> {
  private a: T;
  private b: T;
  private flipped = false;
  constructor(a: T, b: T) { this.a = a; this.b = b; }
  get current(): T { return this.flipped ? this.b : this.a; }
  get next(): T { return this.flipped ? this.a : this.b; }
  swap(): void { this.flipped = !this.flipped; }
}
