// Simulation pacing, shared by the global and regional workers: a target speed (model seconds per wall
// second, 0 = as fast as possible) and an optional stop time ("run until").

export class Pacer {
  /** model seconds per wall-clock second; 0 = full speed */
  target = 0;
  /** model time (s) at which to pause, or null */
  until: number | null = null;
  private wall = 0;
  private model = 0;

  /** Restart the speed reference (on run, on a speed change, after loading). */
  reset(modelTime: number): void { this.wall = performance.now(); this.model = modelTime; }

  /** Number of steps (at most `want`) that may run now; 0 = wait. */
  allow(modelTime: number, dt: number, want: number): number {
    let n = Math.max(1, want);
    if (this.target > 0) {
      const now = performance.now();
      const allowed = this.model + this.target * (now - this.wall) / 1000;
      // falling far behind (slow hardware, long frame): do not bank the deficit for a later burst
      if (allowed - modelTime > 2 * this.target) { this.wall = now; this.model = modelTime; return Math.min(n, Math.max(1, Math.ceil(this.target / dt))); }
      if (modelTime + dt > allowed + 1e-9) return 0;
      n = Math.min(n, Math.floor((allowed - modelTime) / dt) + 1);
    }
    if (this.until !== null) n = Math.min(n, Math.max(1, Math.ceil((this.until - modelTime) / dt - 1e-6)));
    return n;
  }

  /** True once the stop time is reached (and clears it). */
  reached(modelTime: number): boolean {
    if (this.until === null || modelTime < this.until - 1e-6) return false;
    this.until = null;
    return true;
  }
}
