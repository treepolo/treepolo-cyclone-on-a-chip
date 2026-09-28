// Unattended experiments on the regional page: run for a set number of model hours at full speed,
// record the storm statistics at a fixed model-time interval, save the final state, and store the report
// in the artifact's database (collection "runs") where the developer can read it. Without a database
// (running outside the artifact viewer) the report stays on the page and can be copied.

import type { RegionalFrame } from './protocol.js';

interface Db { doc(path: string): { set(d: Record<string, unknown>): Promise<void> } }
type Claude = { use(n: string): Promise<unknown> };

/** The artifact runtime of this page, or of the host page when embedded in its overlay. */
export function claudeRuntime(): Claude | null {
  const own = (window as unknown as { claude?: Claude }).claude;
  if (own) return own;
  try { if (window.parent !== window) return (window.parent as unknown as { claude?: Claude }).claude ?? null; } catch { /* cross-origin */ }
  return null;
}

export interface RunConfig { hours: number; everyMin: number; note: string; autoSave: boolean }
export interface RunHooks {
  experiment(): string; grid(): string;
  start(cfg: RunConfig): void;                      // set full speed, display cadence, run-until, run
  finish(): void;                                   // restore display cadence
  save(): Promise<string | null>;                   // save the final state, resolves its title
  log(s: string): void; status(s: string): void;
}

const KEYS = ['t', 'wmax', 'wmin', 'qcmax', 'qrmax', 'rainmax', 'vmax', 'dp', 'rmw', 'zeta', 'vground', 'ew1r', 'ew1v', 'ew2r', 'ew2v', 'ew3r', 'ew3v'] as const;

export class UnattendedRun {
  private active = false;
  private series: Record<string, (number | null)[]> = {};
  private cfg: RunConfig | null = null;
  private id = '';
  private meta: Record<string, unknown> = {};
  private lastFlush = 0;
  private writing: Promise<void> = Promise.resolve();
  private db: Db | null = null;
  constructor(private readonly h: RunHooks) {}

  get running(): boolean { return this.active; }

  async begin(cfg: RunConfig): Promise<void> {
    this.cfg = cfg; this.active = true; this.series = {}; for (const k of KEYS) this.series[k] = [];
    this.id = `run-${new Date().toISOString().replace(/[:.]/g, '-')}`;
    this.meta = { experiment: this.h.experiment(), grid: this.h.grid(), note: cfg.note, hours: cfg.hours, everyMin: cfg.everyMin, started: new Date().toISOString(), userAgent: navigator.userAgent };
    const cl = claudeRuntime();
    this.db = cl ? await cl.use('db').catch(() => null) as Db | null : null;
    this.h.log(this.db ? '無人值守實驗開始；報告會自動存到雲端資料庫 / unattended run started; the report is stored in the artifact database'
      : '無人值守實驗開始（這個檢視沒有雲端資料庫，報告只保留在本頁）/ unattended run started (no artifact database here: the report stays on this page)');
    this.lastFlush = performance.now();
    this.h.start(cfg);
    this.status();
  }

  /** Record a frame (called for every frame while running). */
  sample(f: RegionalFrame): void {
    if (!this.active) return;
    const s = f.stats, ew = s.eyewalls ?? [];
    const row: Record<(typeof KEYS)[number], number | null> = {
      t: f.time, wmax: s.wmax, wmin: s.wmin, qcmax: s.qcmax, qrmax: s.qrmax, rainmax: s.rainmax, vmax: s.vmax, dp: s.dp, rmw: s.rmw,
      zeta: s.zetaMax, vground: s.vGround, ew1r: ew[0]?.r ?? null, ew1v: ew[0]?.v ?? null, ew2r: ew[1]?.r ?? null, ew2v: ew[1]?.v ?? null, ew3r: ew[2]?.r ?? null, ew3v: ew[2]?.v ?? null,
    };
    const ts = this.series.t!;
    if (ts.length && f.time <= (ts[ts.length - 1] as number) + 1e-6) return;     // duplicate frame
    for (const k of KEYS) this.series[k]!.push(row[k] === null ? null : +Number(row[k]).toPrecision(5));
    this.status();
    if (performance.now() - this.lastFlush > 5 * 60 * 1000) { this.lastFlush = performance.now(); void this.flush('running'); }
  }

  /** The run-until pause arrived: store the final report (and state). */
  async end(reason: string): Promise<void> {
    if (!this.active) return;
    this.active = false;
    this.h.finish();
    let saved: string | null = null;
    if (this.cfg?.autoSave) { try { saved = await this.h.save(); } catch (e) { this.h.log(`自動存檔失敗 / auto-save failed: ${String(e)}`); } }
    this.meta.savedState = saved;
    await this.flush(reason);
    this.h.log(`無人值守實驗結束 / unattended run finished: ${reason}${this.db ? '（報告已存到雲端）/ report stored' : ''}`);
    this.status();
  }

  async abort(reason: string): Promise<void> { if (this.active) { this.active = false; this.h.finish(); await this.flush(reason); this.status(); } }

  private thinned(): Record<string, (number | null)[]> {
    // keep the document well under the 256 KiB limit: at most 1200 samples (every n-th, last kept)
    const n = this.series.t!.length, stride = Math.max(1, Math.ceil(n / 1200)), out: Record<string, (number | null)[]> = {};
    for (const k of KEYS) out[k] = this.series[k]!.filter((_, i) => i % stride === 0 || i === n - 1);
    return out;
  }

  private flush(status: string): Promise<void> {
    const doc = { ...this.meta, status, updated: new Date().toISOString(), samples: this.series.t!.length, series: this.thinned() };
    this.writing = this.writing.then(async () => {
      if (!this.db) return;
      try { await this.db.doc(`runs/${this.id}`).set(doc); }
      catch (e) { this.h.log(`報告寫入失敗 / report write failed: ${(e as { code?: string }).code ?? String(e)}`); }
    });
    return this.writing;
  }

  /** Compact text of the report (for copying when there is no database). */
  reportText(): string { return JSON.stringify({ ...this.meta, samples: this.series.t?.length ?? 0, series: this.thinned() }); }

  private status(): void {
    const n = this.series.t?.length ?? 0, t = n ? (this.series.t![n - 1] as number) : 0;
    this.h.status(this.active ? `執行中 / running · ${n} 筆紀錄 / samples · t = ${(t / 3600).toFixed(2)} h` : n ? `已結束 / finished · ${n} 筆紀錄 / samples` : '—');
  }
}
