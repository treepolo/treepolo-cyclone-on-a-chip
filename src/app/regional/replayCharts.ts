// Chart data of a replayed frame. A frame recorded with composite maps gives the maps at once (unpacked here); a frame recorded
// in full also gives sections, soundings, slices and radius-height means: the worker computes them on a model of its own from
// the packed state (worker.ts replayChart). The worker is sent the model's meta and the frame's data only when it does not hold
// them, so stepping through the same frame with other chart settings costs no transfer.
import { dequant } from '../../regional/replayData.js';
import { pressure } from '../../regional/diagnostics.js';
import type { ChartData, ChartRequest, FromRegionalWorker, ToRegionalWorker } from './protocol.js';
import type { ReplayFrame, ReplayStore } from './replay.js';

export const NOTE_VIEW = '這個畫面只記錄了 3D 畫面，沒有圖表資料。要分析請把「記錄內容」改成「合成圖」或「完整」再記錄（軸對稱模式只能記錄 3D 畫面）。 / This frame holds only the 3-D view, no chart data. Set the recording content to composite maps or full to analyse frames (the axisymmetric model records its 3-D view only).';
export const NOTE_MAPS = '這個畫面只記錄了合成圖；水平切片、剖面、探空和徑向－高度圖需要「完整」記錄。 / This frame holds only the composite maps; slices, sections, soundings and radius-height means need the full recording.';
const NOTE_META = '找不到這個畫面的模式資料（檔案不完整）。 / The model data of this frame is missing (incomplete file).';

type Answer = Extract<FromRegionalWorker, { type: 'replayChart' }>;

export class ReplayCharts {
  private seq = 0;
  private readonly pending = new Map<number, (a: Answer) => void>();
  /** what the worker holds: the meta id and the key of the frame data (-1: nothing) */
  private sent = { metaId: -1, key: -1 };
  private readonly keys = new WeakMap<object, number>();
  private nextKey = 1;
  constructor(private readonly send: (m: ToRegionalWorker) => void) {}

  /** An answer from the worker. */
  onAnswer(a: Answer): void { const r = this.pending.get(a.id); if (r) { this.pending.delete(a.id); r(a); } }
  /** Forget what the worker holds (it was restarted or its model changed). */
  reset(): void { this.sent = { metaId: -1, key: -1 }; }

  /** a number for a frame or a meta, unique in this page (the worker holds one meta and one frame at a time, by these numbers) */
  private keyOf(o: object): number {
    let k = this.keys.get(o);
    if (!k) { k = this.nextKey++; this.keys.set(o, k); }
    return k;
  }

  /** The maps of a frame that holds them, unpacked; the base-state temperature and pressure of the levels. */
  private local(f: ReplayFrame, store: ReplayStore, req: ChartRequest): ChartData {
    const rec = f.rec, meta = rec ? store.metas.get(rec.metaId) : undefined, charts: ChartData = { maps: {}, slice: null, section: null, sounding: null, rz: null };
    if (rec?.maps) for (const v of req.maps) { const q = rec.maps[v]; if (q) charts.maps[v] = dequant<Float32Array>(q); }
    if (meta) {
      const b = meta.base;
      charts.tz = Float32Array.from(b.pi0, (p, k) => b.th0[k]! * p);
      charts.p0 = Float32Array.from(b.pi0, (p) => pressure(p) / 100);
    }
    return charts;
  }

  /** The chart data of a frame for a request, and what to tell where the frame holds no data for a chart. */
  async compute(f: ReplayFrame, store: ReplayStore, req: ChartRequest): Promise<{ charts: ChartData; note: string }> {
    const rec = f.rec;
    if (!rec) return { charts: this.local(f, store, req), note: NOTE_VIEW };
    const fullWanted = !!(req.slice || req.section || req.sounding || req.rz);
    if (!fullWanted && rec.maps) return { charts: this.local(f, store, req), note: '' };
    if (!rec.state) return { charts: this.local(f, store, req), note: NOTE_MAPS };
    const meta = store.metas.get(rec.metaId);
    if (!meta) return { charts: this.local(f, store, req), note: NOTE_META };
    const key = this.keyOf(f), metaKey = this.keyOf(meta);
    for (let attempt = 0; attempt < 3; attempt++) {
      const id = ++this.seq, withMeta = this.sent.metaId !== metaKey, withFrame = withMeta || this.sent.key !== key;
      const answer = new Promise<Answer>((resolve) => this.pending.set(id, resolve));
      this.send({ type: 'replayChart', id, req, metaId: metaKey, ...(withMeta ? { meta } : {}), key, ...(withFrame ? { frame: { state: rec.state, aux: rec.aux, centre: rec.centre } } : {}) });
      if (withMeta) this.sent.metaId = metaKey;
      if (withFrame) this.sent.key = key;
      const a = await answer;
      if (a.need) { if (a.need === 'meta') this.sent.metaId = -1; this.sent.key = -1; continue; }
      if (a.error || !a.charts) throw new Error(a.error ?? 'no chart data');
      return { charts: a.charts, note: '' };
    }
    throw new Error('the worker did not take the frame');
  }
}
