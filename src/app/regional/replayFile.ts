// Replay data as a file of its own (not part of the ordinary saves, which would grow with it): the chosen frames with their
// recordings in the save container of saves.ts (kind 'replay'), which the page offers as a ZIP like the saves.
//
// header: one entry per frame (time, grid, storms, numbers, the ranges of its packed arrays) and per model (meta); the arrays:
//   f<i>.cloud / .rain / .ground (bytes of the 3-D view), f<i>.ncloud / .nrain (the eye nest's), f<i>.m.<map> (16-bit packed map),
//   f<i>.s.<n> (16-bit packed state field n), f<i>.a.<name> (horizontal fields, f32), m<id>.land
import { packSave, unpackSave, type SaveArrays, type SaveMeta } from '../saves.js';
import type { StormNow } from '../../regional/storms.js';
import { recBytes, type Q16, type ReplayAux, type ReplayMeta, type ReplayRec, type ReplayTier } from '../../regional/replayData.js';
import { size, type ReplayFrame } from './replay.js';
import type { NestFrame, RegionalFrame } from './protocol.js';

export interface ExportOptions { segments: { t0: number; t1: number }[]; tier: ReplayTier; stride: number }

export const tierOf = (f: ReplayFrame): ReplayTier => f.rec?.tier ?? 'view';

/** The recording of a frame cut down to a tier (a frame never gains data). */
function cut(f: ReplayFrame, tier: ReplayTier): ReplayFrame {
  const rec = f.rec;
  if (!rec) return f;
  if (tier === 'view') return { ...f, rec: null };
  if (tier === 'maps' && rec.state) return { ...f, rec: { ...rec, tier: 'maps', state: null, aux: null } };
  return f;
}

/** The frames in the segments (every `stride`-th of them), with the recordings cut down to the tier. */
export function chooseFrames(all: ReplayFrame[], o: ExportOptions): ReplayFrame[] {
  const inSegs = all.filter((f) => o.segments.some((s) => f.t >= s.t0 - 1e-6 && f.t <= s.t1 + 1e-6));
  const st = Math.max(1, Math.round(o.stride));
  return inSegs.filter((_, i) => i % st === 0).map((f) => cut(f, o.tier));
}

/** Bytes (before compression) the chosen frames come to. */
export const estimateBytes = (frames: ReplayFrame[]): number => frames.reduce((a, f) => a + size(f), 0);

interface FrameHead {
  t: number; nx: number; ny: number; nz: number; dx: number; dz: number; Lx: number; Ly: number; top: number;
  origin: { x: number; y: number };
  stats: RegionalFrame['stats'];
  nest: Omit<NestFrame, 'cloud' | 'rain'> | null;
  rec: null | { tier: ReplayTier; metaId: number; centre: { x: number; y: number } | null; maps: { name: string; lo: number; hi: number }[]; state: { lo: number; hi: number }[]; aux: string[] | null; vel: { u: number; v: number } | null };
}
interface MetaHead { id: number; meta: Omit<ReplayMeta, 'land'>; land: boolean }
interface ReplayHead extends SaveMeta { kind: 'replay'; version: 1; created: number; frames: FrameHead[]; metas: MetaHead[] }

const bytesOf = (q: Q16): Uint8Array => new Uint8Array(q.q.buffer, q.q.byteOffset, q.q.byteLength);
const AUX = ['rain', 'snow', 'rate', 'cu', 'uh', 'wind', 'sst'] as const;

/** The container of the frames (only the models they use are written). */
export function packReplay(frames: ReplayFrame[], metas: Map<number, ReplayMeta>, title: string): { meta: SaveMeta; data: ArrayBuffer } {
  const arrays: SaveArrays = {}, heads: FrameHead[] = [], used = new Set<number>();
  frames.forEach((f, i) => {
    arrays[`f${i}.cloud`] = f.cloud; arrays[`f${i}.rain`] = f.rain; arrays[`f${i}.ground`] = f.ground;
    let nest: FrameHead['nest'] = null;
    if (f.nest) { const { cloud, rain, ...info } = f.nest; nest = info; arrays[`f${i}.ncloud`] = cloud; arrays[`f${i}.nrain`] = rain; }
    let rec: FrameHead['rec'] = null;
    const r = f.rec;
    if (r) {
      used.add(r.metaId);
      const maps = Object.entries(r.maps ?? {});
      maps.forEach(([k, q]) => { arrays[`f${i}.m.${k}`] = bytesOf(q); });
      (r.state ?? []).forEach((q, n) => { arrays[`f${i}.s.${n}`] = bytesOf(q); });
      let aux: string[] | null = null;
      if (r.aux) { aux = []; for (const k of AUX) { const a = r.aux[k]; if (a) { arrays[`f${i}.a.${k}`] = a; aux.push(k); } } }
      rec = { tier: r.tier, metaId: r.metaId, centre: r.centre, maps: maps.map(([name, q]) => ({ name, lo: q.lo, hi: q.hi })), state: (r.state ?? []).map((q) => ({ lo: q.lo, hi: q.hi })), aux, vel: r.aux?.vel ?? null };
    }
    heads.push({ t: f.t, nx: f.nx, ny: f.ny, nz: f.nz, dx: f.dx, dz: f.dz, Lx: f.Lx, Ly: f.Ly, top: f.top, origin: f.origin, stats: f.stats, nest, rec });
  });
  const mh: MetaHead[] = [];
  for (const id of used) {
    const m = metas.get(id); if (!m) continue;
    const { land, ...rest } = m;
    if (land) arrays[`m${id}.land`] = land;
    mh.push({ id, meta: rest, land: !!land });
  }
  const meta: ReplayHead = { kind: 'replay', title, version: 1, created: Date.now(), frames: heads, metas: mh };
  return { meta, data: packSave(meta, arrays) };
}

/** Frames and models of a replay file's container. */
export function unpackReplay(buf: ArrayBuffer): { frames: ReplayFrame[]; metas: Map<number, ReplayMeta>; title: string } {
  const { meta, arrays } = unpackSave(buf);
  if ((meta.kind as string) !== 'replay') throw new Error('這不是回放資料檔 / not a replay data file');
  const h = meta as ReplayHead;
  if (h.version !== 1) throw new Error('回放資料版本不支援 / unsupported replay data version');
  const u8 = (name: string): Uint8Array => { const a = arrays[name]; if (!(a instanceof Uint8Array)) throw new Error(`檔案缺少 ${name} / missing ${name}`); return a; };
  const q16 = (name: string, lo: number, hi: number): Q16 => { const a = u8(name); return { lo, hi, q: new Uint16Array(a.buffer, a.byteOffset, a.length >> 1) }; };
  const metas = new Map<number, ReplayMeta>();
  for (const m of h.metas) metas.set(m.id, { ...m.meta, land: m.land ? u8(`m${m.id}.land`) : null });
  const frames = h.frames.map((fh, i): ReplayFrame => {
    let nest: NestFrame | null = null;
    if (fh.nest) nest = { ...fh.nest, cloud: u8(`f${i}.ncloud`), rain: u8(`f${i}.nrain`) };
    let rec: ReplayRec | null = null;
    const r = fh.rec;
    if (r) {
      const maps: Record<string, Q16> = {};
      for (const m of r.maps) maps[m.name] = q16(`f${i}.m.${m.name}`, m.lo, m.hi);
      const state = r.state.length ? r.state.map((s, n) => q16(`f${i}.s.${n}`, s.lo, s.hi)) : null;
      let aux: ReplayAux | null = null;
      if (r.aux && r.vel) {
        const f32 = (k: string): Float32Array | null => (r.aux!.includes(k) ? (arrays[`f${i}.a.${k}`] as Float32Array) : null);
        aux = { rain: f32('rain')!, snow: f32('snow')!, rate: f32('rate'), cu: f32('cu'), uh: f32('uh'), wind: f32('wind'), sst: f32('sst'), vel: r.vel };
      }
      rec = { tier: r.tier, metaId: r.metaId, aux, maps, state, centre: r.centre };
    }
    return { t: fh.t, nx: fh.nx, ny: fh.ny, nz: fh.nz, dx: fh.dx, dz: fh.dz, Lx: fh.Lx, Ly: fh.Ly, top: fh.top, cloud: u8(`f${i}.cloud`), rain: u8(`f${i}.rain`), ground: u8(`f${i}.ground`),
      storms: (fh.stats.storms ?? []) as StormNow[], nest, stats: fh.stats, origin: fh.origin, rec };
  });
  return { frames, metas, title: h.title };
}

/** Bytes of the recordings of the frames (the part beyond the 3-D view). */
export const recordingBytes = (frames: ReplayFrame[]): number => frames.reduce((a, f) => a + recBytes(f.rec), 0);
