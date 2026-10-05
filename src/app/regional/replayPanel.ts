// The replay panel: recording settings (interval, content, memory, what to do when full), the state of the store, and the
// export / load of replay data as a file of its own (replayFile.ts).
import type { ReplayMeta, ReplayTier } from '../../regional/replayData.js';
import { exportSave, fmtBytes, paint, readZip, type SaveRecord } from '../saves.js';
import type { ToRegionalWorker } from './protocol.js';
import type { FullPolicy, ReplayFrame, ReplayStore } from './replay.js';
import { chooseFrames, estimateBytes, packReplay, tierOf, unpackReplay } from './replayFile.js';

export interface RecSettings { on: boolean; tier: ReplayTier; /** model seconds */ every: number; /** MB */ mem: number; full: FullPolicy }
const KEY = 'regional-replay';
/** the default interval (model seconds): tropical cyclones change slowly, convective storms quickly */
const EVERY = { tc: 600, storm: 60 };
const TIER_LABEL: Record<ReplayTier, string> = { view: '3D 畫面', maps: '3D + 合成圖', full: '完整' };

export interface PanelDeps {
  $: <T extends HTMLElement>(id: string) => T;
  send: (m: ToRegionalWorker) => void;
  log: (s: string) => void;
  live: ReplayStore; file: ReplayStore;
  fileMode: () => boolean;
  /** the step size now (s) */
  step: () => number;
  /** the model's grid (the cost of a frame is estimated from it) */
  grid: () => { nx: number; ny: number; nz: number } | null;
  /** model time (s) of the frame being replayed, null when not replaying */
  replayTime: () => number | null;
  openReplay: () => void;
  /** a replay file was read: show it */
  loaded: (frames: ReplayFrame[], metas: Map<number, ReplayMeta>, title: string) => void;
  /** the live frames are cleared by the panel's clear button */
  cleared: () => void;
}

const fmtSpan = (s: number): string => (s >= 7200 * 3 ? `${(s / 3600).toFixed(1)} h` : s >= 120 ? `${(s / 60).toFixed(s % 60 ? 1 : 0)} min` : `${s.toFixed(0)} s`);

export function mountReplayPanel(d: PanelDeps): { settings: RecSettings; apply(): void; refresh(): void; onReady(tc: boolean, axi: boolean): void; onFull(): void } {
  const $ = d.$, on = $<HTMLInputElement>('rcOn'), every = $<HTMLInputElement>('rcEvery'), unit = $<HTMLSelectElement>('rcUnit'), tier = $<HTMLSelectElement>('rcTier'),
    mem = $<HTMLInputElement>('rcMem'), full = $<HTMLSelectElement>('rcFull');
  let stored: Record<string, unknown> = {};
  try { stored = JSON.parse(localStorage.getItem(KEY) ?? '{}') as Record<string, unknown>; } catch { stored = {}; }
  const save = (): void => { try { localStorage.setItem(KEY, JSON.stringify(stored)); } catch { /* storage unavailable */ } };
  const defMem = Math.min(400, 64 * ((navigator as unknown as { deviceMemory?: number }).deviceMemory ?? 4));
  let tcNow = false, axiNow = false;
  const settings: RecSettings = { on: stored.on !== false, tier: (stored.tier as ReplayTier) ?? 'maps', every: EVERY.storm, mem: Number(stored.mem) || defMem, full: (stored.full as FullPolicy) ?? 'thin' };

  const setEvery = (s: number): void => {
    const u = s % 3600 === 0 ? 3600 : s % 60 === 0 ? 60 : 1;
    unit.value = String(u); every.value = String(+(s / u).toFixed(2));
  };
  const readEvery = (): number => { const v = Number(every.value) * Number(unit.value); return Number.isFinite(v) && v >= 1 ? v : settings.every; };
  const put = (): void => { on.checked = settings.on; tier.value = settings.tier; mem.value = String(settings.mem); full.value = settings.full; setEvery(settings.every); };

  /** Apply the inputs: the worker records by them, the store keeps within them. */
  const apply = (): void => {
    settings.on = on.checked; settings.tier = tier.value as ReplayTier; settings.every = readEvery();
    settings.mem = Math.max(50, Math.min(8000, Number(mem.value) || defMem)); settings.full = full.value as FullPolicy;
    stored.on = settings.on; stored.tier = settings.tier; stored.mem = settings.mem; stored.full = settings.full; stored[tcNow ? 'everyTc' : 'everyStorm'] = settings.every; save();
    d.live.budget = settings.mem * 1e6; d.live.policy = settings.full; d.live.full = false;
    d.file.budget = settings.mem * 1e6;
    d.send({ type: 'record', on: settings.on, tier: settings.tier, every: settings.every });
    refresh();
  };
  for (const el of [on, every, unit, tier, mem, full]) el.onchange = apply;

  /** bytes of a frame the settings record on the current grid (about: 11 state fields, 37 maps) */
  const frameCost = (): number => {
    const g = d.grid(); if (!g) return 0;
    const n2 = g.nx * g.ny, n3 = n2 * g.nz, q = n3 > 1.2e6 ? 0.25 : 1;
    let b = (2 * n3 + 4 * n2) * q + 4000;
    if (settings.tier !== 'view') b += 37 * n2 * 2;
    if (settings.tier === 'full') b += 11 * (g.nx + 6) * (g.ny + 6) * (g.nz + 1) * 2 + 7 * 4 * n2;
    return b;
  };

  // ---- export segments
  let segs: { t0: number; t1: number }[] = [], segsAuto = true;
  const store = (): ReplayStore => (d.fileMode() ? d.file : d.live);
  const range = (): { t0: number; t1: number } | null => { const f = store().frames; return f.length ? { t0: f[0]!.t, t1: f[f.length - 1]!.t } : null; };
  const segHost = $('rxSegs'), info = $('rxInfo'), status = $('rxStatus');
  const stride = $<HTMLInputElement>('rxStride'), xtier = $<HTMLSelectElement>('rxTier');
  let rows = '';
  const renderSegs = (): void => {
    const r = range();
    if (segsAuto && r) segs = [{ t0: r.t0, t1: r.t1 }];
    const u = r && r.t1 < 6 * 3600 ? 60 : 3600, un = u === 60 ? '分鐘 min' : '小時 h';
    const key = JSON.stringify([segs, u]);
    if (key === rows) return;
    rows = key;
    segHost.replaceChildren();
    segs.forEach((s, n) => {
      const row = document.createElement('div'); row.className = 'controls';
      const a = document.createElement('input'), b = document.createElement('input'), sp = document.createElement('span');
      for (const el of [a, b]) { el.type = 'number'; el.step = '0.05'; el.style.width = '5.5em'; }
      a.value = String(+(s.t0 / u).toFixed(2)); b.value = String(+(s.t1 / u).toFixed(2));
      sp.textContent = `～  ${un}`; sp.style.alignSelf = 'center';
      const edit = (): void => { const t0 = Number(a.value) * u, t1 = Number(b.value) * u; if (Number.isFinite(t0) && Number.isFinite(t1)) { segsAuto = false; segs[n] = { t0: Math.min(t0, t1), t1: Math.max(t0, t1) }; rows = ''; info_(); } };
      a.onchange = edit; b.onchange = edit;
      const mk = (label: string, title: string, f: () => void): HTMLButtonElement => { const x = document.createElement('button'); x.textContent = label; x.title = title; x.onclick = f; return x; };
      const here = mk('起=目前', '起點設為正在回放的位置 / start at the replay position', () => { const t = d.replayTime(); if (t !== null) { a.value = String(+(t / u).toFixed(2)); edit(); } });
      const there = mk('終=目前', '終點設為正在回放的位置 / end at the replay position', () => { const t = d.replayTime(); if (t !== null) { b.value = String(+(t / u).toFixed(2)); edit(); } });
      const del = mk('✕', '刪除這個區段 / remove', () => { segsAuto = false; segs.splice(n, 1); rows = ''; info_(); });
      row.append(a, b, sp, here, there, del);
      segHost.append(row);
    });
  };
  const options = (): { segments: { t0: number; t1: number }[]; tier: ReplayTier; stride: number } => ({ segments: segs, tier: xtier.value as ReplayTier, stride: Number(stride.value) || 1 });
  /** the frames the export would hold */
  const chosen = (): ReplayFrame[] => chooseFrames(store().frames, options());
  function info_(): void {
    renderSegs();
    const fr = chosen(), n = fr.length, bytes = estimateBytes(fr);
    info.textContent = n ? `匯出 ${n} 格，約 ${fmtBytes(bytes)}（壓縮前；實際檔案通常小 2–3 倍）${bytes > 1.5e9 ? ' ⚠ 太大了，請縮小區段、少取幾格或降低內容 / too large: narrow the segments, take fewer frames or less content' : ''} / ${n} frames` : '沒有符合的畫面 / no frames in the segments';
  }
  $('rxAdd').onclick = (): void => { const r = range(); segsAuto = false; segs.push(r ? { t0: r.t0, t1: r.t1 } : { t0: 0, t1: 3600 }); rows = ''; info_(); };
  $('rxAll').onclick = (): void => { segsAuto = true; rows = ''; info_(); };
  for (const el of [stride, xtier]) el.onchange = info_;

  // ---- store state
  function refresh(): void {
    const st = store(), n = st.length, st$ = $('rcStatus');
    if (!n) st$.textContent = d.fileMode() ? '（空）' : '尚無 / none yet';
    else {
      const f0 = st.frames[0]!, f1 = st.frames[n - 1]!, gap = n > 1 ? (f1.t - f0.t) / (n - 1) : 0;
      st$.textContent = `${n} 格 · ${st.megabytes.toFixed(0)} / ${settings.mem} MB · 間隔約 ${n > 1 ? fmtSpan(gap) : '—'} · ${fmtSpan(f0.t)}～${fmtSpan(f1.t)} · ${TIER_LABEL[tierOf(f1)]}${st.dropped ? ` · 已為了記憶體丟掉 ${st.dropped} 格 / dropped ${st.dropped}` : ''}${st.full ? ' · 已滿，停止記錄 / full: recording stopped' : ''}`;
    }
    const c = frameCost();
    const warn = d.step() > settings.every * 1.0001 && settings.on ? ' ⚠ 間隔比步長還短，每一步都會記，模擬會明顯變慢；建議間隔至少是步長的幾倍 / the interval is shorter than the step: every step is recorded and the run slows down' : '';
    if (axiNow) { $('rcCost').textContent = '軸對稱模式只記錄 3D 畫面（它的圖表直接由 1 維狀態算，沒有 3D 狀態可存）/ the axisymmetric model records its 3-D view only'; info_(); return; }
    const tiny = c && settings.mem * 1e6 < 4 * c ? ' ⚠ 記憶體上限太小（不到 4 格），請調高或降低記錄內容 / the memory limit holds fewer than 4 frames: raise it or record less' : '';
    $('rcCost').textContent = c ? `每格約 ${fmtBytes(c)}（${TIER_LABEL[settings.tier]}）· ${settings.mem} MB 約可記 ${Math.floor(settings.mem * 1e6 / c)} 格 = ${fmtSpan(Math.floor(settings.mem * 1e6 / c) * settings.every)} 的模式時間 / about ${fmtBytes(c)} per frame${warn}${tiny}` : '';
    info_();
  }

  // ---- buttons, jobs
  let working = false;
  const buttons = (): HTMLButtonElement[] => ['rxGo', 'rxAdd', 'rxAll', 'rcClear'].map((id) => $<HTMLButtonElement>(id));
  const fileIn = $<HTMLInputElement>('rpFile');
  const job = async (what: string, f: (step: (s: string) => void) => Promise<string | void>): Promise<void> => {
    if (working) return;
    working = true; for (const b of buttons()) b.disabled = true; fileIn.disabled = true;
    const t0 = performance.now(), step = (s: string): void => { status.textContent = `${what}：${s}`; };
    status.textContent = `${what}… / working`; await paint();
    try { const r = await f(step); d.log(`${r ?? what} · ${((performance.now() - t0) / 1000).toFixed(1)} s`); }
    catch (e) { d.log(`${what}失敗 / failed: ${String((e as Error).message ?? e)}`); }
    working = false; fileIn.disabled = false; status.textContent = ''; for (const b of buttons()) b.disabled = false;
  };
  $('rxGo').onclick = (): void => {
    void job('匯出回放資料', async (step) => {
      const frames = chosen();
      if (!frames.length) throw new Error('沒有符合的畫面 / no frames in the segments');
      if (estimateBytes(frames) > 1.5e9) throw new Error('資料太大，請縮小區段、少取幾格或降低內容 / too large: narrow the segments, take fewer frames or less content');
      step(`打包 ${frames.length} 格… / packing`); await paint();
      const st = store(), exp = [...st.metas.values()][0]?.experiment ?? 'regional';
      const title = `回放 ${exp} ${fmtSpan(frames[0]!.t)}-${fmtSpan(frames[frames.length - 1]!.t)}`;
      const { meta, data } = packReplay(frames, st.metas, title);
      const rec: SaveRecord = { id: '', kind: 'replay', title: meta.title, created: Date.now(), bytes: data.byteLength, data };
      return `${await exportSave(rec, step)}: ${title}`;
    });
  };
  fileIn.onchange = (): void => {
    const f = fileIn.files?.[0]; fileIn.value = '';
    if (!f) return;
    void job('載入回放資料', async (step) => {
      step(`讀取檔案（${fmtBytes(f.size)}）… / reading`);
      const data = await readZip(await f.arrayBuffer());
      step('整理畫面… / unpacking'); await paint();
      const { frames, metas, title } = unpackReplay(data);
      if (!frames.length) throw new Error('檔案裡沒有畫面 / the file holds no frames');
      d.loaded(frames, metas, title);
      return `已載入 / loaded: ${title}（${frames.length} 格）`;
    });
  };
  $('rcClear').onclick = (): void => { d.live.clear(); d.cleared(); refresh(); };
  $('rcNow').onclick = (): void => { if (!d.fileMode()) d.send({ type: 'record', on: true, tier: settings.tier, every: settings.every, now: true }); };
  $('rcOpen').onclick = (): void => d.openReplay();

  put();
  return {
    settings, apply, refresh,
    /** a new model: its default interval (the user's, when set), recording settings to the worker */
    onReady(tc: boolean, axi: boolean): void {
      tcNow = tc; axiNow = axi;
      const own = Number(stored[tc ? 'everyTc' : 'everyStorm']);
      settings.every = own >= 1 ? own : tc ? EVERY.tc : EVERY.storm;
      put(); d.live.budget = settings.mem * 1e6; d.live.policy = settings.full; d.live.full = false;
      d.send({ type: 'record', on: settings.on, tier: settings.tier, every: settings.every });
      refresh();
    },
    /** the store refused a frame ('stop' policy): recording stops */
    onFull(): void {
      d.send({ type: 'record', on: false, tier: settings.tier, every: settings.every });
      on.checked = false; settings.on = false;
      d.log(`回放記憶體已滿（${settings.mem} MB），已停止記錄；可以調高上限或改用其他處理方式 / replay memory is full: recording stopped`);
      refresh();
    },
  };
}
