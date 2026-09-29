// Set-up form of the regional page: presets and every condition of a run (setup.ts), generated from one field table.
import { PRESETS, autoDt, presetById, sanitize, setupCells, type RegionalSetup } from './setup.js';
import { tcSounding } from '../../regional/tropical.js';
import { weismanKlempQ } from '../../regional/kessler.js';
import { quarterCircleWind, bunkersRightMover } from '../../regional/supercell.js';
import { parcelAscent } from '../../regional/diagnostics.js';

type Key = Exclude<keyof RegionalSetup, 'preset'>;
interface Field {
  key: Key; label: string; group: 'grid' | 'env' | 'init';
  /** number input (value shown = setup value / scale), select or checkbox */
  kind: 'num' | 'sel' | 'chk'; scale?: number; step?: number; options?: [string, string][];
  /** shown only when this holds for the current values */
  show?: (s: RegionalSetup) => boolean;
  /** label depending on the current values */
  dyn?: (s: RegionalSetup) => string;
  tip?: string;
}
const notAxi = (s: RegionalSetup): boolean => !presetById(s.preset)?.axi;
const FIELDS: Field[] = [
  { key: 'L', label: '範圍 / Width (km)', group: 'grid', kind: 'num', scale: 1000, step: 10, show: notAxi },
  { key: 'dx', label: '水平格距 / Δx (km)', group: 'grid', kind: 'num', scale: 1000, step: 0.25, tip: '越細越真實也越慢：格點數 ∝ 1/Δx² / finer is more realistic and slower: cells grow as 1/Δx²' },
  { key: 'top', label: '模式頂 / Top (km)', group: 'grid', kind: 'num', scale: 1000, step: 1, show: notAxi },
  { key: 'dz', label: '垂直格距 / Δz (m)', group: 'grid', kind: 'num', step: 50, show: notAxi },
  { key: 'dt', label: '時間步長 / Δt (s, 0 = 自動 auto)', group: 'grid', kind: 'num', step: 1, show: notAxi, tip: '0 = 依格距自動選擇；太大會數值發散 / 0 picks one from the grid spacing; too large blows up' },
  { key: 'boundary', label: '側邊界 / Boundaries', group: 'grid', kind: 'sel', options: [['periodic', '週期 / periodic'], ['open', '開放 / open']], show: notAxi,
    tip: '週期：從一邊出去的空氣從對邊回來；開放：邊緣向環境場鬆弛 / periodic: what leaves one side enters the other; open: the edges relax to the environment' },
  { key: 'follow', label: '計算範圍跟隨風暴 / Domain follows the storm', group: 'grid', kind: 'chk', show: notAxi },
  { key: 'lat', label: '緯度 / Latitude (°)', group: 'env', kind: 'num', step: 1, tip: '決定科氏力；0° 沒有地球自轉效應 / sets the Coriolis force; 0° has no rotation' },
  { key: 'surface', label: '地表 / Surface', group: 'env', kind: 'sel', options: [['sea', '海洋 / sea'], ['land', '陸地 / land']], show: notAxi },
  { key: 'sst', label: '海溫 / Sea temperature (°C)', group: 'env', kind: 'num', step: 0.5, show: (s) => s.surface === 'sea' || !notAxi(s) },
  { key: 'fluxes', label: '地面熱量、水氣與摩擦 / Surface fluxes', group: 'env', kind: 'chk', show: notAxi },
  { key: 'sounding', label: '探空 / Sounding', group: 'env', kind: 'sel', options: [['tropical', '熱帶不穩定 / unstable tropical'], ['re87', 'RE87 中性 / neutral'], ['wk82', 'Weisman–Klemp 1982']],
    tip: '熱帶不穩定：CAPE 約 1000 J/kg，外圍會長對流；RE87：CAPE 0，只有眼牆對流；WK82：中緯度強對流環境 / unstable tropical: CAPE about 1000 J/kg, outer convection grows; RE87: zero CAPE, eyewall only; WK82: mid-latitude severe-storm environment' },
  { key: 'qvBL', label: '邊界層水氣上限 / BL vapour cap (g/kg)', group: 'env', kind: 'num', step: 0.5, show: (s) => s.sounding === 'wk82' },
  { key: 'wind', label: '環境風 / Environmental wind', group: 'env', kind: 'sel', show: notAxi,
    options: [['calm', '無風 / calm'], ['trade', '信風（東風）/ trade easterly'], ['shear', '直線風切 / straight-line shear'], ['quarter', '四分之一圓風徑圖 / quarter-circle hodograph']] },
  { key: 'windU', label: '風速 / Wind (m/s)', group: 'env', kind: 'num', step: 1, show: (s) => notAxi(s) && s.wind !== 'calm',
    dyn: (s) => (s.wind === 'trade' ? '3 km 以下東風 / easterly below 3 km (m/s)' : s.wind === 'shear' ? '0–6 km 風速差 / 0–6 km shear (m/s)' : '6 km 風速 / wind at 6 km (m/s)') },
  { key: 'windR', label: '風徑圖半徑 / Hodograph radius (m/s)', group: 'env', kind: 'num', step: 1, show: (s) => notAxi(s) && s.wind === 'quarter' },
  { key: 'windDepth', label: '彎曲段深度 / Curved depth (km)', group: 'env', kind: 'num', scale: 1000, step: 0.25, show: (s) => notAxi(s) && s.wind === 'quarter' },
  { key: 'radiation', label: '輻射 / Radiation', group: 'env', kind: 'sel', options: [['const', '固定冷卻 / constant cooling'], ['relax', '向探空鬆弛 / relax (RE87)'], ['none', '無 / none']] },
  { key: 'radRate', label: '冷卻率 / Cooling (K/day)', group: 'env', kind: 'num', step: 0.25, show: (s) => s.radiation !== 'none' },
  { key: 'vmin', label: '通量最小風速 / Min. flux wind (m/s)', group: 'env', kind: 'num', step: 0.5, show: (s) => s.fluxes || !notAxi(s) },
  { key: 'gust', label: '陣風（對流與降雨）/ Gustiness', group: 'env', kind: 'chk', show: (s) => notAxi(s) && s.fluxes,
    tip: '無風時仍有亂流與下衝流造成的陣風，讓海面照常蒸發 / turbulent and downdraft gusts keep the surface fluxes going in light wind' },
  { key: 'blNoise', label: '邊界層擾動 / BL perturbations (K)', group: 'env', kind: 'num', step: 0.05, show: (s) => notAxi(s) && (s.fluxes || s.radiation !== 'none'),
    tip: '每 10 分鐘在 1 km 以下加入 ± 這麼多的隨機溫度擾動（代表次網格亂流）/ random temperature noise below 1 km every 10 minutes (sub-grid turbulence)' },
  { key: 'init', label: '初始擾動 / Initial disturbance', group: 'init', kind: 'sel', options: [['vortex', '暖心渦旋 / warm-core vortex'], ['bubble', '暖泡 / warm bubble'], ['none', '無 / none']], show: notAxi },
  { key: 'initAmp', label: '強度 / Strength', group: 'init', kind: 'num', step: 0.5, show: (s) => s.init !== 'none',
    dyn: (s) => (s.init === 'vortex' ? '渦旋最大風 / vortex max wind (m/s)' : '暖泡溫度 / bubble (K)') },
];

/** Largest grid the form accepts (cells). */
export const MAX_CELLS = 30e6;

export class SetupForm {
  private cur: RegionalSetup;
  private readonly rows = new Map<Key, { row: HTMLElement; text: HTMLElement; input: HTMLInputElement | HTMLSelectElement; f: Field }>();
  /** the page applies a set-up (restart) */
  onApply: (s: RegionalSetup) => void = () => {};
  /** the preset menu switched to the nest in the global model */
  onNest: () => void = () => {};

  constructor(private readonly presetSel: HTMLSelectElement, groups: Record<Field['group'], HTMLElement>, private readonly info: HTMLElement, private readonly applyBtn: HTMLButtonElement, resetBtn: HTMLButtonElement) {
    for (const p of PRESETS) presetSel.append(new Option(p.label, p.id));
    presetSel.append(new Option('自訂（沿用目前條件）/ Custom', 'custom'));
    this.cur = { ...PRESETS[0]!.setup };
    for (const f of FIELDS) {
      const row = document.createElement('label'); row.className = f.kind === 'chk' ? 'f chk' : 'f';
      const text = document.createElement('span'); text.textContent = f.label; if (f.tip) row.title = f.tip;
      let input: HTMLInputElement | HTMLSelectElement;
      if (f.kind === 'sel') { const s = document.createElement('select'); for (const [v, t] of f.options!) s.append(new Option(t, v)); input = s; }
      else { const i = document.createElement('input'); i.type = f.kind === 'chk' ? 'checkbox' : 'number'; if (f.step) i.step = String(f.step); input = i; }
      input.addEventListener(f.kind === 'num' ? 'input' : 'change', () => this.changed());
      if (f.kind === 'chk') row.append(input, text); else row.append(text, input);
      groups[f.group].append(row);
      this.rows.set(f.key, { row, text, input, f });
    }
    presetSel.addEventListener('change', () => {
      const v = presetSel.value;
      if (v === 'nest') { this.onNest(); return; }
      if (v === 'custom') { this.changed(); return; }
      this.set({ ...presetById(v)!.setup });
      this.onApply(this.value());
    });
    applyBtn.addEventListener('click', () => { const s = this.value(); if (setupCells(s) <= MAX_CELLS) this.onApply(s); });
    resetBtn.addEventListener('click', () => { const p = presetById(this.cur.preset); if (p) this.set({ ...p.setup }); });
    this.set(this.cur);
  }

  /** Show a set-up (from a preset, or the one a run was built with). */
  set(s: RegionalSetup): void {
    this.cur = { ...s };
    this.presetSel.value = presetById(s.preset) ? s.preset : 'custom';
    for (const [k, r] of this.rows) {
      const v = s[k];
      if (r.f.kind === 'chk') (r.input as HTMLInputElement).checked = !!v;
      else if (r.f.kind === 'sel') r.input.value = String(v);
      else r.input.value = String(+((v as number) / (r.f.scale ?? 1)).toFixed(4));
    }
    this.changed(false);
  }

  /** Select the nest entry (added when the page is embedded by the global model). */
  addNest(select: boolean): void {
    if (!Array.from({ length: this.presetSel.options.length }, (_, i) => this.presetSel.options[i]!.value).includes('nest')) this.presetSel.prepend(new Option('全球模式巢狀區域 / Nest in the global model', 'nest'));
    if (select) this.presetSel.value = 'nest';
    this.showFields();
  }
  get isNest(): boolean { return this.presetSel.value === 'nest'; }

  /** The set-up in the form (sanitized). Grid changes turn a preset into a custom set-up. */
  value(): RegionalSetup {
    const s: RegionalSetup = { ...this.cur };
    const w = s as unknown as Record<string, unknown>;
    for (const [k, r] of this.rows) {
      if (r.f.kind === 'chk') w[k] = (r.input as HTMLInputElement).checked;
      else if (r.f.kind === 'sel') w[k] = r.input.value;
      else { const x = Number(r.input.value); w[k] = Number.isFinite(x) ? x * (r.f.scale ?? 1) : (this.cur[k] as number); }
    }
    const p = presetById(s.preset), out = sanitize(s);
    if (p && !p.axi && (['L', 'dx', 'top', 'dz', 'dt'] as const).some((k) => Math.abs(out[k] - p.setup[k]) > 1e-6)) out.preset = 'custom';
    return out;
  }

  private showFields(): void {
    const nest = this.isNest, s = this.value();
    for (const r of this.rows.values()) {
      r.row.hidden = nest || !(r.f.show?.(s) ?? true);
      if (r.f.dyn) r.text.textContent = r.f.dyn(s);
    }
  }

  private changed(edited = true): void {
    this.showFields();
    if (this.isNest) { this.info.textContent = '巢狀區域的條件來自全球模式 / the nest takes its conditions from the global model'; this.applyBtn.disabled = true; return; }
    const s = this.value(), cells = setupCells(s), nz = Math.round(s.top / s.dz), nx = Math.round(s.L / s.dx);
    const parts = [presetById(s.preset)?.axi ? `軸對稱 / axisymmetric, Δr ${Math.max(1, Math.min(4, s.dx / 1000))} km` : `${nx}×${nx}×${nz} = ${(cells / 1e6).toFixed(cells < 1e6 ? 2 : 1)} M 格點 / cells · Δt ${autoDt(s)} s`, envText(s)];
    if (!presetById(s.preset)?.axi) {
      if (cells > MAX_CELLS) parts.push(`⚠ 太大（上限 ${MAX_CELLS / 1e6} M）/ too large (limit ${MAX_CELLS / 1e6} M)`);
      else if (cells > 10e6) parts.push('⚠ 可能超過顯示卡記憶體 / may exceed the GPU memory');
      else if (cells > 1.5e6) parts.push('CPU 會很慢，需要 WebGPU / slow on the CPU, needs WebGPU');
    }
    if (edited) parts.push('按「套用」才會生效 / press Apply to use');
    this.info.textContent = parts.filter(Boolean).join(' · ');
    this.applyBtn.disabled = cells > MAX_CELLS && !presetById(s.preset)?.axi;
    if (edited && this.presetSel.value !== 'custom' && s.preset === 'custom') this.presetSel.value = 'custom';
  }
}

/** CAPE, LCL and (quarter-circle hodographs) storm-relative helicity of a set-up's environment. */
function envText(s: RegionalSetup): string {
  const snd = s.sounding === 'wk82' ? weismanKlempQ(s.qvBL / 1000) : tcSounding(s.sounding === 're87' ? 're87' : 'unstable', s.sst + 273.15);
  const nz = 80, dz = 250, T = new Float64Array(nz), p = new Float64Array(nz), q = new Float64Array(nz);
  // the soundings are tabulated as theta and qv: integrate the Exner function hydrostatically from 1000 hPa
  let pi = 1;
  for (let k = 0; k < nz; k++) {
    const z = (k + 0.5) * dz, a = snd(z);
    if (k > 0) { const b = snd(z - dz); pi -= 9.80665 / (1004.5 * 0.5 * (a.theta * (1 + 0.61 * a.qv) + b.theta * (1 + 0.61 * b.qv))) * dz; }
    else pi -= 9.80665 / (1004.5 * a.theta * (1 + 0.61 * a.qv)) * 0.5 * dz;
    T[k] = a.theta * pi; p[k] = 1e5 * Math.pow(pi, 1004.5 / 287.05); q[k] = a.qv;
  }
  const pc = parcelAscent(T, p, q, dz);
  let t = `CAPE ${pc.cape.toFixed(0)} J/kg · LCL ${pc.lcl >= 0 ? ((pc.lcl + 0.5) * dz / 1000).toFixed(1) : '—'} km`;
  if (s.wind === 'quarter' && !presetById(s.preset)?.axi) {
    const h = { R: s.windR, U6: s.windU, depth: s.windDepth }, rm = bunkersRightMover((z) => quarterCircleWind(z, h));
    const srh = (top: number): number => { let a = 0; const n = 200; for (let i = 0; i < n; i++) { const u = quarterCircleWind(i * top / n, h), v = quarterCircleWind((i + 1) * top / n, h); a += (v.u - rm.u) * (u.v - rm.v) - (u.u - rm.u) * (v.v - rm.v); } return a; };
    t += ` · SRH 0–1 km ${srh(1000).toFixed(0)}、0–3 km ${srh(3000).toFixed(0)} m²/s²`;
  }
  return t;
}
