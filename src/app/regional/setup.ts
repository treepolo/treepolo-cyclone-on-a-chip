// Experiment set-up of the regional model: every condition of a run in one record. The former experiments are
// presets (ready-made sets of conditions); any field can be changed before a run.

export type Sounding = 'tropical' | 're87' | 'wk82';
export type EnvWind = 'calm' | 'trade' | 'shear' | 'quarter';
export type Radiation = 'const' | 'relax' | 'none';
export type InitKind = 'vortex' | 'bubble' | 'none';

export interface RegionalSetup {
  /** preset it started from (id in PRESETS) */
  preset: string;
  /** domain width (m, square), grid spacing (m), model top (m), vertical spacing (m); time step (s, 0 = automatic) */
  L: number; dx: number; top: number; dz: number; dt: number;
  /** lateral boundaries: periodic (what leaves one side enters the other) or open (relaxing to the environment) */
  boundary: 'periodic' | 'open';
  /** the domain moves with the strongest storm */
  follow: boolean;
  /** latitude (degrees; sets the Coriolis parameter, 0 = no rotation) */
  lat: number;
  /** surface: sea (fixed sea-surface temperature) or land (drier, rougher, ground at the air temperature) */
  surface: 'sea' | 'land';
  /** sea-surface temperature (°C) */
  sst: number;
  /** surface fluxes of heat, moisture and momentum */
  fluxes: boolean;
  /** thermodynamic sounding: conditionally unstable tropical, neutral RE87 tropical, or Weisman-Klemp (1982) */
  sounding: Sounding;
  /** boundary-layer water-vapour cap of the WK82 sounding (g/kg) */
  qvBL: number;
  /** environmental wind: none, trade easterly, straight-line shear, quarter-circle hodograph */
  wind: EnvWind;
  /** trade: easterly speed below 3 km (m/s); shear: 0-6 km speed change; quarter: wind at 6 km */
  windU: number;
  /** quarter circle: radius (m/s) and curved-segment depth (m) */
  windR: number; windDepth: number;
  /** radiation: constant clear-sky tropospheric cooling, relaxation to the sounding (RE87), or none; rate or cap (K/day) */
  radiation: Radiation; radRate: number;
  /** minimum wind in the surface fluxes (m/s), surface gustiness, boundary-layer perturbations (K per 10 min) */
  vmin: number; gust: boolean; blNoise: number;
  /** scale-aware cumulus parameterization (sub-grid convection on grids coarser than 3 km, full from 12 km) */
  cumulus: boolean;
  /** initial disturbance: balanced warm-core vortex (vmax m/s), warm bubble (K), or none */
  init: InitKind; initAmp: number;
}

/** Tropical sea: conditionally unstable sounding, constant cooling, gustiness, trade wind off. */
const TROPICAL: Omit<RegionalSetup, 'preset' | 'L' | 'dx' | 'top' | 'dz' | 'dt'> = {
  boundary: 'periodic', follow: false, lat: 20, surface: 'sea', sst: 28, fluxes: true, sounding: 'tropical', qvBL: 16,
  wind: 'calm', windU: 5, windR: 12, windDepth: 1000, radiation: 'const', radRate: 1.5, vmin: 1, gust: true, blNoise: 0.1, cumulus: true, init: 'vortex', initAmp: 15,
};
/** Continental convective-storm environment: WK82 sounding, no surface fluxes or radiation. */
const STORM: Omit<RegionalSetup, 'preset' | 'L' | 'dx' | 'top' | 'dz' | 'dt'> = {
  boundary: 'periodic', follow: false, lat: 0, surface: 'land', sst: 28, fluxes: false, sounding: 'wk82', qvBL: 14,
  wind: 'shear', windU: 30, windR: 12, windDepth: 1000, radiation: 'none', radRate: 1.5, vmin: 1, gust: false, blNoise: 0, cumulus: false, init: 'bubble', initAmp: 2,
};
/** Tornadic supercell: strong low-level shear, moist boundary layer, open boundaries following the storm, surface drag. */
const TORNADIC: Omit<RegionalSetup, 'preset' | 'L' | 'dx' | 'top' | 'dz' | 'dt'> = {
  ...STORM, boundary: 'open', follow: true, fluxes: true, qvBL: 16, wind: 'quarter', windU: 30, windR: 12, windDepth: 1000, initAmp: 3,
};

export interface Preset { id: string; label: string; setup: RegionalSetup; gpu?: boolean; refine?: string; axi?: boolean }
const P = (id: string, base: typeof TROPICAL, dom: Pick<RegionalSetup, 'L' | 'dx' | 'top' | 'dz' | 'dt'>): RegionalSetup => ({ preset: id, ...base, ...dom });
export const PRESETS: Preset[] = [
  { id: 'tc', label: '颱風 15 km / Tropical cyclone 15 km', setup: P('tc', TROPICAL, { L: 1200000, dx: 15000, top: 25000, dz: 1000, dt: 60 }), refine: 'tc_hr' },
  { id: 'tc_hr', label: '颱風 5 km / Tropical cyclone 5 km', setup: P('tc_hr', TROPICAL, { L: 1200000, dx: 5000, top: 25000, dz: 500, dt: 30 }), gpu: true, refine: 'tc_3' },
  { id: 'tc_3', label: '颱風 3 km（獨立顯卡）/ Tropical cyclone 3 km (discrete GPU)', setup: P('tc_3', TROPICAL, { L: 1200000, dx: 3000, top: 25000, dz: 500, dt: 18 }), gpu: true },
  { id: 'supercell', label: '超大胞 2 km / Supercell 2 km', setup: P('supercell', STORM, { L: 120000, dx: 2000, top: 20000, dz: 500, dt: 6 }), refine: 'supercell_hr' },
  { id: 'supercell_hr', label: '超大胞 1 km / Supercell 1 km', setup: P('supercell_hr', STORM, { L: 120000, dx: 1000, top: 20000, dz: 333.3333, dt: 3 }), gpu: true },
  { id: 'tornado_c', label: '龍捲超大胞 1 km（可細化到 250 m）/ Tornadic supercell 1 km', setup: P('tornado_c', TORNADIC, { L: 100000, dx: 1000, top: 16000, dz: 400, dt: 6 }), refine: 'tornado' },
  { id: 'tornado', label: '龍捲超大胞 250 m（高階顯卡）/ Tornadic supercell 250 m (fast GPU)', setup: P('tornado', TORNADIC, { L: 50000, dx: 250, top: 16000, dz: 250, dt: 2 }), gpu: true },
  { id: 'tc_axi', label: '颱風軸對稱快速版（研究用）/ Axisymmetric TC (research)', setup: P('tc_axi', TROPICAL, { L: 1600000, dx: 4000, top: 25000, dz: 500, dt: 20 }), axi: true },
];
export const presetById = (id: string): Preset | undefined => PRESETS.find((p) => p.id === id);
export const setupOf = (id: string): RegionalSetup => ({ ...(presetById(id) ?? PRESETS[0]!).setup });

/** Time step (s) picked for a grid: about 4 s per km of grid spacing and at most dz / (15 m/s), between 0.5 and 60 s. */
export const defaultDt = (s: Pick<RegionalSetup, 'dx' | 'dz'>): number => Math.max(0.5, Math.min(60, Math.round(2 * Math.min(4 * s.dx / 1000, s.dz / 15)) / 2));
/** Time step of a set-up (s): the configured one, or the default for its grid when 0. */
export const autoDt = (s: Pick<RegionalSetup, 'dx' | 'dz' | 'dt'>): number => (s.dt > 0 ? s.dt : defaultDt(s));
/** Cells of a set-up (x, y, z). */
export const setupCells = (s: RegionalSetup): number => Math.round(s.L / s.dx) ** 2 * Math.round(s.top / s.dz);
/** A tropical-cyclone-like run: surface pressure, radius of maximum wind and eyewall diagnostics apply. */
export const tcLike = (s: RegionalSetup): boolean => s.init === 'vortex';

/** Keep a set-up within what the model can handle (cell counts, spacings, ranges). */
export function sanitize(s: RegionalSetup): RegionalSetup {
  const c = (x: number, lo: number, hi: number, d: number): number => (Number.isFinite(x) ? Math.max(lo, Math.min(hi, x)) : d);
  // at most 1024 columns across and 200 levels; a configured time step at most twice the default (larger ones blow up)
  const dx = c(s.dx, 100, 50000, 15000), dz = c(s.dz, 50, 2000, 500);
  const L = c(Math.round(s.L / dx) * dx, 16 * dx, 1024 * dx, 1200000), top = c(Math.round(Math.min(s.top, 200 * dz) / dz) * dz, 8 * dz, 40000, 25000);
  return { ...s, dx, dz, L, top, dt: c(s.dt, 0, 2 * defaultDt({ dx, dz }), 0), lat: c(s.lat, -80, 80, 20), sst: c(s.sst, -2, 36, 28), qvBL: c(s.qvBL, 8, 20, 14),
    windU: c(s.windU, -60, 60, 5), windR: c(s.windR, 0, 40, 12), windDepth: c(s.windDepth, 250, 5500, 1000), radRate: c(s.radRate, 0, 5, 1.5),
    vmin: c(s.vmin, 0, 10, 1), blNoise: c(s.blNoise, 0, 1, 0.1), cumulus: !!s.cumulus, initAmp: c(s.initAmp, 0, s.init === 'vortex' ? 60 : 10, s.init === 'vortex' ? 15 : 2) };
}

/** Short description of a set-up (Chinese / English). */
export function describe(s: RegionalSetup): string {
  const km = (m: number): string => (m >= 1000 ? `${+(m / 1000).toFixed(2)} km` : `${m} m`);
  const snd = { tropical: '熱帶不穩定 / unstable tropical', re87: 'RE87 中性 / neutral', wk82: `WK82（${s.qvBL} g/kg）` }[s.sounding];
  const wind = { calm: '無風 / calm', trade: `信風 ${s.windU} m/s / trade`, shear: `風切 ${s.windU} m/s / shear`, quarter: `四分之一圓 ${s.windR}–${s.windU} m/s / quarter circle` }[s.wind];
  return `${km(s.L)} × ${km(s.top)}，Δx ${km(s.dx)}，${s.boundary === 'periodic' ? '週期邊界 / periodic' : '開放邊界 / open'}${s.follow ? '，跟隨風暴 / following' : ''}，` +
    `${+s.lat.toFixed(1)}°，${s.surface === 'sea' ? `海 ${+s.sst.toFixed(1)} °C / sea` : '陸地 / land'}，${snd}，${wind}${s.cumulus && s.dx > 3000 ? '，積雲參數化 / cumulus scheme' : ''}`;
}
