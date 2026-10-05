// Fixed colour scales and contour / arrow references of the regional charts. A scale never follows the data of the
// moment: a chart must read the same at every time and in every run, so that two pictures can be compared. Each variable
// has a range for tropical-cyclone runs and one for convective-storm runs (their magnitudes differ by a factor of ten
// or more); diverging variables are symmetric about zero (the larger absolute value counts). The user can set another
// range per variable (kept in this browser): it is fixed as well.

/** [lo, hi] of a sequential variable, [-a, a] of a diverging one */
type Range = [number, number];
const R = (tc: Range, storm: Range = tc): { tc: Range; storm: Range } => ({ tc, storm });
const pm = (a: number): Range => [-a, a];

export const SCALES: Record<string, { tc: Range; storm: Range }> = {
  // horizontal slices and cross-sections (section 'along' and 'normal' wind like u, v)
  w: R(pm(6), pm(30)), speed: R([0, 80], [0, 50]), u: R(pm(80), pm(40)), v: R(pm(80), pm(40)), along: R(pm(80), pm(40)), normal: R(pm(80), pm(40)),
  thp: R(pm(12), pm(10)), thetaE: R([320, 380]), rh: R([0, 100]), zeta: R(pm(3e-3), pm(0.03)), div: R(pm(100), pm(400)), pp: R(pm(30), pm(8)),
  qv: R([0, 20]), cloud: R([0, 3], [0, 5]), precip: R([0, 8], [0, 15]), T: R([-80, 40]),
  // radius-height means
  'rz:vt': R(pm(80), pm(40)), 'rz:vr': R(pm(30), pm(20)), 'rz:w': R(pm(3), pm(15)), 'rz:thp': R(pm(12), pm(10)), 'rz:cond': R([0, 2], [0, 4]),
  // composite and surface maps
  ctopZ: R([0, 18], [0, 16]), uh: R(pm(150), pm(500)), wMax: R([0, 15], [0, 60]), rainRate: R([0, 100], [0, 200]), cuRain: R([0, 30]),
  rain: R([0, 400], [0, 100]), snow: R([0, 50], [0, 30]), slp: R([900, 1020], [990, 1020]), sfcWind: R([0, 80], [0, 40]), sfcThp: R(pm(5), pm(10)),
  sfcThetaE: R([320, 380]), cape: R([0, 3000], [0, 5000]), cin: R([-300, 0]), pw: R([0, 80]), sfcT: R([20, 34], [0, 40]), sfcTd: R([15, 30], [0, 25]),
  shear06: R([0, 40]), srh01: R(pm(300), pm(500)), srh03: R(pm(600)), lcl: R([0, 3]), li: R(pm(10)), stp: R([0, 5]), scp: R(pm(15)),
  etop: R([0, 18], [0, 16]), vil: R([0, 40], [0, 80]), uhSwath: R([0, 100], [0, 500]), windSwath: R([0, 80], [0, 40]),
  sst: R([22, 34], [10, 35]), shear850200: R([0, 40]), shear01: R([0, 25]),
};

const KEY = 'regional-scale-overrides';
let overrides: Record<string, Range> = {};
try { overrides = JSON.parse(localStorage.getItem(KEY) ?? '{}') as Record<string, Range>; } catch { overrides = {}; }
const okey = (name: string, tc: boolean): string => `${name}|${tc ? 'tc' : 'storm'}`;

/** The fixed range of a variable: the user's, or the default of the experiment kind. */
export function scaleRange(name: string, tc: boolean): Range {
  const o = overrides[okey(name, tc)];
  if (o) return o;
  const d = SCALES[name];
  return d ? (tc ? d.tc : d.storm) : [0, 1];
}
/** The default range (without the user's). */
export function defaultRange(name: string, tc: boolean): Range { const d = SCALES[name]; return d ? (tc ? d.tc : d.storm) : [0, 1]; }
/** Set the user's range of a variable (null: back to the default). */
export function setScaleRange(name: string, tc: boolean, r: Range | null): void {
  if (r) overrides[okey(name, tc)] = r; else delete overrides[okey(name, tc)];
  try { localStorage.setItem(KEY, JSON.stringify(overrides)); } catch { /* storage unavailable */ }
}
export const hasOverride = (name: string, tc: boolean): boolean => !!overrides[okey(name, tc)];

/** Contour intervals and arrow references, fixed like the colour scales. */
export const FIXED = {
  /** reference arrow of wind arrows (m/s) in maps */
  arrowRef: (tc: boolean): number => (tc ? 60 : 30),
  /** reference of the in-plane wind vectors of a cross-section (m/s) */
  sectionRef: (tc: boolean): number => (tc ? 60 : 40),
  /** isobar interval of sea-level pressure (hPa) */
  isobar: (tc: boolean): number => (tc ? 4 : 0.5),
  /** interval of the height contours of a pressure-level chart (m) */
  heightStep: (p: number): number => ({ 925: 10, 850: 20, 700: 20, 500: 40, 300: 60, 200: 80 } as Record<number, number>)[p] ?? 40,
  /** contour interval of the other quantity drawn over a radius-height section (tangential wind, w): m/s */
  rzContour: (overVt: boolean): number => (overVt ? 10 : 0.5),
  /** radius (m/s) and ring interval of the hodograph */
  hodograph: (tc: boolean): { r: number; ring: number } => (tc ? { r: 80, ring: 20 } : { r: 40, ring: 10 }),
};
