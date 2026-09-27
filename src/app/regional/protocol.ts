// Messages between the regional-model UI and its worker.

export type RegionalExperiment = 'supercell' | 'tc' | 'supercell_hr' | 'tc_hr' | 'nest' | 'tornado';

/** Global-model state handed to the regional page for one-way nesting. Grid arrays are [k][lat][lon]. */
export interface NestPayload {
  preset: string; day: number;
  nlat: number; nlon: number; K: number;
  lat: Float64Array; lon: Float64Array; sigma: Float64Array; sigmaHalf: Float64Array;
  u: Float32Array; v: Float32Array; T: Float32Array; ps: Float32Array; q: Float32Array | null;
  phis: Float32Array;
  ts: Float32Array | null;       // surface (skin) temperature, K
  wet: Float32Array | null;      // surface wetness: 1 over sea, bucket fraction over land
  land: Uint8Array | null;
}
/** 'meso': 1200 km at 12 km (fronts, cyclones, monsoon rain bands); 'storm': 480 km at 4 km and
 *  'cp3': 960 km at 3 km (convection-permitting, GPU) */
export type NestSize = 'meso' | 'storm' | 'cp3';
export const NEST_HALF_WIDTH_KM: Record<NestSize, number> = { meso: 600, storm: 240, cp3: 480 };
export type GroundField = 'rain' | 'wind' | 'theta' | 'snow';

export type ToRegionalWorker =
  | { type: 'init'; experiment: RegionalExperiment; backend: 'auto' | 'cpu' }
  | { type: 'initNest'; payload: NestPayload; lat0: number; lon0: number; size: NestSize; backend: 'auto' | 'cpu' }
  | { type: 'run'; running: boolean }
  | { type: 'speed'; stepsPerTick: number }
  | { type: 'ground'; field: GroundField }
  /** newer global state: refresh the lateral-boundary relaxation targets of the running nest */
  | { type: 'boundary'; payload: NestPayload };

export interface RegionalFrame {
  type: 'frame';
  time: number;               // s
  nx: number; ny: number; nz: number;
  dx: number; dz: number;
  cloud: Uint8Array;          // [k][j][i] cloud water, 0..255 (scaled)
  rain: Uint8Array;           // [k][j][i] rain water, 0..255 (scaled)
  ground: Float32Array;       // [j][i]
  groundField: GroundField;
  groundRange: [number, number];
  stats: { wmax: number; wmin: number; qcmax: number; qrmax: number; rainmax: number; vmax: number; dp: number | null; rmw: number | null; eyewalls: { r: number; v: number }[] | null; zetaMax: number; vGround: number };
  stepsPerSecond: number;
}

export type FromRegionalWorker =
  | RegionalFrame
  | { type: 'ready'; experiment: RegionalExperiment; nx: number; ny: number; nz: number; dx: number; dz: number; dt: number; description: string; backend: 'cpu' | 'gpu'; note: string; land: Uint8Array | null }
  | { type: 'error'; message: string };
