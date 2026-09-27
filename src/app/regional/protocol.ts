// Messages between the regional-model UI and its worker.

export type RegionalExperiment = 'supercell' | 'tc';
export type GroundField = 'rain' | 'wind' | 'theta';

export type ToRegionalWorker =
  | { type: 'init'; experiment: RegionalExperiment }
  | { type: 'run'; running: boolean }
  | { type: 'speed'; stepsPerTick: number }
  | { type: 'ground'; field: GroundField };

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
  stats: { wmax: number; wmin: number; qcmax: number; qrmax: number; rainmax: number; vmax: number; dp: number | null; rmw: number | null };
  stepsPerSecond: number;
}

export type FromRegionalWorker =
  | RegionalFrame
  | { type: 'ready'; experiment: RegionalExperiment; nx: number; ny: number; nz: number; dx: number; dz: number; dt: number; description: string }
  | { type: 'error'; message: string };
