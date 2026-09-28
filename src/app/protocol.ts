// Messages between the UI thread and the simulation worker.
import type { NestPayload } from './regional/protocol.js';

export type FieldId = 'T' | 'u' | 'v' | 'speed' | 'vor' | 'ps' | 'div' | 'precip' | 'snow' | 'sst' | 'q' | 'olr' | 'ice' | 'sat';

export type ToWorker =
  /** spinup: Earth presets with ocean heat transport start from the spun-up July state (data/spinup_earth_t42q.bin) */
  | { type: 'init'; preset: string; backend: 'auto' | 'cpu'; spinup?: boolean; state?: ArrayBuffer }
  /** capture the current state as a save (answered with saveData) */
  | { type: 'save' }
  | { type: 'run'; running: boolean }
  | { type: 'speed'; stepsPerTick: number }
  | { type: 'view'; field: FieldId; level: number }
  | { type: 'resetAverage' }
  | { type: 'snapshot' };

export interface FrameMessage {
  type: 'frame';
  day: number;
  steps: number;
  stepsPerSecond: number;
  nlat: number;
  nlon: number;
  K: number;
  lat: Float64Array;          // radians, north -> south
  sigma: Float64Array;
  level: number;
  field: FieldId;
  scalar: Float32Array;       // selected field at level, [lat][lon]
  u: Float32Array;            // wind at level for tracers
  v: Float32Array;
  maxWind: number;
  psDrift: number;
  /** diagnosed 3-D cloud and precipitation [k][lat][lon][2] (bytes: cloud fraction, precipitation) on cloudNz height levels from sea level to cloudTop (m) */
  cloud3d: Uint8Array | null;
  cloudNz: number;
  cloudTop: number;
  /** season information for seasonal experiments */
  declinationDeg: number | null;
}

export interface ZonalMessage {
  type: 'zonal';
  samples: number;
  fromDay: number;
  lat: number[];
  sigma: number[];
  sigmaHalf: number[];
  u: number[];
  T: number[];
  psi: number[];
}

export type FromWorker =
  | FrameMessage
  | ZonalMessage
  | { type: 'snapshot'; payload: NestPayload }
  | { type: 'saveData'; meta: import('./saves.js').SaveMeta; buffer: ArrayBuffer }
  | { type: 'ready'; preset: string; trunc: number; nlat: number; nlon: number; K: number; dt: number; moist: boolean; lat: Float64Array; land: Uint8Array | null; backend: 'cpu' | 'gpu'; note: string }
  | { type: 'error'; message: string };
