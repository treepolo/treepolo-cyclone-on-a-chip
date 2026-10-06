// Messages between the regional-model UI and its worker.
import type { MapVar, SliceVar, SectionVar, RzVar } from '../../regional/diagnostics.js';
import type { StormNow } from '../../regional/storms.js';
import type { ReplayAux, ReplayMeta, ReplayRec, ReplayTier, Q16 } from '../../regional/replayData.js';

export type RegionalExperiment = 'supercell' | 'tc' | 'supercell_hr' | 'tc_hr' | 'tc_3' | 'nest' | 'tornado' | 'tornado_c' | 'tc_axi' | 'custom';
/** tropical cyclones: mean rain rate in the core and the outer region over the last model hour (mm/h), outer wet fraction */
export interface TcRain { core: number; outer: number; wet: number }

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
export type GroundField = 'rain' | 'wind' | 'theta' | 'snow' | 'none';

export type ToRegionalWorker =
  | { type: 'init'; setup: import('./setup.js').RegionalSetup; backend: 'auto' | 'cpu' }
  | { type: 'initNest'; payload: NestPayload; lat0: number; lon0: number; size: NestSize; backend: 'auto' | 'cpu' }
  | { type: 'run'; running: boolean }
  | { type: 'speed'; stepsPerTick: number }
  | { type: 'ground'; field: GroundField }
  /** newer global state: refresh the lateral-boundary relaxation targets of the running nest */
  | { type: 'boundary'; payload: NestPayload }
  /** time every GPU kernel for a few steps and report */
  | { type: 'profile' }
  | { type: 'adaptive'; on: boolean }
  /** step size: automatic (from the Courant number; `cfl` is its target) or manual (`dt` seconds, kept as it is) */
  | { type: 'step'; mode: 'auto' | 'manual'; dt?: number; cfl?: number; /** counted by the page; the frames echo the last one handled, so a frame sent before it is not taken for the new state */ seq?: number }
  /** recording for the replay: every `every` model seconds a frame is kept (`keep` in the frame) with what `tier` says */
  | { type: 'record'; on: boolean; tier: ReplayTier; every: number; /** keep the next frame at once */ now?: boolean }
  /** the charts of a recorded frame (tier 'full'): computed on a model of its own from the packed state; the meta and the
   *  frame data are sent only when the worker does not hold them (`need` in the answer) */
  | { type: 'replayChart'; id: number; req: ChartRequest; metaId: number; meta?: ReplayMeta; key: number;
      frame?: { state: Q16[]; aux: ReplayAux | null; centre: { x: number; y: number } | null } }
  /** continue the running simulation on the finer grid of its set-up (build.ts refinedSetup) */
  | { type: 'refine' }
  /** refine the eye and eyewall: a finer grid (spacings near dx, dz; m) in a cylinder of radius R (m) at the centre of the
   *  running domain, coupled both ways (twoway.ts); replaces a running one */
  | { type: 'nestStart'; R: number; dx: number; dz: number }
  | { type: 'nestStop' }
  /** back to the coarser grid (the kept one, with the fine run averaged into it) */
  | { type: 'coarsen' }
  | { type: 'save' }
  /** target speed in model seconds per wall second (0 = full speed) */
  | { type: 'pace'; target: number }
  /** pause after this many model hours from now (0 clears) */
  | { type: 'runUntil'; hours: number }
  | { type: 'step1' }
  /** display cadence: every 0.5 s, every `every` model seconds, or rarely (fast-forward) */
  | { type: 'frames'; kind: 'wall' | 'model' | 'fast'; every?: number }
  | { type: 'load'; buffer: ArrayBuffer; backend: 'auto' | 'cpu' }
  /** chart data to add to every frame (null: none, e.g. the globe-embedded nest) */
  | { type: 'charts'; req: ChartRequest | null }
  /** second channel of the 3-D view: 0 precipitation, 1 updraft, 2 cyclonic vertical vorticity */
  | { type: 'volMode'; mode: number }
  /** draw sub-grid (partial) cloud in the 3-D view and the visible image (display.ts) */
  | { type: 'subgrid'; on: boolean }
  /** the saturation switches of the cloud physics (ice.ts MicroOpts): ice supersaturation, and the time (s) liquid water takes to condense (0: at once) */
  | { type: 'micro'; iceSS?: boolean; liqTau?: number }
  /** the domain follows the storm (on) or stays on the ground (off), at any time: the page's switch; the worker answers with the same message when it changes by itself */
  | { type: 'follow'; on: boolean }
  /** tracer particles for the 3-D view (0: off) */
  | { type: 'tracers'; n: number }
  /** interaction (conditions only): a warm bubble or a cold pool centred at (x, y) m in domain coordinates, optionally
   *  at height z (m; default 1.5 km warm, the ground cold) with horizontal radius and full depth (m) and theta amplitude
   *  amp (K; default +3 warm, -6 cold) */
  | { type: 'perturb'; kind: 'warm' | 'cold'; x: number; y: number; z?: number; radius?: number; depth?: number; amp?: number }
  /** a vortex added at (x, y) m: strongest wind `vmax` (m/s) at height z (m: the ground for a warm-core vortex, aloft for a cold-core one) and radius `rm` (m; 0: a fifth
   *  of the outer radius), outer `radius` and full `depth` (m) as the other tools, `dir` +1 counter-clockwise, -1 clockwise; balanced pressure and temperature (vortex.ts) */
  | { type: 'vortex'; kind: 'warm' | 'cold'; x: number; y: number; z: number; radius: number; depth: number; vmax: number; rm: number; dir: 1 | -1 }
  /** the liquid water (cloud water and rain) or the solid water (ice, snow and graupel) of a region (centre x, y, z, radius, full depth, m): multiplied by `amount`,
   *  added `amount` g/kg (negative removes), or set to `amount` g/kg (0 removes it all); the same all over the region */
  | { type: 'condensate'; phase: 'liquid' | 'solid'; x: number; y: number; z: number; radius: number; depth: number; mode: 'mul' | 'add' | 'set'; amount: number }
  /** change the water vapour in a region (centre x, y, z, radius, full depth, m): multiply it by `amount`, add `amount` g/kg (negative removes), or
   *  set it to the relative humidity `amount` % (over water); the same all over the region */
  | { type: 'moisture'; x: number; y: number; z: number; radius: number; depth: number; mode: 'mul' | 'add' | 'rh'; amount: number }
  /** wind in a region (forcing.ts): speed (m/s), direction the push blows toward (az degrees clockwise from north,
   *  el degrees above the horizontal), form, sign (rotate: +1 counter-clockwise; converge: +1 inward); minutes 0: once,
   *  > 0: lasting that many model minutes, < 0: lasting until cleared */
  | { type: 'wind'; x: number; y: number; z: number; radius: number; depth: number; speed: number; az: number; el: number; form: 'push' | 'rotate' | 'converge'; sign: 1 | -1; minutes: number }
  /** stop every lasting wind forcing */
  | { type: 'clearForcing' }
  /** paint the surface within `radius` m of (x, y): sea temperature change of `amount` K (default 2) or land / sea */
  /** paint the surface: land, sea, or the sea (surface) temperature set to `amount` °C */
  | { type: 'paint'; kind: 'sst' | 'land' | 'sea'; x: number; y: number; radius: number; amount?: number }
  /** change the environment now: add du6 (m/s) of westerly wind at 6 km (linear from the ground) and multiply
   *  the 1-8 km water vapour by humidity (capped at saturation) */
  | { type: 'environment'; du6: number; humidity: number };

/** A lasting wind forcing as shown on the page. */
export interface ForcingInfo { id: number; x: number; y: number; z: number; radius: number; depth: number; speed: number; az: number; el: number; form: 'push' | 'rotate' | 'converge'; sign: 1 | -1; until: number | null }

/** What the charts need in each frame. Positions in m from the domain origin (lower-left corner). */
export interface ChartRequest {
  maps: MapVar[];
  slice: { k: number; vars: SliceVar[] } | null;
  section: { x0: number; y0: number; x1: number; y1: number } | null;
  sounding: { x: number; y: number } | null;
  /** azimuthal means about the surface-pressure minimum ('auto') or a given centre */
  rz: 'auto' | { x: number; y: number } | null;
}

/** Chart data of a frame. 2-D maps are [j][i]; section fields [k][point]; r-z fields [k][ring]. */
export interface ChartData {
  maps: Partial<Record<MapVar, Float32Array>>;
  /** base-state temperature (K) at the level centres (the infrared picture's temperature of height) */
  tz?: Float32Array;
  /** base-state pressure (hPa) at the level centres (the slices on pressure levels pick the nearest level) */
  p0?: Float32Array;
  slice: { k: number; z: number; vars: Partial<Record<SliceVar, Float32Array>> } | null;
  section: { x0: number; y0: number; x1: number; y1: number; np: number; vars: Record<SectionVar, Float32Array> } | null;
  sounding: { x: number; y: number; vars: Record<SectionVar, Float32Array> } | null;
  rz: { xc: number; yc: number; dr: number; nr: number; vars: Record<RzVar, Float32Array> } | null;
}

/** The running eye nest: its grid and where it lies (m, domain coordinates of the outer grid). */
export interface NestInfo {
  R: number; dx: number; dz: number; r: number; rz: number; nx: number; nz: number; cells: number;
  /** inner steps per outer step now */
  nsub: number;
  /** the inner box: lower-left corner and width; the cylinder's centre */
  x0: number; y0: number; L: number; cx: number; cy: number;
  /** feedback taper width inside R (m) */
  Wf: number;
}
/** The eye nest's 3-D view bytes ([k][j][i], nx x nx x nz) with its geometry. */
export interface NestFrame extends NestInfo { cloud: Uint8Array; rain: Uint8Array }

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
  stats: {
    wmax: number; wmin: number; qcmax: number; qrmax: number; rainmax: number; vmax: number; dp: number | null; rmw: number | null; eyewalls: { r: number; v: number }[] | null; zetaMax: number; vGround: number;
    /** column-max reflectivity (dBZ), 2-5 km updraft helicity (m^2/s^2) and surface-based CAPE (J/kg) maxima */
    dbzMax: number; uhMax: number; capeMax: number;
    /** most negative 2-5 km updraft helicity (anticyclonic left-moving storms), m^2/s^2 */
    uhMin: number;
    /** the main storm (the followed one, else the deepest vortex or the strongest cell): ground-relative position (m), or
     *  null before any storm exists */
    storm: { x: number; y: number } | null;
    /** every storm now (storms.ts) and the id of the main one */
    storms: StormNow[]; mainId: number | null;
    /** tropical cyclones: mean precipitation rates (mm/h) in the core (< 60 km) and the outer region (100-300 km) over the last
     *  completed model hour, and the outer area fraction raining more than 1 mm/h (rainband diagnostics) */
    tcRain?: TcRain | null;
    /** tropical cyclones: environmental deep-layer shear: the vector difference of the 200 and 850 hPa winds averaged over the
     *  200-800 km annulus around the centre (the vortex itself averages out): magnitude (m/s) and the direction it points toward
     *  (degrees clockwise from north) */
    shear?: { mag: number; dir: number } | null;
    /** tropical cyclones: azimuthal-mean tangential wind at 1.5 km, rings of width dr from the centre */
    vtProfile: { dr: number; vt: number[] } | null;
    /** tornado-like vortex at the lowest level (grids of 500 m or finer): vertical vorticity >= 0.1 s^-1 with a
     *  ground-relative wind >= 29 m/s within 1.5 km; EF rating from that wind; ground-relative position (m) */
    tornado: { zeta: number; v: number; ef: number; x: number; y: number } | null;
  };
  /** ground-relative position (m) of the domain origin (moves with the frame of storm-following experiments) */
  origin: { x: number; y: number };
  charts: ChartData | null;
  /** tracer particles: x, y, z (m, domain coordinates) and age (s) per particle, or null when off */
  tracers: Float32Array | null;
  /** the eye nest's volume (null: none) */
  nest: NestFrame | null;
  stepsPerSecond: number;
  /** current step size (s): varies with the automatic step size */
  dt: number;
  /** the step size control: mode, the target Courant number of the automatic mode, the nominal step (s) of the set-up, and the
   *  Courant numbers of the current step: advective (sum over the three directions of |u| dt / dx) and of the acoustic sub-steps */
  step: { mode: 'auto' | 'manual'; cfl: number; dt0: number; adv: number; ac: number; seq: number };
  /** a frame the replay keeps (recording on and its interval reached), with what the recording tier adds */
  keep?: boolean;
  rec?: ReplayRec | null;
}

export type FromRegionalWorker =
  | RegionalFrame
  | { type: 'ready'; experiment: RegionalExperiment; nx: number; ny: number; nz: number; dx: number; dz: number; dt: number; description: string; backend: 'cpu' | 'gpu'; note: string; land: Uint8Array | null;
      /** grid spacing (m) the run continues on after 'refine', or null */
      refineTo: number | null;
      /** grid spacing (m) after 'coarsen' (coarsenBack: a kept coarser state), whether the eye nest applies */
      coarsenTo?: number | null; coarsenBack?: boolean; nestOk?: boolean;
      /** tropical-cyclone diagnostics apply (a vortex run) */
      tc: boolean;
      /** the set-up of this run (null: nest in the global model) */
      setup: import('./setup.js').RegionalSetup | null }
  /** the eye nest started, changed or stopped (null) */
  | { type: 'nest'; info: NestInfo | null }
  /** surface changed (painting): land mask per column (1 land) */
  | { type: 'land'; land: Uint8Array | null }
  /** the lasting wind forcings now (domain positions, m; until: model time they end, null: until cleared) */
  | { type: 'forcings'; list: ForcingInfo[] }
  | { type: 'profile'; text: string }
  /** charts of a recorded frame; `need`: the worker holds no meta / no frame data of that id, send it again */
  | { type: 'replayChart'; id: number; charts: ChartData | null; need?: 'meta' | 'frame'; error?: string }
  | { type: 'paused'; reason: string }
  /** whether the domain follows the storm now (the worker turned it on or off, or confirms the page's switch) */
  | { type: 'follow'; on: boolean }
  | { type: 'log'; text: string }
  | { type: 'saveData'; meta: import('../saves.js').SaveMeta; buffer: ArrayBuffer }
  | { type: 'error'; message: string };
