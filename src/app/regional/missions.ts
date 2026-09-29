// Challenge missions on the regional page: goals the simulated storm must reach by itself, checked
// automatically from the model's statistics. Players may only change the conditions (experiment, sea
// temperature, humidity, shear, bubbles, cold pools, painted sea / land); nothing edits the result.
// Completion is remembered in this browser (a per-viewer convenience).
import type { RegionalExperiment, RegionalFrame } from './protocol.js';

/** what the current run is: its experiment id and whether it is a tropical cyclone */
export interface RunKind { e: RegionalExperiment; tc: boolean }

interface Mission { id: string; title: string; hint: string; applies(k: RunKind): boolean; test(f: RegionalFrame, held: (key: string, ok: boolean, seconds: number) => boolean): boolean }

const supercells = (k: RunKind): boolean => !k.tc;
const tcs = (k: RunKind): boolean => k.tc;
const MISSIONS: Mission[] = [
  { id: 'ty51', title: '強烈颱風：地面最大風 ≥ 51 m/s / Intense typhoon: surface wind ≥ 51 m/s', hint: '提示：暖海溫、濕的中層、較小的混合 / hint: warm sea, moist mid-levels, less mixing',
    applies: tcs, test: (f) => f.stats.vmax >= 51 },
  { id: 'eyes2', title: '雙眼牆：同時兩圈切向風極大維持 3 小時 / Concentric eyewalls for 3 hours', hint: '提示：3 km 或軸對稱 1–2 km、長時間積分、外圍要有對流 / hint: 3 km or axisymmetric 1–2 km, long runs, outer convection',
    applies: tcs, test: (f, held) => held('eyes2', (f.stats.eyewalls?.length ?? 0) >= 2, 3 * 3600) },
  { id: 'split', title: '超大胞分裂：左右移胞同時存在（UH > 50 與 < −50）/ Supercell split: right and left movers (UH > 50 and < −50)', hint: '提示：強的直線風切 / hint: strong straight-line shear',
    applies: supercells, test: (f) => f.stats.uhMax > 50 && f.stats.uhMin < -50 },
  { id: 'tornado', title: '龍捲：偵測到 EF1 以上並維持 1 分鐘 / Tornado: EF1 or stronger for a minute', hint: '提示：250 m 網格、低層強風切、濕邊界層 / hint: 250 m grid, strong low-level shear, moist boundary layer',
    applies: supercells, test: (f, held) => held('tornado', (f.stats.tornado?.ef ?? -1) >= 1, 60) },
  { id: 'rain300', title: '豪雨：單點累積 300 mm / Extreme rain: 300 mm at one place', hint: '提示：移動慢、持續的對流 / hint: slow-moving, persistent convection',
    applies: () => true, test: (f) => f.stats.rainmax >= 300 },
];

export class Missions {
  private done: Record<string, string> = {};
  private since = new Map<string, number>();
  constructor(private readonly root: HTMLElement, private readonly log: (s: string) => void) {
    try { this.done = JSON.parse(localStorage.getItem('cyclone-missions') ?? '{}') as Record<string, string>; } catch { this.done = {}; }
    this.render(null);
  }

  /** Check the goals against a frame of the current experiment. */
  check(f: RegionalFrame, k: RunKind): void {
    const held = (key: string, ok: boolean, seconds: number): boolean => {
      if (!ok) { this.since.delete(key); return false; }
      const s = this.since.get(key) ?? f.time; this.since.set(key, s);
      return f.time - s >= seconds;
    };
    let changed = false;
    for (const m of MISSIONS) {
      if (this.done[m.id] || !m.applies(k)) continue;
      if (m.test(f, held)) {
        this.done[m.id] = `${new Date().toLocaleDateString()} · ${k.e} · t = ${(f.time / 3600).toFixed(1)} h`;
        this.log(`挑戰完成 / Challenge completed: ${m.title}`);
        changed = true;
      }
    }
    if (changed) { try { localStorage.setItem('cyclone-missions', JSON.stringify(this.done)); } catch { /* storage unavailable */ } this.render(k); }
  }

  /** Forget the progress of a new model (hold timers restart). */
  reset(k: RunKind): void { this.since.clear(); this.render(k); }

  private render(k: RunKind | null): void {
    this.root.replaceChildren();
    for (const m of MISSIONS) {
      const row = document.createElement('div'); row.className = 'mission' + (this.done[m.id] ? ' done' : '');
      const mark = document.createElement('span'); mark.className = 'mark'; mark.textContent = this.done[m.id] ? '✓' : '○';
      const body = document.createElement('div');
      const t = document.createElement('div'); t.textContent = m.title;
      const sub = document.createElement('div'); sub.className = 'muted';
      sub.textContent = this.done[m.id] ? `完成 / done: ${this.done[m.id]}` : (k && !m.applies(k) ? '（這個實驗不適用 / not for this experiment）' : m.hint);
      body.append(t, sub); row.append(mark, body); this.root.append(row);
    }
  }
}
