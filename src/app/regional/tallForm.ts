// The "局部加高" (tall grid) block of the regional page: where (the footprint's centre and width), how high (the new top) and how fine (whole-number ratios of the model's
// spacings). It shows what the grid would be (cells, cost), draws the footprint on the 3-D view's ground, and starts and stops the grid. The grid itself is tall.ts / tallNest.ts.

import type { RegionalConfig } from '../../regional/core.js';
import { tallGeometry, tallMaxCols, tallTopAdvice, type TallSpec } from '../../regional/tall.js';
import { defaultDt } from './setup.js';
import type { TallInfo } from './protocol.js';

const $ = <T extends HTMLElement = HTMLElement>(id: string): T => document.getElementById(id) as T;
const kmTxt = (d: number): string => (d >= 1000 ? `${+(d / 1000).toFixed(2)} km` : `${Math.round(d)} m`);

export interface TallHost { nx: number; ny: number; nz: number; dx: number; dz: number; dt: number; open: boolean }

export class TallForm {
  private host: TallHost | null = null;
  private running: TallInfo | null = null;
  /** the main storm's position (m, domain coordinates), or null */
  private storm: { x: number; y: number } | null = null;

  constructor(private readonly cb: { start: (spec: TallSpec) => void; stop: () => void; footprint: (r: [number, number, number, number] | null) => void }) {
    for (const id of ['tallAt', 'tallW', 'tallTop', 'tallR', 'tallRz']) $(id).addEventListener('input', () => this.refresh());
    $('tallMax').addEventListener('click', () => { if (this.host) { $<HTMLInputElement>('tallW').value = String(+(tallMaxCols(this.cfg()) * this.host.dx / 1000).toFixed(1)); this.refresh(); } });
    $('tallGo').addEventListener('click', () => { const s = this.spec(); if (s) { $<HTMLButtonElement>('tallGo').disabled = true; this.cb.start(s); } });
    $('tallStop').addEventListener('click', () => { $<HTMLButtonElement>('tallStop').disabled = true; this.cb.stop(); });
    $('secTall').addEventListener('toggle', () => this.refresh());
  }

  /** The model's grid now (a new run, a changed grid, an outer grid added: its edges are open then). */
  setHost(h: TallHost | null): void {
    this.host = h;
    if (h) {
      const top = $<HTMLInputElement>('tallTop');
      // a first look: a new top 1.5 times the model's, a footprint of half the domain
      if (!top.dataset.touched) top.value = String(+(1.5 * h.nz * h.dz / 1000).toFixed(1));
      const w = $<HTMLInputElement>('tallW');
      if (!w.dataset.touched) w.value = String(+(Math.min(tallMaxCols(this.cfg()), Math.round(h.nx / 2)) * h.dx / 1000).toFixed(1));
    }
    this.refresh();
  }
  setStorm(s: { x: number; y: number } | null): void { this.storm = s; if (($<HTMLSelectElement>('tallAt')).value === 'storm' && !this.running) this.refresh(); }
  /** The running tall grid (null: none). */
  setRunning(t: TallInfo | null): void {
    this.running = t;
    $<HTMLButtonElement>('tallStop').disabled = !t;
    $('tallNow').textContent = t ? `加高中 / running: ${kmTxt(t.L)} 見方、頂 ${kmTxt(t.Z2)}（原 ${kmTxt(t.Z1)}），Δx ${kmTxt(t.dx)}、Δz ${kmTxt(t.dz)}，${(t.cells / 1e6).toFixed(2)} M 格，每個主網格步 ${t.nsub} 個子步 / sub-steps per step` : '未加高 / not running';
    this.refresh();
  }

  private cfg(): RegionalConfig {
    const h = this.host!;
    return { nx: h.nx, ny: h.nx, nz: h.nz, dx: h.dx, dy: h.dx, dz: h.dz, lateral: h.open ? 'open' : 'periodic', relaxCells: Math.max(6, Math.round(2500 / h.dx)) } as unknown as RegionalConfig;
  }
  private num(id: string, d: number): number { const v = Number($<HTMLInputElement>(id).value); return Number.isFinite(v) && v > 0 ? v : d; }

  /** What the panel asks for (m), or null without a model. */
  spec(): TallSpec | null {
    const h = this.host; if (!h) return null;
    const at = $<HTMLSelectElement>('tallAt').value, c = at === 'storm' && this.storm ? this.storm : { x: h.nx * h.dx / 2, y: h.nx * h.dx / 2 };
    return { cx: c.x, cy: c.y, width: this.num('tallW', 50) * 1000, top: this.num('tallTop', 24) * 1000, r: Math.round(this.num('tallR', 1)), rz: Math.round(this.num('tallRz', 1)) };
  }

  private refresh(): void {
    const h = this.host, info = $('tallInfo'), go = $<HTMLButtonElement>('tallGo');
    for (const id of ['tallW', 'tallTop']) { const el = $<HTMLInputElement>(id); el.oninput = (): void => { el.dataset.touched = '1'; this.refresh(); }; }
    if (!h) { info.textContent = ''; go.disabled = true; this.cb.footprint(null); return; }
    const spec = this.spec()!, g = tallGeometry(this.cfg(), spec);
    if (typeof g === 'string') { info.textContent = `⚠ ${g}`; go.disabled = true; this.cb.footprint(null); return; }
    const nsub = Math.max(1, Math.ceil(h.dt / defaultDt({ dx: g.dx, dz: g.dz }) - 1e-6)), cells = g.nx * g.nx * g.nz, outer = h.nx * h.nx * h.nz;
    const cost = 1 + cells * nsub / outer, c = cost < 10 ? cost.toFixed(1) : cost.toFixed(0), notes: string[] = [];
    if (g.Z2 < tallTopAdvice(g.Z1)) notes.push(`新頂比原頂（${kmTxt(g.Z1)}）高不到 6 km：新頂的海綿層會蓋住原頂附近，那一段被主網格拉著走；建議至少 ${kmTxt(tallTopAdvice(g.Z1))} / the new top is less than 6 km above the old: its Rayleigh layer covers the old top; at least ${kmTxt(tallTopAdvice(g.Z1))} advised`);
    if (cells > 8e6) notes.push('⚠ 需要較強的顯卡 / needs a strong GPU');
    notes.push('側邊在原本海綿層高度吃到的是被主網格海綿層壓過的值，風暴要在範圍中央、範圍要夠大 / at the height of the old Rayleigh layer the sides take the damped values of the model: keep the storm well inside');
    info.textContent = `實際 / actual: ${kmTxt(g.np * h.dx)} 見方（${g.np}×${g.np} 個主格）、頂 ${kmTxt(g.Z2)}（原 ${kmTxt(g.Z1)}）、Δx ${kmTxt(g.dx)}（主網格的 1/${g.r}）、Δz ${kmTxt(g.dz)}（1/${g.rz}），` +
      `${g.nx}×${g.nx}×${g.nz} = ${(cells / 1e6).toFixed(2)} M 格，每個主網格步約 ${nsub} 個子步；每模式秒的計算量約為目前的 ${c} 倍 · ${notes.join(' · ')}`;
    go.disabled = !!this.running;
    const L = h.nx * h.dx;
    this.cb.footprint(this.running ? [this.running.x0 / L, this.running.y0 / L, (this.running.x0 + this.running.L) / L, (this.running.y0 + this.running.L) / L] : ($('secTall') as HTMLDetailsElement).open ? [g.x0 / L, g.y0 / L, (g.x0 + g.np * h.dx) / L, (g.y0 + g.np * h.dx) / L] : null);
  }
}
