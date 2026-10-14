// The settings that can be changed while the run goes on (liveSettings.ts), as a form made from the set-up form's field table (setupForm.ts): the same fields,
// labels and tips, only those the running model can take; "Apply" sends them to the worker, which changes the model in place (no restart: the time, the recording and
// the history go on).
import { FIELDS, type Field, type Key } from './setupForm.js';
import { LIVE_KEYS, changedKeys, liveMerge } from './liveSettings.js';
import { sanitize, setupCells, type RegionalSetup } from './setup.js';

export class LiveForm {
  private cur: RegionalSetup | null = null;
  private readonly rows = new Map<Key, { row: HTMLElement; text: HTMLElement; input: HTMLInputElement | HTMLSelectElement; f: Field }>();
  /** the page sends the settings to the worker */
  onApply: (s: RegionalSetup) => void = () => {};

  constructor(private readonly box: HTMLElement, private readonly info: HTMLElement, private readonly applyBtn: HTMLButtonElement) {
    for (const f of FIELDS.filter((x) => (LIVE_KEYS as readonly string[]).includes(x.key))) {
      const row = document.createElement('label'); row.className = f.kind === 'chk' ? 'f chk' : 'f';
      const text = document.createElement('span'); text.textContent = f.label; if (f.tip) row.title = f.tip;
      let input: HTMLInputElement | HTMLSelectElement;
      if (f.kind === 'sel') { const s = document.createElement('select'); for (const [v, t] of f.options!) s.append(new Option(t, v)); input = s; }
      else { const i = document.createElement('input'); i.type = f.kind === 'chk' ? 'checkbox' : 'number'; if (f.step) i.step = String(f.step); input = i; }
      input.addEventListener(f.kind === 'num' ? 'input' : 'change', () => this.changed());
      if (f.kind === 'chk') row.append(input, text); else row.append(text, input);
      box.append(row);
      this.rows.set(f.key, { row, text, input, f });
    }
    applyBtn.addEventListener('click', () => { if (this.cur) this.onApply(this.value()); });
    this.enable(false);
  }

  /** Show the set-up the running model has. */
  set(s: RegionalSetup): void {
    this.cur = { ...s };
    for (const [k, r] of this.rows) {
      const v = s[k];
      if (r.f.kind === 'chk') (r.input as HTMLInputElement).checked = !!v;
      else if (r.f.kind === 'sel') r.input.value = String(v);
      else r.input.value = String(+((v as number) / (r.f.scale ?? 1)).toFixed(4));
    }
    this.changed(false);
  }

  /** The set-up in the form (the running model's with the settings shown: sanitized). */
  value(): RegionalSetup {
    const s = { ...this.cur! } as unknown as Record<string, unknown>;
    for (const [k, r] of this.rows) {
      if (r.f.kind === 'chk') s[k] = (r.input as HTMLInputElement).checked;
      else if (r.f.kind === 'sel') s[k] = r.input.value;
      else { const x = Number(r.input.value); s[k] = Number.isFinite(x) ? x * (r.f.scale ?? 1) : (this.cur![k] as number); }
    }
    return liveMerge(this.cur!, s as unknown as RegionalSetup);
  }

  /** No settings to change (the axisymmetric model, a nest in the global model, no model yet). */
  enable(on: boolean): void {
    this.box.hidden = !on; this.applyBtn.hidden = !on; this.info.hidden = !on;
  }

  private changed(edited = true): void {
    if (!this.cur) return;
    const s = this.value();
    for (const r of this.rows.values()) {
      r.row.hidden = !(r.f.show?.(s) ?? true);
      if (r.f.dyn) r.text.textContent = r.f.dyn(s);
    }
    const n = changedKeys(this.cur, s).length;
    this.applyBtn.disabled = n === 0;
    this.info.textContent = edited && n ? `${n} 項設定和目前的模擬不同；按「套用」只改這些，模擬接著跑、不重來 / ${n} settings differ from the run: Apply changes only these, the run goes on` : '';
  }
}

/**
 * The grid settings that can change while the run goes on: the domain (smaller only), the model top and the vertical spacing. "Apply" continues the run on the new grid
 * (the state interpolated into it: like a refinement); the page keeps the recording and the charts.
 */
export class GridForm {
  private cur: RegionalSetup | null = null;
  private readonly rows = new Map<Key, { input: HTMLInputElement; f: Field }>();
  /** the page sends the grid to the worker */
  onApply: (s: RegionalSetup) => void = () => {};

  constructor(private readonly box: HTMLElement, private readonly info: HTMLElement, private readonly applyBtn: HTMLButtonElement) {
    for (const f of FIELDS.filter((x) => x.key === 'L' || x.key === 'top' || x.key === 'dz')) {
      const row = document.createElement('label'); row.className = 'f';
      const text = document.createElement('span'); text.textContent = f.label; if (f.tip) row.title = f.tip;
      const input = document.createElement('input'); input.type = 'number'; if (f.step) input.step = String(f.step);
      input.addEventListener('input', () => this.changed());
      row.append(text, input); box.append(row);
      this.rows.set(f.key, { input, f });
    }
    applyBtn.addEventListener('click', () => { if (this.cur) this.onApply(this.value()); });
    this.enable(false);
  }

  set(s: RegionalSetup): void {
    this.cur = { ...s };
    for (const [k, r] of this.rows) r.input.value = String(+((s[k] as number) / (r.f.scale ?? 1)).toFixed(4));
    this.changed(false);
  }

  /** The set-up with the grid of the form (sanitized). */
  value(): RegionalSetup {
    const s = { ...this.cur! } as unknown as Record<string, unknown>;
    for (const [k, r] of this.rows) { const x = Number(r.input.value); s[k] = Number.isFinite(x) ? x * (r.f.scale ?? 1) : (this.cur![k] as number); }
    return sanitize(s as unknown as RegionalSetup);
  }

  enable(on: boolean): void { this.box.hidden = !on; this.applyBtn.hidden = !on; this.info.hidden = !on; }

  private changed(edited = true): void {
    if (!this.cur) return;
    const s = this.value(), c = this.cur, cells = setupCells(s), nz = Math.round(s.top / s.dz), nx = Math.round(s.L / s.dx);
    const same = Math.abs(s.L - c.L) < 1 && Math.abs(s.top - c.top) < 1 && Math.abs(s.dz - c.dz) < 1e-9;
    const parts = [`${nx}×${nx}×${nz} = ${(cells / 1e6).toFixed(cells < 1e6 ? 2 : 1)} M 格點 / cells`];
    if (s.L > c.L + 1) parts.push('⚠ 範圍加大要用外圈粗網格（還沒做）/ a larger domain needs a coarse outer grid (not done yet)');
    if (cells > 30e6) parts.push('⚠ 太大（上限 30 M）/ too large (limit 30 M)');
    if (edited && !same) parts.push('按「套用網格」後模擬接著跑（狀態內插到新網格，眼區細化會停止）/ Apply continues the run on the new grid (the eye nest stops)');
    this.info.textContent = parts.join(' · ');
    this.applyBtn.disabled = same || s.L > c.L + 1 || cells > 30e6;
  }
}
