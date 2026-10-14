// The settings that can be changed while the run goes on (liveSettings.ts), as a form made from the set-up form's field table (setupForm.ts): the same fields,
// labels and tips, only those the running model can take; "Apply" sends them to the worker, which changes the model in place (no restart: the time, the recording and
// the history go on).
import { FIELDS, type Field, type Key } from './setupForm.js';
import { LIVE_KEYS, changedKeys, liveMerge } from './liveSettings.js';
import type { RegionalSetup } from './setup.js';

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
