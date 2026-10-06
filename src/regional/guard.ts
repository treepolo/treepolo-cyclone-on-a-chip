// The state guard of the regional model: after every step (and every interaction) each cell is looked at once.
//   * a cell with a value that is not a number (or beyond 1e30) is put back to the environment (the base state's wind, temperature and vapour, no condensate):
//     whatever made it can no longer be told, and a number that is not one would spread to the whole domain (as a growing square)
//   * values the equations cannot take are held at what they can: temperature at most T_CEIL, water (each species) at most Q_CEIL kg per kg of dry air, wind at most V_CEIL,
//     the Exner function between 5 % of its base value and PI_CEIL
// An ordinary run never gets near these (air of 180 to 350 K, water below 0.05 kg/kg, wind below 150 m/s), so nothing changes there. What was done is counted, and
// the page says it in the log: an interaction beyond the physics is held at the limit instead of breaking the model (the GPU kernel in regionalGpu.ts does the same).
import { T_CEIL, Q_CEIL, V_CEIL, PI_CEIL } from '../core/constants.js';
import type { RegionalModel } from './core.js';

/** cells put back to the environment (not numbers), cells with too much water, too hot, too fast, too much pressure */
export interface GuardCounts { lost: number; water: number; hot: number; wind: number; pressure: number }
export const noGuard = (): GuardCounts => ({ lost: 0, water: 0, hot: 0, wind: 0, pressure: 0 });
export const guardAny = (c: GuardCounts): boolean => c.lost + c.water + c.hot + c.wind + c.pressure > 0;
export const addGuard = (a: GuardCounts, b: GuardCounts): void => { a.lost += b.lost; a.water += b.water; a.hot += b.hot; a.wind += b.wind; a.pressure += b.pressure; };

/** What the guard did, in words (Chinese / English), or '' when it did nothing. */
export function guardNotes(c: GuardCounts, cellSteps = false): string {
  const zh: string[] = [], en: string[] = [], u = cellSteps ? '格次' : '格', ue = cellSteps ? 'cell-steps' : 'cells';
  if (c.water) { zh.push(`${c.water} ${u}的水氣或水量超過每公斤乾空氣 ${Q_CEIL} 公斤，壓回 ${Q_CEIL}（再多就是水比空氣還多，不再是空氣）`); en.push(`${c.water} ${ue} held at ${Q_CEIL} kg of water per kg of air (more is a steam atmosphere, not air)`); }
  if (c.hot) { zh.push(`${c.hot} ${u}的溫度超過 ${T_CEIL.toExponential(0).replace('e+', '×10^')} K，壓回該溫度（再熱是電漿，理想氣體公式不適用）`); en.push(`${c.hot} ${ue} held at ${T_CEIL.toExponential(0)} K (hotter is a plasma, not an ideal gas)`); }
  if (c.wind) { zh.push(`${c.wind} ${u}的風速超過 ${V_CEIL.toExponential(0).replace('e+', '×10^')} m/s，壓回該風速`); en.push(`${c.wind} ${ue} held at ${V_CEIL.toExponential(0)} m/s`); }
  if (c.pressure) { zh.push(`${c.pressure} ${u}的氣壓超出能算的範圍，壓回邊界`); en.push(`${c.pressure} ${ue} held to the pressure the formulas can take`); }
  if (c.lost) { zh.push(`${c.lost} ${u}的數值已經不是數字，重設成環境值`); en.push(`${c.lost} ${ue} held numbers that were not numbers (beyond what the model can compute): reset to the environment`); }
  return zh.length ? `${zh.join('；')} / ${en.join('; ')}` : '';
}

/** Look at every cell of a CPU model (all levels, w included): repair and count. */
export function guardModel(m: RegionalModel): GuardCounts {
  const out = noGuard(), { nx, ny, nz } = m.c, ns = m.scalars.length;
  const bad = (x: number): boolean => !(Math.abs(x) < 1e30);
  for (let k = 0; k <= nz; k++) {
    const kb = Math.min(k, nz - 1), pi0 = m.pi0[kb]!;
    for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      const q = m.idx(i, j, k);
      let isBad = bad(m.u[q]!) || bad(m.v[q]!) || bad(m.w[q]!) || bad(m.th[q]!) || bad(m.pp[q]!);
      for (let s = 0; s < ns && !isBad; s++) isBad = bad(m.scalars[s]![q]!);
      if (isBad) {
        out.lost++;
        m.u[q] = m.ub[kb]!; m.v[q] = m.vb[kb]!; m.w[q] = 0; m.th[q] = m.th0[kb]!; m.pp[q] = 0;
        for (let s = 0; s < ns; s++) m.scalars[s]![q] = s === 0 ? m.qv0[kb]! : 0;
        continue;
      }
      let hitV = false;
      if (Math.abs(m.u[q]!) > V_CEIL) { m.u[q] = Math.sign(m.u[q]!) * V_CEIL; hitV = true; }
      if (Math.abs(m.v[q]!) > V_CEIL) { m.v[q] = Math.sign(m.v[q]!) * V_CEIL; hitV = true; }
      if (Math.abs(m.w[q]!) > V_CEIL) { m.w[q] = Math.sign(m.w[q]!) * V_CEIL; hitV = true; }
      const pp = Math.min(Math.max(m.pp[q]!, -0.95 * pi0), PI_CEIL - pi0);
      if (pp !== m.pp[q]) { m.pp[q] = pp; out.pressure++; }
      const thMax = T_CEIL / (pi0 + pp);
      if (m.th[q]! > thMax) { m.th[q] = thMax; out.hot++; }
      let hitQ = false;
      for (let s = 0; s < ns; s++) if (m.scalars[s]![q]! > Q_CEIL) { m.scalars[s]![q] = Q_CEIL; hitQ = true; }
      if (hitV) out.wind++;
      if (hitQ) out.water++;
    }
  }
  return out;
}
