import type { Rgb } from './globe.js';

const TURBO: Rgb[] = [[0.19, 0.07, 0.23], [0.27, 0.42, 0.89], [0.16, 0.73, 0.93], [0.19, 0.95, 0.6], [0.64, 0.99, 0.24], [0.93, 0.82, 0.23], [0.98, 0.5, 0.13], [0.82, 0.18, 0.02], [0.48, 0.02, 0.01]];
const DIVERGING: Rgb[] = [[0.02, 0.19, 0.38], [0.13, 0.4, 0.67], [0.4, 0.65, 0.81], [0.82, 0.9, 0.94], [0.97, 0.97, 0.97], [0.99, 0.86, 0.78], [0.96, 0.65, 0.51], [0.84, 0.38, 0.3], [0.4, 0.0, 0.12]];

function ramp(stops: Rgb[], t: number): Rgb {
  const x = Math.max(0, Math.min(1, t)) * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(x)), w = x - i;
  const a = stops[i]!, b = stops[i + 1]!;
  return [a[0] + w * (b[0] - a[0]), a[1] + w * (b[1] - a[1]), a[2] + w * (b[2] - a[2])];
}

export const sequential = (t: number): Rgb => ramp(TURBO, t);
/** t in [-1, 1] */
export const diverging = (t: number): Rgb => ramp(DIVERGING, 0.5 + 0.5 * t);

export function rgbCss(c: Rgb): string {
  return `rgb(${Math.round(c[0] * 255)},${Math.round(c[1] * 255)},${Math.round(c[2] * 255)})`;
}
