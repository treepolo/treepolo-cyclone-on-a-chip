let passed = 0;
let failed = 0;

export function check(name: string, ok: boolean, value?: number | string): void {
  const v = value === undefined ? '' : `  (${typeof value === 'number' ? value.toExponential(3) : value})`;
  if (ok) { passed++; console.log(`  PASS ${name}${v}`); }
  else { failed++; console.log(`  FAIL ${name}${v}`); }
}

export function summary(suite: string): void {
  console.log(`${suite}: ${passed}/${passed + failed} passed`);
  if (failed > 0) process.exitCode = 1;
}

export { rng } from '../core/random.js';
