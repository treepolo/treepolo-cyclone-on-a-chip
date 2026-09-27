import { gdone } from './harness.js';
import { transformTests } from './transformTest.js';
import { dycoreTests } from './dycoreTest.js';
import { moistTests } from './moistTest.js';

async function main(): Promise<void> {
  try {
    const which = new URLSearchParams(location.search).get('only');
    if (!which || which === 'transform') await transformTests();
    if (!which || which === 'dycore') await dycoreTests();
    if (!which || which === 'moist') await moistTests();
  } catch (e) {
    console.log(`GPUTEST FAIL exception ${String(e)} ${(e as Error).stack ?? ''}`);
  }
  gdone();
}
void main();
