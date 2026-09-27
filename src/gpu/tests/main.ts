import { gdone } from './harness.js';
import { transformTests } from './transformTest.js';
import { dycoreTests } from './dycoreTest.js';
import { moistTests, earthTests } from './moistTest.js';
import { regionalTests, regionalNestTests, regionalIceTests, regionalPerf } from './regionalTest.js';
import { regionalDebug } from './regionalDebug.js';

async function main(): Promise<void> {
  try {
    const which = new URLSearchParams(location.search).get('only');
    if (!which || which === 'transform') await transformTests();
    if (!which || which === 'dycore') await dycoreTests();
    if (!which || which === 'moist') await moistTests();
    if (!which || which === 'earth') await earthTests();
    if (!which || which === 'regional') await regionalTests();
    if (!which || which === 'regional' || which === 'nest') await regionalNestTests();
    if (!which || which === 'regional' || which === 'ice') await regionalIceTests();
    if (which === 'perf') await regionalPerf();
    if (which === 'rdebug') await regionalDebug();
  } catch (e) {
    console.log(`GPUTEST FAIL exception ${String(e)} ${(e as Error).stack ?? ''}`);
  }
  gdone();
}
void main();
