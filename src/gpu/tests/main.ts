import { gdone } from './harness.js';
import { transformTests } from './transformTest.js';

async function main(): Promise<void> {
  try {
    await transformTests();
  } catch (e) {
    console.log(`GPUTEST FAIL exception ${String(e)} ${(e as Error).stack ?? ''}`);
  }
  gdone();
}
void main();
