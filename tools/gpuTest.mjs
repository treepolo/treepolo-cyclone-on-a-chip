// Run the WebGPU validation page in headless Chromium (SwiftShader software adapter).
// Usage: node tools/gpuTest.mjs [page=gpu-test.html]   (serves the repo root on a free port)
import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { extname, join } from 'node:path';
import { createRequire } from 'node:module';
import { execSync } from 'node:child_process';

const require = createRequire(import.meta.url);
let playwright;
try { playwright = require('playwright'); } catch { playwright = require(join(execSync('npm root -g').toString().trim(), 'playwright')); }
const page = process.argv[2] ?? 'gpu-test.html';
const mime = { '.html': 'text/html', '.js': 'text/javascript', '.json': 'application/json', '.map': 'application/json' };
const server = createServer(async (req, res) => {
  try { const f = join(process.cwd(), decodeURIComponent(req.url.split('?')[0])); res.writeHead(200, { 'content-type': mime[extname(f)] ?? 'application/octet-stream' }); res.end(await readFile(f)); }
  catch { res.writeHead(404); res.end(); }
}).listen(0, '127.0.0.1');
await new Promise((r) => server.once('listening', r));
const port = server.address().port;
const browser = await playwright.chromium.launch({ args: ['--enable-unsafe-webgpu', '--use-webgpu-adapter=swiftshader', '--enable-features=Vulkan'] });
const p = await browser.newPage();
let done = false, failed = 0;
const finished = new Promise((resolve) => {
  p.on('console', (m) => {
    const t = m.text();
    if (t.startsWith('GPUTEST')) { console.log(t.slice(8)); if (t.includes(' FAIL ')) failed++; if (t.startsWith('GPUTEST DONE')) { done = true; resolve(); } }
  });
  p.on('pageerror', (e) => { console.log('pageerror', e.message); failed++; resolve(); });
});
await p.goto(`http://127.0.0.1:${port}/${page}`);
await Promise.race([finished, new Promise((r) => setTimeout(r, Number(process.env.GPU_TEST_TIMEOUT ?? 600000)))]);
await browser.close();
server.close();
if (!done) { console.log('timeout'); process.exit(1); }
process.exit(failed ? 1 : 0);
