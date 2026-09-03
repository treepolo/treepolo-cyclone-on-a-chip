import { createServer } from 'node:http';
import { readFile, stat } from 'node:fs/promises';
import { extname, join, normalize } from 'node:path';

const root = process.cwd();
const port = Number(process.env.PORT || 5173);
const mime = new Map([
  ['.html', 'text/html; charset=utf-8'], ['.js', 'text/javascript; charset=utf-8'],
  ['.map', 'application/json'], ['.css', 'text/css; charset=utf-8'], ['.json', 'application/json']
]);

createServer(async (req, res) => {
  try {
    const raw = decodeURIComponent((req.url || '/').split('?')[0]);
    const rel = raw === '/' ? '/index.html' : raw;
    const safe = normalize(rel).replace(/^(\.\.[/\\])+/, '');
    let file = join(root, safe);
    const s = await stat(file);
    if (s.isDirectory()) file = join(file, 'index.html');
    const body = await readFile(file);
    res.writeHead(200, {'content-type': mime.get(extname(file)) || 'application/octet-stream', 'cache-control': 'no-store'});
    res.end(body);
  } catch {
    res.writeHead(404, {'content-type': 'text/plain; charset=utf-8'});
    res.end('Not found');
  }
}).listen(port, '127.0.0.1', () => {
  console.log(`Cyclone on a Chip: http://127.0.0.1:${port}`);
});
