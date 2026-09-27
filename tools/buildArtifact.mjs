// Prepare the app for publishing as a claude.ai Artifact: writes <outDir>/index.html (the page body
// without <!doctype>/<html>/<head>/<body>, which the host adds) and <outDir>/files.json (published path ->
// source path for the compiled modules, regional.html and the data files). Run `npx tsc -p .` first.
// Usage: node tools/buildArtifact.mjs <outDir>
import { readFileSync, writeFileSync, mkdirSync, readdirSync, statSync } from 'node:fs';
import { join } from 'node:path';

const out = process.argv[2];
if (!out) throw new Error('usage: node tools/buildArtifact.mjs <outDir>');
mkdirSync(out, { recursive: true });
const html = readFileSync('index.html', 'utf8');
const title = html.match(/<title>[\s\S]*?<\/title>/)[0];
const style = html.match(/<style>[\s\S]*?<\/style>/)[0].replace(':root{--bg:', ':root{color-scheme:dark;--bg:');
const body = html.match(/<body>([\s\S]*)<\/body>/)[1].trim();
writeFileSync(join(out, 'index.html'), `${title}\n${style}\n${body}\n`);
const files = {};
const walk = (d) => { for (const f of readdirSync(d)) { const p = join(d, f); if (statSync(p).isDirectory()) { if (f !== 'tools' && f !== 'tests') walk(p); } else if (p.endsWith('.js')) files[p] = p; } };
walk('dist');
files['regional.html'] = 'regional.html';
for (const f of ['data/earth_t42.json', 'data/qflux_gray_t21.json']) files[f] = f;
writeFileSync(join(out, 'files.json'), JSON.stringify(files, null, 1));
console.log(`${Object.keys(files).length} files -> ${out}`);
