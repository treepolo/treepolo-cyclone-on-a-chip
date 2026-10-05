// Saved simulations, shared by the global and the regional page.
//
// * container: 'CYCS' magic, u32 version, u32 header length, header JSON (UTF-8, padded to 4 bytes),
//   then the arrays listed in the header (Float32 or raw bytes), each padded to 4 bytes
// * storage: IndexedDB in this browser (can be cleared with the site data)
// * export / import: a ZIP file with one deflate-compressed entry (the only archive type the artifact
//   host lets a page offer); the artifact's download capability when present, a normal browser
//   download otherwise

export interface SaveArrays { [name: string]: Float32Array | Uint8Array }
export interface SaveMeta { kind: 'global' | 'regional' | 'replay'; title: string; [k: string]: unknown }
export interface SaveRecord { id: string; kind: 'global' | 'regional' | 'replay'; title: string; created: number; bytes: number; data: ArrayBuffer }

const MAGIC = 0x53435943;   // 'CYCS' little-endian

export function packSave(meta: SaveMeta, arrays: SaveArrays): ArrayBuffer {
  const list = Object.entries(arrays).map(([name, a]) => ({ name, type: a instanceof Float32Array ? 'f32' : 'u8', length: a.length }));
  const head = new TextEncoder().encode(JSON.stringify({ meta, arrays: list }));
  const pad4 = (n: number): number => (n + 3) & ~3;
  let size = 12 + pad4(head.length);
  for (const [, a] of Object.entries(arrays)) size += pad4(a.byteLength);
  const out = new ArrayBuffer(size), dv = new DataView(out), u8 = new Uint8Array(out);
  dv.setUint32(0, MAGIC, true); dv.setUint32(4, 1, true); dv.setUint32(8, head.length, true);
  u8.set(head, 12);
  let o = 12 + pad4(head.length);
  for (const [, a] of Object.entries(arrays)) { u8.set(new Uint8Array(a.buffer, a.byteOffset, a.byteLength), o); o += pad4(a.byteLength); }
  return out;
}

export function unpackSave(buf: ArrayBuffer): { meta: SaveMeta; arrays: SaveArrays } {
  const dv = new DataView(buf);
  if (buf.byteLength < 12 || dv.getUint32(0, true) !== MAGIC) throw new Error('不是本模擬器的存檔 / not a simulation save file');
  if (dv.getUint32(4, true) !== 1) throw new Error('存檔版本不支援 / unsupported save version');
  const hl = dv.getUint32(8, true);
  const head = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 12, hl))) as { meta: SaveMeta; arrays: { name: string; type: 'f32' | 'u8'; length: number }[] };
  const pad4 = (n: number): number => (n + 3) & ~3;
  let o = 12 + pad4(hl);
  const arrays: SaveArrays = {};
  for (const a of head.arrays) {
    const bytes = a.type === 'f32' ? 4 * a.length : a.length;
    arrays[a.name] = a.type === 'f32' ? new Float32Array(buf.slice(o, o + bytes)) : new Uint8Array(buf.slice(o, o + bytes));
    o += pad4(bytes);
  }
  return { meta: head.meta, arrays };
}

// ---------------------------------------------------------------- IndexedDB
// 'saves' holds each whole record (with its data, which can be hundreds of MB), 'meta' the same records without their
// data: the list reads only 'meta', so opening a page never brings every save's data into memory (a list read with
// getAll on 'saves' did: several GB with a few large saves). Version 1 had only 'saves'; the upgrade fills 'meta' with a
// cursor, one record at a time.
type SaveInfo = Omit<SaveRecord, 'data'>;
const info = ({ data: _d, ...r }: SaveRecord): SaveInfo => r;
function openDb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const r = indexedDB.open('cyclone-on-a-chip-saves', 2);
    // an older page of this site holding the database open blocks the upgrade (the request then never answers): say so
    // after a while instead of waiting for ever
    const blocked = (): void => { setTimeout(() => reject(new Error('存檔資料庫被另一個開著的本站分頁佔用，請關閉其他分頁後重試 / the save database is held open by another tab of this site: close the other tabs and retry')), 6000); };
    r.onblocked = blocked;
    r.onupgradeneeded = (ev): void => {
      const db = r.result, old = (ev as IDBVersionChangeEvent).oldVersion;
      if (old < 1) db.createObjectStore('saves', { keyPath: 'id' });
      if (old < 2) {
        const meta = db.createObjectStore('meta', { keyPath: 'id' });
        if (old >= 1) {
          const cur = r.transaction!.objectStore('saves').openCursor();
          cur.onsuccess = (): void => { const c = cur.result; if (!c) return; meta.put(info(c.value as SaveRecord)); c.continue(); };
        }
      }
    };
    r.onsuccess = (): void => { r.result.onversionchange = (): void => r.result.close(); resolve(r.result); };
    r.onerror = (): void => reject(r.error);
  });
}
/** Run f in a transaction over the stores; resolves with its request's result when the transaction completes. */
async function tx<T>(stores: string[], mode: IDBTransactionMode, f: (t: IDBTransaction) => IDBRequest<T>): Promise<T> {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const t = db.transaction(stores, mode), req = f(t);
    t.oncomplete = (): void => { db.close(); resolve(req.result); };
    t.onerror = (): void => { db.close(); reject(t.error ?? req.error); };
    t.onabort = (): void => { db.close(); reject(t.error ?? new Error('aborted')); };
  });
}
export async function storeSave(meta: SaveMeta, data: ArrayBuffer): Promise<SaveRecord> {
  const rec: SaveRecord = { id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`, kind: meta.kind, title: meta.title, created: Date.now(), bytes: data.byteLength, data };
  await tx(['saves', 'meta'], 'readwrite', (t) => { t.objectStore('saves').put(rec); return t.objectStore('meta').put(info(rec)); });
  return rec;
}
/** Saves of one kind, newest first, without their data (read from 'meta' only). */
export async function listSaves(kind: 'global' | 'regional'): Promise<SaveInfo[]> {
  const all = await tx(['meta'], 'readonly', (t) => t.objectStore('meta').getAll() as IDBRequest<SaveInfo[]>);
  return all.filter((r) => r.kind === kind).sort((a, b) => b.created - a.created);
}
export async function loadSave(id: string): Promise<SaveRecord> {
  const r = await tx(['saves'], 'readonly', (t) => t.objectStore('saves').get(id) as IDBRequest<SaveRecord | undefined>);
  if (!r) throw new Error('找不到存檔 / save not found');
  return r;
}
export async function deleteSave(id: string): Promise<void> {
  await tx(['saves', 'meta'], 'readwrite', (t) => { t.objectStore('saves').delete(id); return t.objectStore('meta').delete(id); });
}

// ---------------------------------------------------------------- ZIP (one deflated entry)
// CRC-32 by 8 bytes at a time (a byte-at-a-time JS loop takes seconds on a few hundred MB), incremental over chunks
const CRC8 = ((): Uint32Array => {
  const t = new Uint32Array(8 * 256);
  for (let n = 0; n < 256; n++) { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1; t[n] = c >>> 0; }
  for (let n = 0; n < 256; n++) { let c = t[n]!; for (let k = 1; k < 8; k++) { c = t[c & 255]! ^ (c >>> 8); t[k * 256 + n] = c >>> 0; } }
  return t;
})();
/** CRC state after more bytes (start from 0xffffffff, finish with (c ^ 0xffffffff) >>> 0). */
function crcUpdate(crc: number, d: Uint8Array): number {
  let c = crc, i = 0;
  const n = d.length, n8 = n - (n & 7), T = CRC8;
  for (; i < n8; i += 8) {
    const a = (d[i]! | (d[i + 1]! << 8) | (d[i + 2]! << 16) | (d[i + 3]! << 24)) ^ c, b = d[i + 4]! | (d[i + 5]! << 8) | (d[i + 6]! << 16) | (d[i + 7]! << 24);
    c = T[1792 + (a & 255)]! ^ T[1536 + ((a >>> 8) & 255)]! ^ T[1280 + ((a >>> 16) & 255)]! ^ T[1024 + (a >>> 24)]! ^ T[768 + (b & 255)]! ^ T[512 + ((b >>> 8) & 255)]! ^ T[256 + ((b >>> 16) & 255)]! ^ T[b >>> 24]!;
  }
  for (; i < n; i++) c = T[(c ^ d[i]!) & 255]! ^ (c >>> 8);
  return c;
}
const crc32 = (d: Uint8Array): number => (crcUpdate(0xffffffff, d) ^ 0xffffffff) >>> 0;
async function streamBytes(input: Uint8Array, t: DecompressionStream): Promise<Uint8Array> {
  const s = new Blob([input]).stream().pipeThrough(t);
  return new Uint8Array(await new Response(s).arrayBuffer());
}
/** Let the page paint (a message set just before a long job must be on screen before the job starts). */
export const paint = (): Promise<void> => new Promise((r) => { const t = setTimeout(r, 60); requestAnimationFrame(() => { clearTimeout(t); setTimeout(r, 0); }); });
/**
 * Deflate in 4 MB pieces (through a CompressionStream, off the main thread) with the CRC computed along the way and the
 * progress reported (done / total bytes): the output stays a list of pieces for the Blob (no copy of the whole).
 */
async function deflateChunks(raw: Uint8Array, onProgress?: (done: number, total: number) => void): Promise<{ parts: Uint8Array[]; size: number; crc: number }> {
  const cs = new CompressionStream('deflate-raw'), writer = cs.writable.getWriter(), reader = cs.readable.getReader();
  const parts: Uint8Array[] = []; let size = 0;
  const drain = (async (): Promise<void> => { for (;;) { const { done, value } = await reader.read(); if (done) return; parts.push(value); size += value.length; } })();
  let c = 0xffffffff;
  const CH = 4 << 20;
  for (let o = 0; o < raw.length; o += CH) {
    const piece = raw.subarray(o, Math.min(raw.length, o + CH));
    c = crcUpdate(c, piece);
    await writer.write(piece);
    onProgress?.(Math.min(raw.length, o + CH), raw.length);
  }
  await writer.close(); await drain;
  return { parts, size, crc: (c ^ 0xffffffff) >>> 0 };
}
export async function makeZip(entryName: string, data: ArrayBuffer, onProgress?: (done: number, total: number) => void): Promise<Blob> {
  const raw = new Uint8Array(data), { parts, size, crc } = await deflateChunks(raw, onProgress);
  const name = new TextEncoder().encode(entryName);
  const local = new DataView(new ArrayBuffer(30));
  local.setUint32(0, 0x04034b50, true); local.setUint16(4, 20, true); local.setUint16(8, 8, true);
  local.setUint16(12, 33, true);                          // DOS date 1980-01-01 (time of day 0)
  local.setUint32(14, crc, true); local.setUint32(18, size, true); local.setUint32(22, raw.length, true); local.setUint16(26, name.length, true);
  const central = new DataView(new ArrayBuffer(46));
  central.setUint32(0, 0x02014b50, true); central.setUint16(4, 20, true); central.setUint16(6, 20, true); central.setUint16(10, 8, true);
  central.setUint16(14, 33, true);
  central.setUint32(16, crc, true); central.setUint32(20, size, true); central.setUint32(24, raw.length, true); central.setUint16(28, name.length, true);
  central.setUint32(42, 0, true);
  const end = new DataView(new ArrayBuffer(22));
  end.setUint32(0, 0x06054b50, true); end.setUint16(8, 1, true); end.setUint16(10, 1, true);
  end.setUint32(12, 46 + name.length, true); end.setUint32(16, 30 + name.length + size, true);
  return new Blob([local, name, ...parts, central, name, end], { type: 'application/zip' });
}
export async function readZip(buf: ArrayBuffer): Promise<ArrayBuffer> {
  const dv = new DataView(buf);
  if (dv.getUint32(0, true) !== 0x04034b50) {
    // not a zip: accept a raw save container too
    if (dv.getUint32(0, true) === MAGIC) return buf;
    throw new Error('不是 ZIP 存檔 / not a ZIP save file');
  }
  const method = dv.getUint16(8, true), csize = dv.getUint32(18, true), nl = dv.getUint16(26, true), xl = dv.getUint16(28, true);
  const body = new Uint8Array(buf, 30 + nl + xl, csize);
  const out = method === 0 ? body.slice() : method === 8 ? await streamBytes(body, new DecompressionStream('deflate-raw')) : null;
  if (!out) throw new Error(`不支援的壓縮方式 / unsupported ZIP method ${method}`);
  if (crc32(out) !== dv.getUint32(14, true)) throw new Error('存檔損毀（CRC 不符）/ corrupted save (CRC mismatch)');
  return out.buffer;
}

// ---------------------------------------------------------------- export
interface Downloads { save(r: { filename: string; data: Blob }): Promise<{ status: string }> }
/** Offer a file with the artifact's download capability; null when this page has none. */
export async function offerDownload(filename: string, data: Blob): Promise<string | null> {
  const cl = (globalThis as { claude?: { use(n: string): Promise<unknown> } }).claude;
  const dl = cl ? await cl.use('downloads').catch(() => null) as Downloads | null : null;
  if (!dl) return null;
  try { await dl.save({ filename, data }); return '已匯出 / exported'; }
  catch (e) { const c = (e as { code?: string }).code; if (c === 'declined') return '已取消 / cancelled'; throw new Error(`匯出失敗 / export failed: ${c ?? String(e)}`); }
}
export async function exportSave(rec: SaveRecord, progress: (text: string) => void = () => {}): Promise<string> {
  const safe = rec.title.replace(/[^\p{L}\p{N}._ -]+/gu, '_').slice(0, 80);
  const filename = `${safe || 'simulation'}.zip`;
  progress('壓縮 0 % / compressing 0 %');
  let last = 0;
  const zip = await makeZip('cyclone-save.bin', rec.data, (d, t) => { const now = performance.now(); if (now - last > 150 || d === t) { last = now; progress(`壓縮 ${Math.round(100 * d / t)} %（${fmtBytes(d)} / ${fmtBytes(t)}）/ compressing`); } });
  progress(`交給下載（${fmtBytes(zip.size)}）… / handing over the download`);
  const own = await offerDownload(filename, zip);
  if (own) return own;
  // embedded in the global page (regional overlay): let the host page offer the file
  if (window.parent !== window) {
    const id = Math.random().toString(36).slice(2);
    progress('等待下載視窗… / waiting for the download dialog');
    const answer = new Promise<string | null>((resolve) => {
      const on = (ev: MessageEvent): void => { if (ev.source === window.parent && ev.data?.type === 'export-save-done' && ev.data.id === id) { removeEventListener('message', on); resolve(ev.data.result as string | null); } };
      addEventListener('message', on);
      setTimeout(() => { removeEventListener('message', on); resolve(null); }, 120000);
    });
    window.parent.postMessage({ type: 'export-save', id, filename, data: zip }, '*');
    const r = await answer;
    if (r) return r;
  }
  const a = document.createElement('a');
  a.href = URL.createObjectURL(zip); a.download = filename;
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 30000);
  return '已匯出 / exported';
}

export const fmtBytes = (n: number): string => n > 1e6 ? `${(n / 1e6).toFixed(1)} MB` : `${Math.max(1, Math.round(n / 1e3))} kB`;

/** Render a saves panel into `root`: list with load / export / delete, a save button and an import
 *  control. `capture` produces the current simulation's save; `onLoad` restores one. Every job shows what it is doing in a
 *  status line at once (and keeps the buttons from being pressed again meanwhile); deleting asks in the page (a
 *  confirm() dialog is never shown inside an artifact, where it answers "no" by itself). */
export function mountSavesPanel(root: HTMLElement, kind: 'global' | 'regional',
  capture: () => Promise<{ meta: SaveMeta; data: ArrayBuffer }>, onLoad: (meta: SaveMeta, data: ArrayBuffer) => void, log: (s: string) => void): { refresh(): Promise<void> } {
  root.innerHTML = `<div class="controls"><button data-a="save">存檔目前模擬 / Save now</button><label class="filebtn"><input type="file" accept=".zip,.bin" hidden />匯入存檔 / Import</label></div>
<div class="muted savestatus" role="status" style="min-height:1.3em;color:#8fc1ff"></div>
<div class="muted">存檔保存在這個瀏覽器；清除網站資料會一起刪除，重要的請「匯出」成檔案。/ Saves live in this browser and are deleted with the site data; export the important ones.</div>
<div class="saves"></div>`;
  const listEl = root.querySelector('.saves') as HTMLElement, statusEl = root.querySelector('.savestatus') as HTMLElement;
  const saveBtn = root.querySelector('[data-a="save"]') as HTMLButtonElement, input = root.querySelector('input[type=file]') as HTMLInputElement;
  let working = false;
  const buttons = (): HTMLButtonElement[] => Array.from(root.querySelectorAll<HTMLButtonElement>("button"));
  const status = (t: string): void => { statusEl.textContent = t; };
  /** one job at a time: status at once, buttons off while it runs, the time it took at the end */
  const job = async (what: string, f: (step: (s: string) => void) => Promise<string | void>): Promise<void> => {
    if (working) return;
    working = true; for (const b of buttons()) b.disabled = true; input.disabled = true;
    const t0 = performance.now(), step = (s: string): void => status(`${what}：${s}`);
    status(`${what}… / working`); await paint();
    try { const r = await f(step); log(`${r ?? what} · ${((performance.now() - t0) / 1000).toFixed(1)} s`); }
    catch (e) { log(`${what}失敗 / failed: ${String((e as Error).message ?? e)}`); }
    working = false; input.disabled = false; status('');
    for (const b of buttons()) b.disabled = false;
  };
  const refresh = async (): Promise<void> => {
    let items: SaveInfo[] = [];
    try { items = await listSaves(kind); } catch (e) { listEl.textContent = `無法使用瀏覽器儲存空間 / browser storage unavailable: ${String((e as Error).message ?? e)}`; return; }
    listEl.replaceChildren();
    if (!items.length) { listEl.textContent = '（尚無存檔 / no saves yet）'; return; }
    for (const it of items) {
      const row = document.createElement('div'); row.className = 'saverow';
      const t = document.createElement('div'); t.className = 'savetitle';
      t.textContent = it.title;
      const sub = document.createElement('div'); sub.className = 'muted';
      sub.textContent = `${new Date(it.created).toLocaleString()} · ${fmtBytes(it.bytes)}`;
      const btns = document.createElement('div'); btns.className = 'controls';
      for (const [a, label] of [['load', '載入 / Load'], ['export', '匯出 / Export'], ['del', '刪除 / Delete']] as const) {
        const b = document.createElement('button'); b.textContent = label;
        let armed = 0;
        b.onclick = (): void => {
          if (a === 'del') {
            // first press arms (the button asks), the second within 4 s deletes
            if (!armed) { armed = window.setTimeout(() => { armed = 0; b.textContent = label; b.style.background = ''; }, 4000); b.textContent = '再按一次確定刪除 / press again to delete'; b.style.background = '#7a2020'; return; }
            clearTimeout(armed); armed = 0;
            void job(`刪除「${it.title}」`, async () => { await deleteSave(it.id); await refresh(); return `已刪除 / deleted: ${it.title}`; });
            return;
          }
          void job(a === 'load' ? `載入「${it.title}」` : `匯出「${it.title}」`, async (step) => {
            step('讀取存檔… / reading the save'); await paint();
            const rec = await loadSave(it.id);
            if (a === 'load') { step('還原模式… / restoring'); await paint(); onLoad(unpackSave(rec.data).meta, rec.data); return `已載入 / loaded: ${rec.title}`; }
            return `${await exportSave(rec, step)}: ${rec.title}`;
          });
        };
        btns.appendChild(b);
      }
      row.append(t, sub, btns);
      listEl.appendChild(row);
    }
  };
  saveBtn.onclick = (): void => {
    void job('存檔', async (step) => {
      step('擷取模式狀態（等目前這步算完）… / capturing the model state');
      const { meta, data } = await capture();
      step(`寫入瀏覽器儲存空間（${fmtBytes(data.byteLength)}）… / writing to browser storage`); await paint();
      await storeSave(meta, data); await refresh();
      return `已存檔 / saved: ${meta.title} (${fmtBytes(data.byteLength)})`;
    });
  };
  input.onchange = (): void => {
    const f = input.files?.[0]; input.value = '';
    if (!f) return;
    void job('匯入', async (step) => {
      step(`讀取檔案（${fmtBytes(f.size)}）… / reading the file`);
      const data = await readZip(await f.arrayBuffer());
      step('檢查內容… / checking'); await paint();
      const { meta } = unpackSave(data);
      if (meta.kind !== kind) throw new Error(meta.kind === 'replay' ? '這是回放資料，請用「回放」面板的「載入回放資料」 / this is replay data: use "Load replay data" in the replay panel' : meta.kind === 'global' ? '這是全球模式的存檔，請在全球模式頁面匯入 / this is a global-model save' : '這是區域模式的存檔，請在區域模式頁面匯入 / this is a regional-model save');
      step('寫入瀏覽器儲存空間… / writing to browser storage');
      await storeSave(meta, data); await refresh();
      return `已匯入 / imported: ${meta.title}`;
    });
  };
  void refresh();
  return { refresh };
}
