// Saved simulations, shared by the global and the regional page.
//
// * container: 'CYCS' magic, u32 version, u32 header length, header JSON (UTF-8, padded to 4 bytes),
//   then the arrays listed in the header (Float32 or raw bytes), each padded to 4 bytes
// * storage: IndexedDB in this browser (can be cleared with the site data)
// * export / import: a ZIP file with one deflate-compressed entry (the only archive type the artifact
//   host lets a page offer); the artifact's download capability when present, a normal browser
//   download otherwise

export interface SaveArrays { [name: string]: Float32Array | Uint8Array }
export interface SaveMeta { kind: 'global' | 'regional'; title: string; [k: string]: unknown }
export interface SaveRecord { id: string; kind: 'global' | 'regional'; title: string; created: number; bytes: number; data: ArrayBuffer }

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
function openDb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const r = indexedDB.open('cyclone-on-a-chip-saves', 1);
    r.onupgradeneeded = (): void => { r.result.createObjectStore('saves', { keyPath: 'id' }); };
    r.onsuccess = (): void => resolve(r.result);
    r.onerror = (): void => reject(r.error);
  });
}
async function tx<T>(mode: IDBTransactionMode, f: (s: IDBObjectStore) => IDBRequest<T>): Promise<T> {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const t = db.transaction('saves', mode), req = f(t.objectStore('saves'));
    req.onsuccess = (): void => resolve(req.result);
    req.onerror = (): void => reject(req.error);
  });
}
export async function storeSave(meta: SaveMeta, data: ArrayBuffer): Promise<SaveRecord> {
  const rec: SaveRecord = { id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`, kind: meta.kind, title: meta.title, created: Date.now(), bytes: data.byteLength, data };
  await tx('readwrite', (s) => s.put(rec));
  return rec;
}
/** Saves of one kind, newest first, without their data. */
export async function listSaves(kind: 'global' | 'regional'): Promise<Omit<SaveRecord, 'data'>[]> {
  const all = await tx('readonly', (s) => s.getAll() as IDBRequest<SaveRecord[]>);
  return all.filter((r) => r.kind === kind).sort((a, b) => b.created - a.created).map(({ data: _d, ...r }) => r);
}
export async function loadSave(id: string): Promise<SaveRecord> {
  const r = await tx('readonly', (s) => s.get(id) as IDBRequest<SaveRecord | undefined>);
  if (!r) throw new Error('找不到存檔 / save not found');
  return r;
}
export async function deleteSave(id: string): Promise<void> { await tx('readwrite', (s) => s.delete(id)); }

// ---------------------------------------------------------------- ZIP (one deflated entry)
const CRC_TABLE = ((): Uint32Array => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1; t[n] = c >>> 0; }
  return t;
})();
function crc32(d: Uint8Array): number {
  let c = 0xffffffff;
  for (let i = 0; i < d.length; i++) c = CRC_TABLE[(c ^ d[i]!) & 255]! ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}
async function streamBytes(input: Uint8Array, t: CompressionStream | DecompressionStream): Promise<Uint8Array> {
  const s = new Blob([input]).stream().pipeThrough(t);
  return new Uint8Array(await new Response(s).arrayBuffer());
}
export async function makeZip(entryName: string, data: ArrayBuffer): Promise<Blob> {
  const raw = new Uint8Array(data), comp = await streamBytes(raw, new CompressionStream('deflate-raw'));
  const name = new TextEncoder().encode(entryName), crc = crc32(raw);
  const local = new DataView(new ArrayBuffer(30));
  local.setUint32(0, 0x04034b50, true); local.setUint16(4, 20, true); local.setUint16(8, 8, true);
  local.setUint16(12, 33, true);                          // DOS date 1980-01-01 (time of day 0)
  local.setUint32(14, crc, true); local.setUint32(18, comp.length, true); local.setUint32(22, raw.length, true); local.setUint16(26, name.length, true);
  const central = new DataView(new ArrayBuffer(46));
  central.setUint32(0, 0x02014b50, true); central.setUint16(4, 20, true); central.setUint16(6, 20, true); central.setUint16(10, 8, true);
  central.setUint16(14, 33, true);
  central.setUint32(16, crc, true); central.setUint32(20, comp.length, true); central.setUint32(24, raw.length, true); central.setUint16(28, name.length, true);
  central.setUint32(42, 0, true);
  const end = new DataView(new ArrayBuffer(22));
  end.setUint32(0, 0x06054b50, true); end.setUint16(8, 1, true); end.setUint16(10, 1, true);
  end.setUint32(12, 46 + name.length, true); end.setUint32(16, 30 + name.length + comp.length, true);
  return new Blob([local, name, comp, central, name, end], { type: 'application/zip' });
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
export async function exportSave(rec: SaveRecord): Promise<string> {
  const safe = rec.title.replace(/[^\p{L}\p{N}._ -]+/gu, '_').slice(0, 80);
  const filename = `${safe || 'simulation'}.zip`;
  const zip = await makeZip('cyclone-save.bin', rec.data);
  const own = await offerDownload(filename, zip);
  if (own) return own;
  // embedded in the global page (regional overlay): let the host page offer the file
  if (window.parent !== window) {
    const id = Math.random().toString(36).slice(2);
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
 *  control. `capture` produces the current simulation's save; `onLoad` restores one. */
export function mountSavesPanel(root: HTMLElement, kind: 'global' | 'regional',
  capture: () => Promise<{ meta: SaveMeta; data: ArrayBuffer }>, onLoad: (meta: SaveMeta, data: ArrayBuffer) => void, log: (s: string) => void): { refresh(): Promise<void> } {
  root.innerHTML = `<div class="controls"><button data-a="save">存檔目前模擬 / Save now</button><label class="filebtn"><input type="file" accept=".zip,.bin" hidden />匯入存檔 / Import</label></div>
<div class="muted">存檔保存在這個瀏覽器；清除網站資料會一起刪除，重要的請「匯出」成檔案。/ Saves live in this browser and are deleted with the site data; export the important ones.</div>
<div class="saves"></div>`;
  const listEl = root.querySelector('.saves') as HTMLElement;
  const refresh = async (): Promise<void> => {
    let items: Omit<SaveRecord, 'data'>[] = [];
    try { items = await listSaves(kind); } catch (e) { listEl.textContent = `無法使用瀏覽器儲存空間 / browser storage unavailable: ${String(e)}`; return; }
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
        b.onclick = async (): Promise<void> => {
          try {
            if (a === 'del') { if (!confirm(`刪除「${it.title}」？/ Delete this save?`)) return; await deleteSave(it.id); await refresh(); return; }
            const rec = await loadSave(it.id);
            if (a === 'load') { log(`載入存檔 / Loading save: ${rec.title}`); onLoad(unpackSave(rec.data).meta, rec.data); }
            else log(await exportSave(rec));
          } catch (e) { log(String((e as Error).message ?? e)); }
        };
        btns.appendChild(b);
      }
      row.append(t, sub, btns);
      listEl.appendChild(row);
    }
  };
  (root.querySelector('[data-a="save"]') as HTMLButtonElement).onclick = async (ev): Promise<void> => {
    const b = ev.currentTarget as HTMLButtonElement;
    b.disabled = true;
    try { const { meta, data } = await capture(); await storeSave(meta, data); log(`已存檔 / Saved: ${meta.title} (${fmtBytes(data.byteLength)})`); await refresh(); }
    catch (e) { log(`存檔失敗 / save failed: ${String((e as Error).message ?? e)}`); }
    b.disabled = false;
  };
  const input = root.querySelector('input[type=file]') as HTMLInputElement;
  input.onchange = async (): Promise<void> => {
    const f = input.files?.[0]; input.value = '';
    if (!f) return;
    try {
      const data = await readZip(await f.arrayBuffer());
      const { meta } = unpackSave(data);
      if (meta.kind !== kind) throw new Error(meta.kind === 'global' ? '這是全球模式的存檔，請在全球模式頁面匯入 / this is a global-model save' : '這是區域模式的存檔，請在區域模式頁面匯入 / this is a regional-model save');
      await storeSave(meta, data); log(`已匯入 / Imported: ${meta.title}`); await refresh();
    } catch (e) { log(String((e as Error).message ?? e)); }
  };
  void refresh();
  return { refresh };
}
