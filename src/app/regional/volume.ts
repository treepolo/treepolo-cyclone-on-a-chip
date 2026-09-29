// WebGL2 volume renderer for the regional model: front-to-back ray marching through a 3-D texture
// (R = cloud water, G = rain water) with single-scattering sun lighting and self-shadowing,
// a coloured ground plane (surface field) and an orbit camera. Vertical scale is exaggerated.
// Tracer particles (with short trails) are drawn first into an offscreen buffer that keeps, per pixel, the
// nearest particle colour and its distance from the eye; the ray march composites that colour where the
// ray passes that distance, so clouds in front hide the particles behind them.

const VS = `#version 300 es
in vec2 aPos; out vec2 vUv;
void main(){ vUv = aPos*0.5+0.5; gl_Position = vec4(aPos,0.0,1.0); }`;

const FS = `#version 300 es
precision highp float; precision highp sampler3D;
in vec2 vUv; out vec4 o;
uniform sampler3D uVol; uniform sampler2D uGround;
uniform mat4 uInvVP; uniform vec3 uEye; uniform vec3 uBox; uniform vec3 uSun;
uniform float uCloudK; uniform float uRainK;
uniform sampler2D uTrC; uniform sampler2D uTrD; uniform int uTrOn;
bool hitBox(vec3 ro, vec3 rd, out float t0, out float t1){
  vec3 inv = 1.0/rd; vec3 a = (vec3(0.0)-ro)*inv; vec3 b = (uBox-ro)*inv;
  vec3 mn = min(a,b), mx = max(a,b);
  t0 = max(max(mn.x,mn.y),mn.z); t1 = min(min(mx.x,mx.y),mx.z);
  return t1 > max(t0,0.0);
}
// cloud extinction grows with the square root of the condensate (the byte is sqrt(q / 3 g/kg)), so thin ice cloud and
// cirrus stay visible next to dense cores; precipitation linearly
vec2 dens(vec3 p){ vec2 s = texture(uVol, p/uBox).rg; return vec2(s.r*uCloudK, s.g*s.g*uRainK); }
void main(){
  vec4 ndc = vec4(vUv*2.0-1.0, 1.0, 1.0);
  vec4 wp = uInvVP*ndc; vec3 rd = normalize(wp.xyz/wp.w - uEye); vec3 ro = uEye;
  vec3 sky = mix(vec3(0.62,0.74,0.88), vec3(0.18,0.3,0.52), clamp(rd.z*1.2+0.2,0.0,1.0));
  vec3 col = vec3(0.0); float trans = 1.0;
  float t0, t1;
  vec3 bg = sky;
  // ground plane z = 0 inside the domain footprint
  if (rd.z < 0.0) {
    float tg = -ro.z/rd.z; vec3 pg = ro + tg*rd;
    if (pg.x >= 0.0 && pg.y >= 0.0 && pg.x <= uBox.x && pg.y <= uBox.y) bg = texture(uGround, pg.xy/uBox.xy).rgb;
    else bg = vec3(0.07,0.09,0.12);
  }
  vec4 tc = vec4(0.0); float tdist = 1e9;
  if (uTrOn == 1) { tc = texture(uTrC, vUv); if (tc.a > 0.0) { vec4 d4 = texture(uTrD, vUv); tdist = (d4.r * 65280.0 + d4.g * 255.0) / 65535.0 * 16.0; } }
  bool tdone = tc.a <= 0.0;
  if (hitBox(ro, rd, t0, t1)) {
    t0 = max(t0, 0.0);
    const int N = 160;
    float dt = (t1-t0)/float(N);
    for (int i = 0; i < N; i++) {
      float t = t0 + (float(i)+0.5)*dt;
      if (!tdone && t > tdist) { col += trans*tc.rgb; trans *= 1.0 - tc.a; tdone = true; }
      vec3 p = ro + t*rd;
      vec2 d = dens(p);
      float ext = d.x + d.y;
      if (ext > 1e-3) {
        // shadow toward the sun
        float od = 0.0;
        for (int s = 1; s <= 6; s++) { vec3 ps = p + uSun*(float(s)*0.04*uBox.x); if (any(lessThan(ps,vec3(0.0))) || any(greaterThan(ps,uBox))) break; vec2 ds = dens(ps); od += (ds.x+ds.y)*0.04*uBox.x; }
        float light = exp(-od*1.2);
        vec3 c = mix(vec3(0.96,0.97,1.0), vec3(0.35,0.5,0.8), d.y/(ext+1e-6));
        c *= 0.35 + 0.75*light;
        float a = 1.0 - exp(-ext*dt);
        col += trans*a*c; trans *= 1.0-a;
        if (trans < 0.01) break;
      }
    }
  }
  if (!tdone) { col += trans*tc.rgb; trans *= 1.0 - tc.a; }
  o = vec4(col + trans*bg, 1.0);
}`;

// tracer lines / points: colour by height (orange near the ground, pale yellow aloft), alpha fades along the trail
const TVS = `#version 300 es
in vec4 aP; uniform mat4 uVP; uniform vec3 uEye; uniform vec3 uBox; uniform float uPt;
out float vA; out float vD; out float vH;
void main(){ vec3 p = aP.xyz; gl_Position = uVP * vec4(p, 1.0); gl_PointSize = uPt; vA = aP.w; vD = distance(p, uEye); vH = p.z / uBox.z; }`;
const TFS = `#version 300 es
precision highp float;
in float vA; in float vD; in float vH;
layout(location=0) out vec4 oC; layout(location=1) out vec4 oD;
void main(){
  vec3 c = mix(vec3(0.90, 0.38, 0.14), vec3(1.0, 0.94, 0.66), clamp(vH / 0.6, 0.0, 1.0));
  oC = vec4(c * vA, vA);
  float v = floor(clamp(vD / 16.0, 0.0, 1.0) * 65535.0 + 0.5), h = floor(v / 256.0);
  oD = vec4(h / 255.0, (v - h * 256.0) / 255.0, 0.0, 1.0);
}`;
const TRAIL = 8;

/** orbit (drag to turn), free flight, or the fixed side and top views */
export type CameraMode = 'orbit' | 'fly' | 'side' | 'top';

export class VolumeView {
  private readonly gl: WebGL2RenderingContext;
  private readonly prog: WebGLProgram;
  private readonly vol: WebGLTexture;
  private readonly groundTex: WebGLTexture;
  private box: [number, number, number] = [1, 1, 0.3];
  // tracer particles
  private trProg: WebGLProgram | null = null;
  private trBuf: WebGLBuffer | null = null;
  private trFbo: { fb: WebGLFramebuffer; c: WebGLTexture; d: WebGLTexture; z: WebGLRenderbuffer; w: number; h: number } | null = null;
  private trHist: Float32Array | null = null; private trCnt: Uint8Array | null = null; private trAge: Float32Array | null = null;
  private trLines = 0; private trPoints = 0;
  private quad: WebGLBuffer | null = null;
  yaw = -0.9; pitch = 0.35; dist = 1.35;

  constructor(private readonly canvas: HTMLCanvasElement) {
    const gl = canvas.getContext('webgl2');
    if (!gl) throw new Error('WebGL2 不可用 / WebGL2 unavailable');
    this.gl = gl;
    const sh = (t: number, s: string): WebGLShader => { const x = gl.createShader(t)!; gl.shaderSource(x, s); gl.compileShader(x); if (!gl.getShaderParameter(x, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(x) ?? 'shader'); return x; };
    const p = gl.createProgram()!;
    gl.attachShader(p, sh(gl.VERTEX_SHADER, VS)); gl.attachShader(p, sh(gl.FRAGMENT_SHADER, FS)); gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p) ?? 'link');
    this.prog = p;
    try {
      const tp = gl.createProgram()!;
      gl.attachShader(tp, sh(gl.VERTEX_SHADER, TVS)); gl.attachShader(tp, sh(gl.FRAGMENT_SHADER, TFS)); gl.linkProgram(tp);
      if (gl.getProgramParameter(tp, gl.LINK_STATUS)) { this.trProg = tp; this.trBuf = gl.createBuffer(); }
    } catch { this.trProg = null; }
    const vb = gl.createBuffer()!;
    this.quad = vb;
    gl.bindBuffer(gl.ARRAY_BUFFER, vb);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
    const loc = gl.getAttribLocation(p, 'aPos');
    gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
    this.vol = gl.createTexture()!;
    this.groundTex = gl.createTexture()!;
    this.controls();
  }

  // ---------------------------------------------------------------- camera controls
  // Orbit camera around a movable target: drag to turn, right-drag / Shift-drag / two fingers to pan, wheel or pinch to
  // zoom, double-click / double-tap to fly to a point on the ground, arrow keys / WASD to pan. Free flight: WASD / QE or
  // the on-screen buttons move, dragging turns the head.
  /** orbit target (box units); null: the domain centre */
  private tgt: V3 | null = null;
  /** inverse view-projection and eye of the last drawn frame (picking) */
  private lastInv: Float32Array | null = null; private lastEye: V3 = [0, 0, 0]; private lastVP: Float32Array | null = null;
  /** tap on the view (no drag): screen position, for interaction tools; returns true when it used the tap */
  onTap: ((clientX: number, clientY: number) => boolean) | null = null;
  /** an interaction tool is active: one-finger / left drags go to onToolDrag (painting), right drags rotate, Shift drags pan */
  toolActive = false;
  onToolDrag: ((clientX: number, clientY: number, phase: 'start' | 'move' | 'end') => void) | null = null;
  /** pointer moving over the view without a button (mouse), or leaving it (null): for a tool's preview */
  onHover: ((clientX: number, clientY: number) => void) | null = null;
  onLeave: (() => void) | null = null;
  private controls(): void {
    const canvas = this.canvas, pts = new Map<number, { x: number; y: number }>();
    let mode: 'rotate' | 'pan' | 'tool' = 'rotate', moved = 0, spread0 = 0, cx0 = 0, cy0 = 0, lastTap = 0;
    const spreadC = (): [number, number, number] => { const p = [...pts.values()]; if (p.length < 2) return [0, 0, 0]; return [Math.hypot(p[0]!.x - p[1]!.x, p[0]!.y - p[1]!.y), (p[0]!.x + p[1]!.x) / 2, (p[0]!.y + p[1]!.y) / 2]; };
    canvas.addEventListener('contextmenu', (e) => e.preventDefault());
    canvas.addEventListener('pointerdown', (e) => {
      canvas.setPointerCapture(e.pointerId);
      pts.set(e.pointerId, { x: e.clientX, y: e.clientY });
      if (pts.size === 1) {
        moved = 0;
        if (this.toolActive) mode = e.button === 0 && !e.shiftKey ? 'tool' : e.button === 2 ? 'rotate' : 'pan';
        else mode = e.button === 2 || e.shiftKey || e.button === 1 ? 'pan' : 'rotate';
        if (mode === 'tool') this.onToolDrag?.(e.clientX, e.clientY, 'start');
      }
      if (pts.size === 2) { [spread0, cx0, cy0] = spreadC(); moved = 99; }
    });
    const end = (e: PointerEvent): void => {
      const was = pts.get(e.pointerId);
      pts.delete(e.pointerId);
      if (was && mode === 'tool' && pts.size === 0) this.onToolDrag?.(e.clientX, e.clientY, 'end');
      if (!was || pts.size > 0 || moved >= 6) return;
      const now = performance.now();
      // double tap: fly to the point (not with a tool: a second tap is a second placement)
      if (!this.toolActive && now - lastTap < 350) { lastTap = 0; this.focusAt(e.clientX, e.clientY); return; }
      lastTap = now;
      this.onTap?.(e.clientX, e.clientY);
    };
    canvas.addEventListener('pointerup', end);
    canvas.addEventListener('pointercancel', end);
    canvas.addEventListener('pointerleave', () => { if (!pts.size) this.onLeave?.(); });
    canvas.addEventListener('pointermove', (e) => {
      const p = pts.get(e.pointerId);
      if (!p) { if (e.pointerType === 'mouse') this.onHover?.(e.clientX, e.clientY); return; }
      const dx = e.clientX - p.x, dy = e.clientY - p.y;
      p.x = e.clientX; p.y = e.clientY;
      if (pts.size >= 2) {
        const [s, cx, cy] = spreadC();
        if (spread0 > 0 && s > 0) this.zoomBy(spread0 / s);
        this.pan(cx - cx0, cy - cy0);
        spread0 = s; cx0 = cx; cy0 = cy;
        return;
      }
      moved += Math.abs(dx) + Math.abs(dy);
      if (mode === 'tool') { this.onToolDrag?.(e.clientX, e.clientY, 'move'); return; }
      if (mode === 'pan' && !this.fly) this.pan(dx, dy);
      else this.turn(dx, dy);
    });
    canvas.addEventListener('wheel', (e) => { e.preventDefault(); this.zoomBy(Math.exp(e.deltaY * 0.0012)); }, { passive: false });
    // keys: free flight WASD / QE (Shift faster); orbit: arrows / WASD pan, Q / E zoom (ignored while typing in a form field)
    const typing = (): boolean => { const a = document.activeElement; return !!a && (a.tagName === 'INPUT' || a.tagName === 'SELECT' || a.tagName === 'TEXTAREA'); };
    const KEYS = ['w', 'a', 's', 'd', 'q', 'e', 'arrowup', 'arrowdown', 'arrowleft', 'arrowright', 'shift'];
    window.addEventListener('keydown', (e) => {
      const k = e.key.toLowerCase();
      if (typing() || !KEYS.includes(k) || this.canvas.hidden) return;
      this.keys.add(k); if (k !== 'shift') e.preventDefault(); this.dirty = true;
    });
    window.addEventListener('keyup', (e) => { this.keys.delete(e.key.toLowerCase()); });
    window.addEventListener('blur', () => this.keys.clear());
  }
  /** press / release a movement key from an on-screen button (w a s d q e) */
  setKey(k: string, down: boolean): void { if (down) this.keys.add(k); else this.keys.delete(k); this.dirty = true; }
  private target(): V3 { const [bx, by, bz] = this.box; return this.tgt ?? [bx / 2, by / 2, bz * 0.3]; }
  private turn(dx: number, dy: number): void {
    if (this.fly) { this.fly.yaw -= dx * 0.004; this.fly.pitch = Math.max(-1.45, Math.min(1.45, this.fly.pitch - dy * 0.004)); }
    else { this.yaw -= dx * 0.006; this.pitch = Math.max(0.02, Math.min(Math.PI / 2, this.pitch + dy * 0.006)); }
    this.dirty = true;
  }
  private zoomBy(f: number): void {
    if (this.fly) this.moveFly(0.6 * (1 - f) * Math.max(0.05, this.fly.eye[2] * 4), 0, 0);   // toward the screen (f < 1) flies forward
    else this.dist = Math.max(0.03, Math.min(6, this.dist * f));
    this.dirty = true;
  }
  /** move the orbit target by a screen-space drag (pixels): the point under the cursor follows it */
  private pan(dx: number, dy: number): void {
    if (this.fly) { this.moveFly(0, -dx * 0.002, dy * 0.002); return; }
    const t = this.target(), k = this.dist * 1.2 / Math.max(1, this.canvas.clientHeight);
    const rx = Math.sin(this.yaw), ry = -Math.cos(this.yaw);                   // screen right on the ground
    const fx = -Math.cos(this.yaw), fy = -Math.sin(this.yaw);                  // away from the eye on the ground
    const up = Math.max(0.2, Math.sin(this.pitch));                            // looking down: vertical drags move along the ground
    this.tgt = [t[0] - k * dx * rx + k * dy * fx / up, t[1] - k * dx * ry + k * dy * fy / up, t[2]];
    this.dirty = true;
  }
  /** Ray through a screen point (client coordinates) from the eye of the last frame, box units. */
  pickRay(clientX: number, clientY: number): { o: V3; d: V3 } | null {
    const inv = this.lastInv; if (!inv) return null;
    const r = this.canvas.getBoundingClientRect(), nx = (clientX - r.left) / r.width * 2 - 1, ny = 1 - (clientY - r.top) / r.height * 2;
    const x = inv[0]! * nx + inv[4]! * ny + inv[8]! + inv[12]!, y = inv[1]! * nx + inv[5]! * ny + inv[9]! + inv[13]!;
    const z = inv[2]! * nx + inv[6]! * ny + inv[10]! + inv[14]!, w = inv[3]! * nx + inv[7]! * ny + inv[11]! + inv[15]!;
    const e = this.lastEye, d: V3 = [x / w - e[0], y / w - e[1], z / w - e[2]], l = Math.hypot(d[0], d[1], d[2]);
    return { o: [e[0], e[1], e[2]], d: [d[0] / l, d[1] / l, d[2] / l] };
  }
  /** Where the ray through a screen point meets the horizontal plane at height zb (box units), or null. */
  pickPlane(clientX: number, clientY: number, zb = 0): V3 | null {
    const r = this.pickRay(clientX, clientY); if (!r || Math.abs(r.d[2]) < 1e-6) return null;
    const t = (zb - r.o[2]) / r.d[2]; if (t <= 0) return null;
    return [r.o[0] + t * r.d[0], r.o[1] + t * r.d[1], zb];
  }
  /** Screen position (CSS pixels in the canvas) of a point in box units as last drawn, or null behind the eye / off screen. */
  project(p: V3): { x: number; y: number } | null {
    const v = this.lastVP; if (!v) return null;
    const cx = v[0]! * p[0] + v[4]! * p[1] + v[8]! * p[2] + v[12]!, cy = v[1]! * p[0] + v[5]! * p[1] + v[9]! * p[2] + v[13]!;
    const cw = v[3]! * p[0] + v[7]! * p[1] + v[11]! * p[2] + v[15]!;
    if (cw <= 1e-6) return null;
    const x = cx / cw, y = cy / cw;
    if (Math.abs(x) > 1.05 || Math.abs(y) > 1.05) return null;
    return { x: (x + 1) / 2 * this.canvas.clientWidth, y: (1 - y) / 2 * this.canvas.clientHeight };
  }
  /** fraction of the box height per model metre (to convert heights) and the box size */
  get boxSize(): V3 { return [this.box[0], this.box[1], this.box[2]]; }
  /** Fly the orbit camera to the ground point under the cursor (double-click / double-tap). */
  private focusAt(clientX: number, clientY: number): void {
    const p = this.pickPlane(clientX, clientY, 0);
    if (!p || this.fly) return;
    const [bx, by, bz] = this.box;
    this.tgt = [Math.max(0, Math.min(bx, p[0])), Math.max(0, Math.min(by, p[1])), bz * 0.3];
    this.dist = Math.max(0.08, this.dist * 0.6);
    this.dirty = true;
  }

  /** free-flight camera: eye position (box units) and look direction; null = orbit camera */
  private fly: { eye: [number, number, number]; yaw: number; pitch: number } | null = null;
  private readonly keys = new Set<string>();
  private lastT = 0;
  setCamera(mode: CameraMode): void {
    // fixed views of the whole domain: level from the south side, or straight down from above (dragging turns them back into the orbit view)
    if (mode === 'side') { this.fly = null; this.tgt = null; this.yaw = -Math.PI / 2; this.pitch = 0; this.dist = 1.0; this.dirty = true; return; }
    if (mode === 'top') { this.fly = null; this.tgt = null; this.yaw = -Math.PI / 2; this.pitch = Math.PI / 2; this.dist = 0.85; this.dirty = true; return; }
    if (mode === 'orbit') { this.fly = null; this.tgt = null; this.yaw = -0.9; this.pitch = 0.35; this.dist = 1.35; this.dirty = true; return; }
    if (this.fly) return;
    // start where the orbit camera is, looking at the same point
    const c = this.target();
    const eye: [number, number, number] = [c[0]! + this.dist * Math.cos(this.pitch) * Math.cos(this.yaw), c[1]! + this.dist * Math.cos(this.pitch) * Math.sin(this.yaw), c[2]! + this.dist * Math.sin(this.pitch)];
    const d = [c[0]! - eye[0], c[1]! - eye[1], c[2]! - eye[2]], l = Math.hypot(d[0]!, d[1]!, d[2]!);
    this.fly = { eye, yaw: Math.atan2(d[1]!, d[0]!), pitch: Math.asin(d[2]! / l) };
    this.dirty = true;
  }
  /** move the flying eye: forward along the look direction, right, up (box units) */
  private moveFly(fwd: number, right: number, up: number): void {
    const f = this.fly; if (!f) return;
    const cp = Math.cos(f.pitch), dir = [cp * Math.cos(f.yaw), cp * Math.sin(f.yaw), Math.sin(f.pitch)], rgt = [Math.sin(f.yaw), -Math.cos(f.yaw), 0];
    for (let i = 0; i < 3; i++) f.eye[i] = f.eye[i]! + fwd * dir[i]! + right * rgt[i]!;
    f.eye[2] = Math.max(0.002, f.eye[2] + up);
    this.dirty = true;
  }

  /** Upload cloud/rain volume [k][j][i] (bytes) and set the box aspect (x, y normalised to 1). */
  setVolume(nx: number, ny: number, nz: number, cloud: Uint8Array, rain: Uint8Array, aspectZ: number): void {
    const gl = this.gl, rg = new Uint8Array(nx * ny * nz * 2);
    for (let i = 0; i < cloud.length; i++) { rg[2 * i] = cloud[i]!; rg[2 * i + 1] = rain[i]!; }
    gl.bindTexture(gl.TEXTURE_3D, this.vol);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texImage3D(gl.TEXTURE_3D, 0, gl.RG8, nx, ny, nz, 0, gl.RG, gl.UNSIGNED_BYTE, rg);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    for (const w of [gl.TEXTURE_WRAP_S, gl.TEXTURE_WRAP_T, gl.TEXTURE_WRAP_R]) gl.texParameteri(gl.TEXTURE_3D, w, gl.CLAMP_TO_EDGE);
    const ax = 1, ay = ny / nx;
    // a new domain shape: back to the domain centre
    if (ax !== this.box[0] || ay !== this.box[1] || Math.abs(aspectZ - this.box[2]) > 1e-9) this.tgt = null;
    this.box = [ax, ay, aspectZ];
    this.dirty = true;
  }

  /** Tracer particles (x, y, z in m and age per particle; null: none) in a domain of Lx x Ly x top. */
  setTracers(pos: Float32Array | null, Lx: number, Ly: number, top: number): void {
    const gl = this.gl;
    if (!pos || !this.trProg) { if (this.trLines || this.trPoints) { this.trLines = 0; this.trPoints = 0; this.dirty = true; } this.trHist = null; return; }
    const n = pos.length / 4, [bx, by, bz] = this.box;
    if (!this.trHist || this.trHist.length !== n * TRAIL * 3) { this.trHist = new Float32Array(n * TRAIL * 3); this.trCnt = new Uint8Array(n); this.trAge = new Float32Array(n); }
    const Hh = this.trHist, cnt = this.trCnt!, age = this.trAge!;
    for (let p = 0; p < n; p++) {
      const x = pos[4 * p]! / Lx * bx, y = pos[4 * p + 1]! / Ly * by, z = pos[4 * p + 2]! / top * bz, a = pos[4 * p + 3]!, o = p * TRAIL * 3;
      // new particle (re-seeded) or wrapped around a periodic boundary: restart its trail
      if (a < age[p]! || (cnt[p]! > 0 && Math.hypot(x - Hh[o]!, y - Hh[o + 1]!) > 0.2)) cnt[p] = 0;
      age[p] = a;
      Hh.copyWithin(o + 3, o, o + 3 * (TRAIL - 1));
      Hh[o] = x; Hh[o + 1] = y; Hh[o + 2] = z;
      cnt[p] = Math.min(TRAIL, cnt[p]! + 1);
    }
    let nl = 0; for (let p = 0; p < n; p++) nl += Math.max(0, cnt[p]! - 1);
    const v = new Float32Array(4 * (2 * nl + n));
    let o = 0;
    for (let p = 0; p < n; p++) for (let s = 0; s < cnt[p]! - 1; s++) {
      const q = p * TRAIL * 3 + 3 * s;
      v[o++] = Hh[q]!; v[o++] = Hh[q + 1]!; v[o++] = Hh[q + 2]!; v[o++] = 0.85 * (1 - s / (TRAIL - 1));
      v[o++] = Hh[q + 3]!; v[o++] = Hh[q + 4]!; v[o++] = Hh[q + 5]!; v[o++] = 0.85 * (1 - (s + 1) / (TRAIL - 1));
    }
    for (let p = 0; p < n; p++) { const q = p * TRAIL * 3; v[o++] = Hh[q]!; v[o++] = Hh[q + 1]!; v[o++] = Hh[q + 2]!; v[o++] = 1; }
    gl.bindBuffer(gl.ARRAY_BUFFER, this.trBuf); gl.bufferData(gl.ARRAY_BUFFER, v, gl.DYNAMIC_DRAW);
    this.trLines = 2 * nl; this.trPoints = n;
    this.dirty = true;
  }

  private tracerPass(vp: Float32Array, eye: [number, number, number], w: number, h: number): boolean {
    const gl = this.gl;
    if (!this.trProg || !this.trPoints) return false;
    let f = this.trFbo;
    if (!f || f.w !== w || f.h !== h) {
      if (f) { gl.deleteFramebuffer(f.fb); gl.deleteTexture(f.c); gl.deleteTexture(f.d); gl.deleteRenderbuffer(f.z); }
      const tex = (): WebGLTexture => {
        const t = gl.createTexture()!; gl.bindTexture(gl.TEXTURE_2D, t);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        return t;
      };
      const c = tex(), d = tex(), z = gl.createRenderbuffer()!;
      gl.bindRenderbuffer(gl.RENDERBUFFER, z); gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT24, w, h);
      const fb = gl.createFramebuffer()!;
      gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, c, 0);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT1, gl.TEXTURE_2D, d, 0);
      gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, z);
      if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) { gl.bindFramebuffer(gl.FRAMEBUFFER, null); this.trProg = null; return false; }
      f = this.trFbo = { fb, c, d, z, w, h };
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, f.fb);
    gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1]);
    gl.viewport(0, 0, w, h);
    gl.clearBufferfv(gl.COLOR, 0, [0, 0, 0, 0]); gl.clearBufferfv(gl.COLOR, 1, [1, 1, 0, 1]); gl.clearBufferfv(gl.DEPTH, 0, [1]);
    gl.enable(gl.DEPTH_TEST); gl.depthFunc(gl.LESS);
    gl.useProgram(this.trProg);
    const u = (n: string): WebGLUniformLocation | null => gl.getUniformLocation(this.trProg!, n);
    gl.uniformMatrix4fv(u('uVP'), false, vp); gl.uniform3f(u('uEye'), eye[0], eye[1], eye[2]); gl.uniform3f(u('uBox'), this.box[0], this.box[1], this.box[2]);
    const loc = gl.getAttribLocation(this.trProg, 'aP');
    gl.bindBuffer(gl.ARRAY_BUFFER, this.trBuf); gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc, 4, gl.FLOAT, false, 0, 0);
    gl.uniform1f(u('uPt'), 1);
    if (this.trLines) gl.drawArrays(gl.LINES, 0, this.trLines);
    gl.uniform1f(u('uPt'), Math.max(2, Math.min(4, 2.5 * w / 1200)));
    gl.drawArrays(gl.POINTS, this.trLines, this.trPoints);
    gl.disableVertexAttribArray(loc);
    gl.disable(gl.DEPTH_TEST);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    // restore the full-screen quad attribute of the volume program
    const ql = gl.getAttribLocation(this.prog, 'aPos');
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quad); gl.enableVertexAttribArray(ql); gl.vertexAttribPointer(ql, 2, gl.FLOAT, false, 0, 0);
    return true;
  }

  /** Upload the ground colour image (RGBA bytes, [j][i]). */
  setGround(nx: number, ny: number, rgba: Uint8Array): void {
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, this.groundTex);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, nx, ny, 0, gl.RGBA, gl.UNSIGNED_BYTE, rgba);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    this.dirty = true;
  }

  /** Set when the picture must be redrawn (new data, camera or size change); the ray march is costly
   *  and shares the GPU with the model, so unchanged frames are not redrawn. */
  private dirty = true;
  private lastK = -1;

  render(cloudK: number, rainK: number): void {
    const gl = this.gl, c = this.canvas;
    // free flight: move with the pressed keys (0.25 box widths per second, 4x with Shift)
    const now = performance.now(), dts = Math.min(0.1, (now - (this.lastT || now)) / 1000); this.lastT = now;
    if (this.keys.size) {
      const k = this.keys, fb = (k.has('w') || k.has('arrowup') ? 1 : 0) - (k.has('s') || k.has('arrowdown') ? 1 : 0);
      const lr = (k.has('d') || k.has('arrowright') ? 1 : 0) - (k.has('a') || k.has('arrowleft') ? 1 : 0), ud = (k.has('e') ? 1 : 0) - (k.has('q') ? 1 : 0);
      if (this.fly) {
        // speed grows with the height above the ground (slow near the surface, fast high up)
        const v = dts * (k.has('shift') ? 4 : 1) * Math.max(0.03, Math.min(0.4, this.fly.eye[2] * 2));
        this.moveFly(fb * v, lr * v, ud * v);
      } else if (fb || lr || ud) {
        const px = 600 * dts * (k.has('shift') ? 3 : 1);
        this.pan(-lr * px, fb * px);
        if (ud) this.zoomBy(Math.exp(-ud * 1.5 * dts));
      }
    }
    // cap the ray-marched pixel count (about 0.9 megapixels): the ray march is the costly part
    const cssW = Math.max(1, c.clientWidth), cssH = Math.max(1, c.clientHeight);
    const scale = Math.min(Math.min(1.5, window.devicePixelRatio || 1), Math.sqrt(9e5 / (cssW * cssH)));
    const w = Math.max(1, Math.round(cssW * scale)), h = Math.max(1, Math.round(cssH * scale));
    if (c.width !== w || c.height !== h) { c.width = w; c.height = h; this.dirty = true; }
    if (cloudK !== this.lastK) { this.lastK = cloudK; this.dirty = true; }
    if (!this.dirty) return;
    this.dirty = false;
    gl.viewport(0, 0, w, h);
    gl.useProgram(this.prog);
    const [bx, by, bz] = this.box;
    let ctr: [number, number, number] = this.target();
    let eye: [number, number, number] = [
      ctr[0] + this.dist * Math.cos(this.pitch) * Math.cos(this.yaw),
      ctr[1] + this.dist * Math.cos(this.pitch) * Math.sin(this.yaw),
      ctr[2] + this.dist * Math.sin(this.pitch),
    ];
    if (this.fly) {
      const f = this.fly, cp = Math.cos(f.pitch);
      eye = [f.eye[0], f.eye[1], f.eye[2]];
      ctr = [eye[0] + cp * Math.cos(f.yaw), eye[1] + cp * Math.sin(f.yaw), eye[2] + Math.sin(f.pitch)];
    }
    const vp = viewProj(eye, ctr, w / h);
    const inv = invert4(vp);
    this.lastInv = inv; this.lastEye = eye; this.lastVP = vp;
    const trOn = this.tracerPass(vp, eye, w, h);
    gl.viewport(0, 0, w, h);
    gl.useProgram(this.prog);
    const u = (n: string): WebGLUniformLocation => gl.getUniformLocation(this.prog, n)!;
    gl.uniformMatrix4fv(u('uInvVP'), false, inv);
    gl.uniform3f(u('uEye'), eye[0], eye[1], eye[2]);
    gl.uniform3f(u('uBox'), bx, by, bz);
    const s = [0.5, -0.4, 0.75], sl = Math.hypot(s[0]!, s[1]!, s[2]!);
    gl.uniform3f(u('uSun'), s[0]! / sl, s[1]! / sl, s[2]! / sl);
    gl.uniform1f(u('uCloudK'), cloudK);
    gl.uniform1f(u('uRainK'), rainK);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_3D, this.vol); gl.uniform1i(u('uVol'), 0);
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, this.groundTex); gl.uniform1i(u('uGround'), 1);
    gl.uniform1i(u('uTrOn'), trOn ? 1 : 0);
    if (trOn) {
      gl.activeTexture(gl.TEXTURE2); gl.bindTexture(gl.TEXTURE_2D, this.trFbo!.c); gl.uniform1i(u('uTrC'), 2);
      gl.activeTexture(gl.TEXTURE3); gl.bindTexture(gl.TEXTURE_2D, this.trFbo!.d); gl.uniform1i(u('uTrD'), 3);
    } else {
      // keep the samplers on 2-D texture units (an unbound sampler type mismatch is an error)
      gl.uniform1i(u('uTrC'), 1); gl.uniform1i(u('uTrD'), 1);
    }
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }
}

type V3 = [number, number, number];
function viewProj(eye: V3, ctr: V3, aspect: number): Float32Array {
  const f = norm([ctr[0] - eye[0], ctr[1] - eye[1], ctr[2] - eye[2]]);
  // looking straight down: north (+y) is up on the screen
  const s = norm(cross(f, Math.abs(f[2]) > 0.999 ? [0, 1, 0] : [0, 0, 1])), u = cross(s, f);
  const view = [s[0], u[0], -f[0], 0, s[1], u[1], -f[1], 0, s[2], u[2], -f[2], 0, -dot(s, eye), -dot(u, eye), dot(f, eye), 1];
  const fov = 0.9, n = 0.01, fa = 100, t = 1 / Math.tan(fov / 2);
  const proj = [t / aspect, 0, 0, 0, 0, t, 0, 0, 0, 0, (fa + n) / (n - fa), -1, 0, 0, 2 * fa * n / (n - fa), 0];
  const out = new Float32Array(16);
  for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) { let a = 0; for (let k = 0; k < 4; k++) a += proj[k * 4 + j]! * view[i * 4 + k]!; out[i * 4 + j] = a; }
  return out;
}
function invert4(m: Float32Array): Float32Array {
  const a = Array.from(m), inv = new Array<number>(16).fill(0);
  for (let i = 0; i < 4; i++) inv[i * 5] = 1;
  for (let c = 0; c < 4; c++) {
    let p = c; for (let r = c + 1; r < 4; r++) if (Math.abs(a[r * 4 + c]!) > Math.abs(a[p * 4 + c]!)) p = r;
    for (let j = 0; j < 4; j++) { [a[c * 4 + j], a[p * 4 + j]] = [a[p * 4 + j]!, a[c * 4 + j]!]; [inv[c * 4 + j], inv[p * 4 + j]] = [inv[p * 4 + j]!, inv[c * 4 + j]!]; }
    const d = a[c * 4 + c]!;
    for (let j = 0; j < 4; j++) { a[c * 4 + j] = a[c * 4 + j]! / d; inv[c * 4 + j] = inv[c * 4 + j]! / d; }
    for (let r = 0; r < 4; r++) if (r !== c) { const f = a[r * 4 + c]!; for (let j = 0; j < 4; j++) { a[r * 4 + j] = a[r * 4 + j]! - f * a[c * 4 + j]!; inv[r * 4 + j] = inv[r * 4 + j]! - f * inv[c * 4 + j]!; } }
  }
  // matrices are column-major (m[col*4+row]); Gauss–Jordan above treated rows as m[i*4+j] consistently
  return Float32Array.from(inv);
}
function cross(a: V3, b: V3): V3 { return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]; }
function dot(a: V3, b: V3): number { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
function norm(a: V3): V3 { const l = Math.hypot(a[0], a[1], a[2]); return [a[0] / l, a[1] / l, a[2] / l]; }
