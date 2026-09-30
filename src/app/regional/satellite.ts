// Satellite pictures of the regional model at the resolution of the screen: the display volume of the 3-D view (the
// model's own cloud and precipitation extinctions, display.ts; nothing added) seen straight down, in the domain's true
// shape (no vertical exaggeration), with the eye nest's finer volume inside its cylinder. A picture pixel is a ray down
// its own column, so the picture has the screen's resolution; the field itself is the model's (interpolated between
// grid points): detail the grid does not resolve is not drawn.
//
// Visible (true colour): extinction scaled for the strongly forward-scattering cloud particles (similarity scaling,
// asymmetry 0.85: beta* = 0.15 beta), both along the view and toward the sun, so a slab of optical depth tau reflects
// about tau / (tau + 7.7) of the light (the two-stream albedo of the column composites) and thin cirrus shows the sea
// through it; sunlit and shaded cloud (the light volume of the true shape: shadows as long as the real ones), cloud
// shadows on the sea and land, and the blue path radiance of the air above whatever is seen (scale height 8 km).
// Infrared (10.8 um window): emission of the cloud (absorption optical depth half the visible one) at the temperature
// of its height (the base-state profile), above the emission of the surface (the lowest level's air temperature);
// brightness temperature from the emitted power (T^4).
import { EXT_MAX } from '../../regional/display.js';
import type { NestFrame } from './protocol.js';
import { nestSunSkyLight, occupancyGrid, sunSkyLight, type V3 } from './light.js';

/** the display volume of a frame (as the 3-D view gets it) */
export interface SatScene {
  nx: number; ny: number; nz: number;
  /** domain width, height and top (m) */
  Lx: number; Ly: number; top: number;
  cloud: Uint8Array; rain: Uint8Array;
  nest?: NestFrame | null;
}
export interface SatRequest {
  kind: 'vis' | 'ir';
  /** the part of the domain shown (fractions: x from u0 to u1, y from v0 to v1) and the picture size (pixels) */
  u0: number; u1: number; v0: number; v1: number; w: number; h: number;
  /** the second channel holds precipitation (not updraft or vorticity) */
  precip: boolean;
  /** land mask per column (1 land), or null: all sea (sea) or all land */
  land: Uint8Array | null; sea: boolean;
  /** infrared: temperature at the outer grid's level centres (K) and at the surface per column (deg C, [j][i]) */
  tz?: Float32Array | null; sfcC?: Float32Array | null;
}
/** picture rows from the top: visible colours, or infrared brightness temperatures (K) */
export interface SatImage { w: number; h: number; rgba?: Uint8ClampedArray; bt?: Float32Array }

/** direction toward the sun: from the north-west, 40 degrees up (shadows fall to the south-east) */
const SUN: V3 = ((): V3 => { const az = 315 * Math.PI / 180, el = 40 * Math.PI / 180; return [Math.cos(el) * Math.sin(az), Math.cos(el) * Math.cos(az), Math.sin(el)]; })();
/** similarity-scaled extinction of cloud particles (1 - asymmetry) */
const SCALED = 0.15;
/** occupancy blocks: cells across, levels up */
const B = 8, BZ = 4;
/** largest picture side (pixels) */
const MAX_SIDE = 2048;

const VS = `#version 300 es
in vec2 aPos; out vec2 vUv;
void main(){ vUv = aPos*0.5+0.5; gl_Position = vec4(aPos,0.0,1.0); }`;

const FS = `#version 300 es
precision highp float; precision highp sampler3D;
in vec2 vUv; out vec4 o;
uniform sampler3D uVol; uniform sampler3D uLight; uniform sampler3D uNVol; uniform sampler3D uNLight; uniform sampler3D uOcc;
uniform sampler2D uLand; uniform sampler2D uTs; uniform sampler2D uTz;
uniform vec4 uWin;      // u0, v0, u1, v1
uniform vec3 uL;        // domain width, height, top (m)
uniform int uIR;        // 0 visible, 1 infrared
uniform float uRain;    // 1: the second channel is precipitation
uniform float uNz;      // outer levels
uniform float uNzN;     // the nest's levels
uniform int uNOn; uniform vec3 uNBox; uniform vec4 uNCyl;   // nest box (x0, y0, width) and cylinder (centre, radius, taper), m
uniform vec3 uOccScale; uniform float uBz;
uniform vec3 uLOff; uniform vec3 uNLOff;   // light lookup offset toward the sun (texture coordinates, outer and nest)
uniform vec3 uN; uniform vec3 uNN;         // texels of the outer and nest volumes
const float EXTMAX = ${EXT_MAX.toFixed(3)};
float nestW(vec2 xy){
  if (uNOn == 0) return 0.0;
  float d = length(xy - uNCyl.xy), a = uNCyl.z - uNCyl.w;
  if (d >= uNCyl.z) return 0.0;
  if (d <= a) return 1.0;
  float c = cos(1.5707963*(d - a)/uNCyl.w); return c*c;
}
vec3 nestQ(vec2 xy, float zf){ return vec3((xy - uNBox.xy)/uNBox.z, zf); }
// Cubic B-spline reconstruction between the grid values (8 trilinear lookups, Sigg & Hadwiger 2005): smooth across
// the cell faces, where trilinear interpolation has kinks that a picture at screen resolution shows as a lattice
vec2 bspline(sampler3D t, vec3 q, vec3 n){
  vec3 c = q*n - 0.5, i = floor(c), f = c - i, f2 = f*f, f3 = f2*f;
  vec3 w0 = (1.0 - 3.0*f + 3.0*f2 - f3)/6.0, w1 = (4.0 - 6.0*f2 + 3.0*f3)/6.0, w2 = (1.0 + 3.0*f + 3.0*f2 - 3.0*f3)/6.0, w3 = f3/6.0;
  vec3 g0 = w0 + w1, g1 = w2 + w3;
  vec3 h0 = (i - 0.5 + w1/g0)/n, h1 = (i + 1.5 + w3/g1)/n;
  vec2 a = mix(texture(t, vec3(h1.x, h1.y, h1.z)).rg, texture(t, vec3(h0.x, h1.y, h1.z)).rg, g0.x);
  vec2 b = mix(texture(t, vec3(h1.x, h0.y, h1.z)).rg, texture(t, vec3(h0.x, h0.y, h1.z)).rg, g0.x);
  vec2 d = mix(texture(t, vec3(h1.x, h1.y, h0.z)).rg, texture(t, vec3(h0.x, h1.y, h0.z)).rg, g0.x);
  vec2 e = mix(texture(t, vec3(h1.x, h0.y, h0.z)).rg, texture(t, vec3(h0.x, h0.y, h0.z)).rg, g0.x);
  return mix(mix(a, b, g0.y), mix(d, e, g0.y), g0.z);
}
vec2 bytesAt(vec3 q, vec2 xy, float w){ vec2 s = bspline(uVol, q, uN); if (w > 0.0) s = mix(s, bspline(uNVol, nestQ(xy, q.z), uNN), w); return s; }
// extinction per metre over EXTMAX: the cube of the bytes (precipitation only when the second channel holds it)
float cube(vec2 b){ return b.x*b.x*b.x + uRain*b.y*b.y*b.y; }
vec2 lightAt(vec3 q, vec2 xy, float w){
  vec2 L = bspline(uLight, q + uLOff, uN);
  if (w > 0.0) L = mix(L, bspline(uNLight, nestQ(xy, q.z) + uNLOff, uNN), w);
  return clamp(L, 0.0, 1.0);
}
// light scattered up by cloud at a point: sunlight through the (scaled) path toward the sun, softened as light diffusing
// inside cloud is, and the bluish light of the sky and the surrounding cloud
vec3 cloudLight(vec2 L){ return vec3(1.0, 0.985, 0.955)*0.80*sqrt(L.r) + vec3(0.60, 0.68, 0.84)*(0.11 + 0.11*L.g); }
float tempAt(float zf){ return 200.0 + texture(uTz, vec2(zf, 0.5)).r; }
float t4(float T){ float x = T*0.01; x *= x; return x*x; }
// share of the air's path radiance above height fraction zf (density scale height 8 km)
float hazeAbove(float zf){ float h = uL.z/8000.0; return (exp(-zf*h) - exp(-h))/(1.0 - exp(-h)); }
void main(){
  vec2 uv = mix(uWin.xy, uWin.zw, vUv), xy = uv*uL.xy;
  float w = nestW(xy);
  bool ir = uIR == 1;
  // IR absorption half the visible extinction; visible: similarity-scaled extinction
  float kx = (ir ? 0.5 : ${SCALED.toFixed(3)})*EXTMAX*uL.z;
  // steps of a quarter of a level (of the nest's levels inside its cylinder): the samples keep to the same heights in
  // every column, and Simpson's rule on each step is exact for the cubic between two level centres
  float nzs = w > 0.0 ? max(uNz, uNzN) : uNz, ds = 0.25/nzs, jump = uBz/uNz - ds;
  const vec3 HAZE = vec3(0.004, 0.010, 0.030);
  vec3 col = vec3(0.0); float trans = 1.0, em = 0.0;
  float s = 1.0;
  vec2 bA = bytesAt(vec3(uv, s), xy, w); float eA = cube(bA);
  vec3 cA = vec3(0.0); float TA = 0.0;
  if (eA > 0.0) { if (ir) TA = t4(tempAt(s)); else cA = cloudLight(lightAt(vec3(uv, s), xy, w)); }
  for (int i = 0; i < 1400; i++) {
    if (s <= 1e-6 || trans < 0.002) break;
    // clear air around: down by most of a block at once
    if (eA == 0.0 && texture(uOcc, vec3(uv, s)*uOccScale).r == 0.0) {
      float sn = max(0.0, s - jump);
      if (!ir) col += trans*HAZE*(hazeAbove(sn) - hazeAbove(s));
      s = sn; bA = bytesAt(vec3(uv, s), xy, w); eA = cube(bA);
      if (eA > 0.0) { if (ir) TA = t4(tempAt(s)); else cA = cloudLight(lightAt(vec3(uv, s), xy, w)); }
      continue;
    }
    float sb = max(0.0, s - ds), sm = 0.5*(s + sb);
    vec2 bM = bytesAt(vec3(uv, sm), xy, w), bB = bytesAt(vec3(uv, sb), xy, w);
    float eM = cube(bM), eB = cube(bB);
    float tau = kx*(eA + 4.0*eM + eB)/6.0*(s - sb);
    vec3 cB = vec3(0.0); float TB = 0.0;
    if (eB > 0.0) { if (ir) TB = t4(tempAt(sb)); else cB = cloudLight(lightAt(vec3(uv, sb), xy, w)); }
    if (tau > 1e-7) {
      float a = 1.0 - exp(-tau), wA = eA, wM = 4.0*eM, wB = eB, sw = wA + wM + wB;
      if (ir) { float TM = eM > 0.0 ? t4(tempAt(sm)) : 0.0; em += trans*a*(wA*TA + wM*TM + wB*TB)/sw; }
      else { vec3 cM = eM > 0.0 ? cloudLight(lightAt(vec3(uv, sm), xy, w)) : vec3(0.0); col += trans*a*(wA*cA + wM*cM + wB*cB)/sw; }
      trans *= 1.0 - a;
    }
    if (!ir) col += trans*HAZE*(hazeAbove(sb) - hazeAbove(s));
    s = sb; bA = bB; eA = eB; cA = cB; TA = TB;
  }
  if (ir) {
    em += trans*t4(200.0 + texture(uTs, uv).r);
    float bt = 100.0*sqrt(sqrt(max(em, 1e-6)));
    float v = floor(clamp((bt - 150.0)/250.0, 0.0, 1.0)*65535.0 + 0.5), hi = floor(v/256.0);
    o = vec4(hi/255.0, (v - hi*256.0)/255.0, 0.0, 1.0);
    return;
  }
  // the surface: sea or land, lit by the sun through the clouds (their shadows) and by the sky
  vec2 Lg = lightAt(vec3(uv, 0.0), xy, w);
  float land = texture(uLand, uv).r;
  vec3 alb = mix(vec3(0.004, 0.014, 0.040), vec3(0.042, 0.072, 0.016), land);
  col += trans*alb*(0.80*Lg.r + 0.20*(0.3 + 0.7*Lg.g));
  o = vec4(pow(max(col, vec3(0.0)), vec3(1.0/2.2)), 1.0);
}`;

export class SatelliteRenderer {
  private readonly gl: WebGL2RenderingContext;
  private readonly prog: WebGLProgram;
  private readonly tex: Record<'vol' | 'light' | 'nVol' | 'nLight' | 'occ' | 'land' | 'ts' | 'tz', WebGLTexture>;
  private fbo: { fb: WebGLFramebuffer; t: WebGLTexture; w: number; h: number } | null = null;
  /** the uploaded scene, whether its light is computed, its occupancy texture scale */
  private scene: SatScene | null = null; private lit = false; private occScale: V3 = [1, 1, 1];
  private landOf: { land: Uint8Array | null; sea: boolean; nx: number; ny: number } | null = null;
  private tsOf: Float32Array | null = null; private tzOf: Float32Array | null = null;
  private last: { scene: SatScene; req: SatRequest; img: SatImage } | null = null;

  /** A renderer, or null where WebGL2 is not available. */
  static create(): SatelliteRenderer | null { try { return new SatelliteRenderer(); } catch { return null; } }

  private constructor() {
    const c = document.createElement('canvas'); c.width = 1; c.height = 1;
    const gl = c.getContext('webgl2', { antialias: false, depth: false, alpha: false });
    if (!gl) throw new Error('WebGL2 unavailable');
    this.gl = gl;
    const sh = (t: number, s: string): WebGLShader => { const x = gl.createShader(t)!; gl.shaderSource(x, s); gl.compileShader(x); if (!gl.getShaderParameter(x, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(x) ?? 'shader'); return x; };
    const p = gl.createProgram()!;
    gl.attachShader(p, sh(gl.VERTEX_SHADER, VS)); gl.attachShader(p, sh(gl.FRAGMENT_SHADER, FS)); gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p) ?? 'link');
    this.prog = p;
    const vb = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, vb);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
    const loc = gl.getAttribLocation(p, 'aPos');
    gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
    const t = (): WebGLTexture => gl.createTexture()!;
    this.tex = { vol: t(), light: t(), nVol: t(), nLight: t(), occ: t(), land: t(), ts: t(), tz: t() };
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    // placeholders until a scene arrives (the sampler types must be complete)
    for (const k of ['vol', 'light', 'nVol', 'nLight'] as const) { const v = k === 'light' || k === 'nLight' ? 255 : 0; this.tex3(this.tex[k], 1, 1, 1, gl.RG8, gl.RG, new Uint8Array([v, v]), true); }
    this.tex3(this.tex.occ, 1, 1, 1, gl.R8, gl.RED, new Uint8Array([255]), false);
    this.tex2(this.tex.land, 1, 1, gl.R8, gl.RED, gl.UNSIGNED_BYTE, new Uint8Array([0]));
    this.tex2(this.tex.ts, 1, 1, gl.R16F, gl.RED, gl.FLOAT, new Float32Array([90]));
    this.tex2(this.tex.tz, 1, 1, gl.R16F, gl.RED, gl.FLOAT, new Float32Array([40]));
  }

  private tex3(t: WebGLTexture, nx: number, ny: number, nz: number, fmt: number, src: number, data: Uint8Array, linear: boolean): void {
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_3D, t);
    gl.texImage3D(gl.TEXTURE_3D, 0, fmt, nx, ny, nz, 0, src, gl.UNSIGNED_BYTE, data);
    const f = linear ? gl.LINEAR : gl.NEAREST;
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MIN_FILTER, f); gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MAG_FILTER, f);
    for (const w of [gl.TEXTURE_WRAP_S, gl.TEXTURE_WRAP_T, gl.TEXTURE_WRAP_R]) gl.texParameteri(gl.TEXTURE_3D, w, gl.CLAMP_TO_EDGE);
  }
  private tex2(t: WebGLTexture, nx: number, ny: number, fmt: number, src: number, type: number, data: ArrayBufferView): void {
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.texImage2D(gl.TEXTURE_2D, 0, fmt, nx, ny, 0, src, type, data);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  }
  private rg(a: Uint8Array, b: Uint8Array): Uint8Array { const o = new Uint8Array(2 * a.length); for (let i = 0; i < a.length; i++) { o[2 * i] = a[i]!; o[2 * i + 1] = b[i]!; } return o; }

  /** Upload a new scene's volumes and occupancy (its light when a visible picture first needs it). */
  private upload(sc: SatScene): void {
    const gl = this.gl, { nx, ny, nz } = sc, n = sc.nest;
    this.tex3(this.tex.vol, nx, ny, nz, gl.RG8, gl.RG, this.rg(sc.cloud, sc.rain), true);
    if (n) this.tex3(this.tex.nVol, n.nx, n.nx, n.nz, gl.RG8, gl.RG, this.rg(n.cloud, n.rain), true);
    const o = occupancyGrid(nx, ny, nz, sc.cloud, sc.rain, B, BZ, n ? { ...n, Lx: sc.Lx, Ly: sc.Ly } : null);
    this.tex3(this.tex.occ, o.mx, o.my, o.mz, gl.R8, gl.RED, o.occ, false);
    this.occScale = [nx / (o.mx * B), ny / (o.my * B), nz / (o.mz * BZ)];
    this.scene = sc; this.lit = false;
  }
  /** Light of the scene in its true shape (box units: the width is 1), cloud extinction similarity-scaled. */
  private light(sc: SatScene): void {
    const gl = this.gl, u = 1 / sc.Lx, box: V3 = [1, sc.Ly * u, sc.top * u], n = sc.nest;
    const { out, od } = sunSkyLight(sc.nx, sc.ny, sc.nz, sc.cloud, box, sc.top, SUN, SCALED, null, !!n);
    this.tex3(this.tex.light, sc.nx, sc.ny, sc.nz, gl.RG8, gl.RG, out, true);
    if (n && od) {
      const nl = nestSunSkyLight({ nx: n.nx, nz: n.nz, cloud: n.cloud, x0: n.x0 * u, y0: n.y0 * u, L: n.L * u }, { nx: sc.nx, ny: sc.ny, nz: sc.nz, top: sc.top, od }, box, SUN, SCALED, null);
      this.tex3(this.tex.nLight, n.nx, n.nx, n.nz, gl.RG8, gl.RG, nl, true);
    }
    this.lit = true;
  }

  /** The picture of a scene (the last one again while nothing changed), or null when it cannot be drawn. */
  render(sc: SatScene, req: SatRequest): SatImage | null {
    const l = this.last;
    if (l && l.scene === sc && sameReq(l.req, req)) return l.img;
    const gl = this.gl;
    if (gl.isContextLost()) return null;
    let w = Math.max(1, Math.round(req.w)), h = Math.max(1, Math.round(req.h));
    const f = Math.min(1, MAX_SIDE / Math.max(w, h)); w = Math.max(1, Math.floor(w * f)); h = Math.max(1, Math.floor(h * f));
    if (this.scene !== sc) this.upload(sc);
    const ir = req.kind === 'ir';
    if (!ir && !this.lit) this.light(sc);
    const { nx, ny, nz } = sc;
    if (!ir && (!this.landOf || this.landOf.land !== req.land || this.landOf.sea !== req.sea || this.landOf.nx !== nx || this.landOf.ny !== ny)) {
      const m = new Uint8Array(nx * ny);
      for (let c = 0; c < nx * ny; c++) m[c] = (req.land ? req.land[c] : !req.sea) ? 255 : 0;
      this.tex2(this.tex.land, nx, ny, gl.R8, gl.RED, gl.UNSIGNED_BYTE, m);
      this.landOf = { land: req.land, sea: req.sea, nx, ny };
    }
    if (ir) {
      if (!req.tz || !req.sfcC || req.sfcC.length !== nx * ny) return null;
      if (this.tzOf !== req.tz) { this.tex2(this.tex.tz, req.tz.length, 1, gl.R16F, gl.RED, gl.FLOAT, req.tz.map((T) => T - 200)); this.tzOf = req.tz; }
      if (this.tsOf !== req.sfcC) { this.tex2(this.tex.ts, nx, ny, gl.R16F, gl.RED, gl.FLOAT, req.sfcC.map((c) => c + 73.15)); this.tsOf = req.sfcC; }
    }
    let fb = this.fbo;
    if (!fb || fb.w !== w || fb.h !== h) {
      if (fb) { gl.deleteFramebuffer(fb.fb); gl.deleteTexture(fb.t); }
      const t = gl.createTexture()!;
      gl.bindTexture(gl.TEXTURE_2D, t);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      const b = gl.createFramebuffer()!;
      gl.bindFramebuffer(gl.FRAMEBUFFER, b);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, t, 0);
      if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) { gl.bindFramebuffer(gl.FRAMEBUFFER, null); return null; }
      fb = this.fbo = { fb: b, t, w, h };
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, fb.fb);
    gl.viewport(0, 0, w, h);
    gl.useProgram(this.prog);
    const u = (s: string): WebGLUniformLocation | null => gl.getUniformLocation(this.prog, s);
    const units: [string, number, WebGLTexture][] = [['uVol', gl.TEXTURE_3D, this.tex.vol], ['uLight', gl.TEXTURE_3D, this.tex.light], ['uNVol', gl.TEXTURE_3D, this.tex.nVol],
      ['uNLight', gl.TEXTURE_3D, this.tex.nLight], ['uOcc', gl.TEXTURE_3D, this.tex.occ], ['uLand', gl.TEXTURE_2D, this.tex.land], ['uTs', gl.TEXTURE_2D, this.tex.ts], ['uTz', gl.TEXTURE_2D, this.tex.tz]];
    units.forEach(([name, target, t], i) => { gl.activeTexture(gl.TEXTURE0 + i); gl.bindTexture(target, t); gl.uniform1i(u(name), i); });
    gl.uniform4f(u('uWin'), req.u0, req.v0, req.u1, req.v1);
    gl.uniform3f(u('uL'), sc.Lx, sc.Ly, sc.top);
    gl.uniform1i(u('uIR'), ir ? 1 : 0);
    gl.uniform1f(u('uRain'), req.precip ? 1 : 0);
    gl.uniform1f(u('uNz'), nz);
    const n = sc.nest;
    gl.uniform1i(u('uNOn'), n ? 1 : 0);
    gl.uniform1f(u('uNzN'), n ? n.nz : nz);
    gl.uniform3f(u('uNBox'), n ? n.x0 : 0, n ? n.y0 : 0, n ? n.L : 1);
    gl.uniform4f(u('uNCyl'), n ? n.cx : 0, n ? n.cy : 0, n ? n.R : 0, n ? Math.max(n.Wf, 1e-3) : 1);
    gl.uniform3f(u('uOccScale'), this.occScale[0], this.occScale[1], this.occScale[2]);
    gl.uniform3f(u('uN'), nx, sc.ny, nz);
    gl.uniform3f(u('uNN'), n ? n.nx : 1, n ? n.nx : 1, n ? n.nz : 1);
    gl.uniform1f(u('uBz'), BZ);
    // light looked up half a level up along the sun (the light that reaches a cloud's top surface, not its inside)
    const c = 0.5 * sc.top / nz / SUN[2], cn = n ? 0.5 * sc.top / n.nz / SUN[2] : c;
    gl.uniform3f(u('uLOff'), SUN[0] * c / sc.Lx, SUN[1] * c / sc.Ly, SUN[2] * c / sc.top);
    gl.uniform3f(u('uNLOff'), n ? SUN[0] * cn / n.L : 0, n ? SUN[1] * cn / n.L : 0, SUN[2] * cn / sc.top);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    const px = new Uint8Array(4 * w * h);
    gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, px);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    const img: SatImage = { w, h };
    if (ir) {
      const bt = new Float32Array(w * h);
      for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) { const o = 4 * ((h - 1 - y) * w + x); bt[y * w + x] = 150 + 250 * (px[o]! * 256 + px[o + 1]!) / 65535; }
      img.bt = bt;
    } else {
      const rgba = new Uint8ClampedArray(4 * w * h), row = 4 * w;
      for (let y = 0; y < h; y++) rgba.set(px.subarray((h - 1 - y) * row, (h - y) * row), y * row);
      img.rgba = rgba;
    }
    this.last = { scene: sc, req: { ...req }, img };
    return img;
  }
}

function sameReq(a: SatRequest, b: SatRequest): boolean {
  return a.kind === b.kind && a.u0 === b.u0 && a.u1 === b.u1 && a.v0 === b.v0 && a.v1 === b.v1 && Math.round(a.w) === Math.round(b.w) && Math.round(a.h) === Math.round(b.h)
    && a.precip === b.precip && a.land === b.land && a.sea === b.sea && (a.kind === 'vis' || (a.tz === b.tz && a.sfcC === b.sfcC));
}
