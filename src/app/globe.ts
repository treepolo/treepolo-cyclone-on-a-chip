// WebGL2 globe: a high-resolution relief map of the planet (ERA5 land mask and orography at about
// 0.35 degrees) with the model field overlaid, a ray-marched cloud shell with exaggerated vertical scale
// (the model's diagnosed 3-D cloud), wind tracer streaks, a graticule and an orbit camera.
import { attachOrbit } from './orbitControls.js';

export type Rgb = [number, number, number];
export type Rgba = [number, number, number, number];

// lines (graticule, marker, tracers)
const VS = `#version 300 es
in vec3 aPos; in vec3 aCol;
uniform mat4 uMVP; uniform float uScale;
out vec3 vCol; out vec3 vN;
void main(){ vCol=aCol; vN=aPos; gl_Position=uMVP*vec4(aPos*uScale,1.0); gl_PointSize=2.0; }`;
const FS = `#version 300 es
precision highp float;
in vec3 vCol; in vec3 vN; uniform vec3 uLight; uniform float uAlpha; uniform float uShade;
out vec4 o;
void main(){ float l = mix(1.0, 0.55+0.45*max(dot(normalize(vN),uLight),0.0), uShade); o=vec4(vCol*l,uAlpha); }`;

const PI_GLSL = 'const float PI = 3.14159265358979;';
// cubic B-spline filtering from four bilinear taps (GPU Gems 2, ch. 20): smooth contours of the coarse
// model grid instead of the staircase of plain bilinear interpolation
const BSPLINE_GLSL = `
void bsTaps(vec2 uv, vec2 size, out vec4 p, out vec2 g0) {
  vec2 st = uv * size - 0.5, i = floor(st), f = st - i, f2 = f * f, f3 = f2 * f;
  vec2 w0 = (1.0 - 3.0 * f + 3.0 * f2 - f3) / 6.0, w1 = (4.0 - 6.0 * f2 + 3.0 * f3) / 6.0;
  vec2 w2 = (1.0 + 3.0 * f + 3.0 * f2 - 3.0 * f3) / 6.0, w3 = f3 / 6.0;
  g0 = w0 + w1;
  p = vec4((i - 0.5 + w1 / g0) / size, (i + 1.5 + w3 / (w2 + w3)) / size);
}
vec4 bspline2(sampler2D t, vec2 uv, vec2 size) {
  vec4 p; vec2 g0; bsTaps(uv, size, p, g0);
  return mix(mix(textureLod(t, p.zw, 0.0), textureLod(t, p.xw, 0.0), g0.x), mix(textureLod(t, p.zy, 0.0), textureLod(t, p.xy, 0.0), g0.x), g0.y);
}
vec4 bspline3(highp sampler3D t, vec3 uvw, vec2 size) {
  vec4 p; vec2 g0; bsTaps(uvw.xy, size, p, g0);
  return mix(mix(textureLod(t, vec3(p.zw, uvw.z), 0.0), textureLod(t, vec3(p.xw, uvw.z), 0.0), g0.x), mix(textureLod(t, vec3(p.zy, uvw.z), 0.0), textureLod(t, vec3(p.xy, uvw.z), 0.0), g0.x), g0.y);
}`;
// planet surface: relief-displaced sphere, map colours, hillshade, coastline, model field overlay
const PVS = `#version 300 es
precision highp float;
${PI_GLSL}
in vec2 aUV;
uniform mat4 uMVP; uniform sampler2D uMap; uniform float uRelief; uniform float uLandOn;
out vec2 vUV; out vec3 vP;
void main(){
  float lat = PI * 0.5 - aUV.y * PI, lon = aUV.x * 2.0 * PI;
  vec3 p = vec3(cos(lat) * cos(lon), sin(lat), -cos(lat) * sin(lon));
  float e = textureLod(uMap, vec2(aUV.x + 0.5 / 1024.0, aUV.y), 0.0).r * uLandOn;
  vUV = aUV; vP = p;
  gl_Position = uMVP * vec4(p * (1.0 + uRelief * e), 1.0);
}`;
const PFS = `#version 300 es
precision highp float;
in vec2 vUV; in vec3 vP;
uniform sampler2D uMap; uniform sampler2D uField; uniform vec3 uLight; uniform float uFieldAlpha; uniform float uLandOn;
uniform vec2 uFieldSize; uniform float uRelief;
${BSPLINE_GLSL}
out vec4 o;
vec3 landColour(float h){ // h: elevation / 6 km
  vec3 c = mix(vec3(0.20, 0.33, 0.16), vec3(0.45, 0.43, 0.26), smoothstep(0.0, 0.18, h));
  c = mix(c, vec3(0.42, 0.33, 0.24), smoothstep(0.18, 0.45, h));
  return mix(c, vec3(0.86, 0.86, 0.88), smoothstep(0.55, 0.85, h));
}
void main(){
  vec2 mu = vec2(vUV.x + 0.5 / 1024.0, vUV.y);
  vec4 m = textureLod(uMap, mu, 0.0);
  float land = m.g * uLandOn, h = m.r * uLandOn;
  // hillshade from the elevation gradient (map texels: 0.3516 deg)
  float dx = 1.0 / 1024.0, dy = 1.0 / 512.0;
  float hx = textureLod(uMap, mu + vec2(dx, 0.0), 0.0).r - textureLod(uMap, mu - vec2(dx, 0.0), 0.0).r;
  float hy = textureLod(uMap, mu - vec2(0.0, dy), 0.0).r - textureLod(uMap, mu + vec2(0.0, dy), 0.0).r;
  vec3 n = normalize(vP);
  vec3 east = normalize(vec3(-n.z, 0.0, -n.x) + 1e-6), north = cross(n, east);
  vec3 nn = normalize(n - (east * hx + north * hy) * uLandOn * 60.0);
  float sun = max(dot(nn, uLight), 0.0);
  vec3 ocean = vec3(0.035, 0.09, 0.20);
  vec3 base = mix(ocean, landColour(h), smoothstep(0.35, 0.65, land));
  // coastline
  float fw = fwidth(land);
  float coast = 1.0 - smoothstep(0.0, 1.2 * fw + 1e-4, abs(land - 0.5));
  base = mix(base, vec3(0.02, 0.02, 0.02), coast * 0.8 * uLandOn);
  vec3 col = base * (0.30 + 0.75 * sun);
  vec4 f = bspline2(uField, vec2(vUV.x + 0.5 / uFieldSize.x, vUV.y), uFieldSize);
  col = mix(col, f.rgb * (0.55 + 0.5 * max(dot(n, uLight), 0.0)), f.a * uFieldAlpha);
  o = vec4(col, 1.0);
}`;
// cloud shell: full-screen ray march between the surface and the top of the exaggerated troposphere
const VVS = `#version 300 es
in vec2 aQ; out vec2 vQ;
void main(){ vQ = aQ; gl_Position = vec4(aQ, 0.0, 1.0); }`;
const VFS = `#version 300 es
precision highp float; precision highp sampler3D;
${PI_GLSL}
in vec2 vQ;
uniform vec3 uEye, uF, uS, uU; uniform float uTan, uAspect, uH, uCloudOn;
uniform sampler3D uCloud; uniform float uNx, uNy; uniform vec3 uLight; uniform float uSteps, uD0;
// embedded regional nest: tangent-plane box (nest.ts geometry) of side uNestL (radians), top uNestTop (m)
uniform sampler3D uNest; uniform float uNestOn, uNestLat0, uNestLon0, uNestL, uNestTop, uCloudTopM; uniform vec2 uNestK; uniform float uNestDx;
const int NEST_SUB = 8;
${BSPLINE_GLSL}
bool nestUV(float lat, float lon, float w, out vec3 q) {
  float dl = lon - uNestLon0; dl -= 2.0 * PI * floor((dl + PI) / (2.0 * PI));
  q = vec3(dl * cos(uNestLat0) / uNestL + 0.5, (lat - uNestLat0) / uNestL + 0.5, w * uCloudTopM / uNestTop);
  return uNestOn > 0.5 && q.x >= 0.0 && q.x <= 1.0 && q.y >= 0.0 && q.y <= 1.0;
}
out vec4 o;
vec2 sph(vec3 ro, vec3 rd, float r){ float b = dot(ro, rd), c = dot(ro, ro) - r * r, d = b * b - c; if (d < 0.0) return vec2(1e9, -1e9); d = sqrt(d); return vec2(-b - d, -b + d); }
void main(){
  vec3 rd = normalize(uF + (uS * vQ.x * uAspect + uU * vQ.y) * uTan);
  vec2 outer = sph(uEye, rd, 1.0 + uH);
  if (outer.y < 0.0 || outer.x > outer.y) { o = vec4(0.0); return; }
  vec2 inner = sph(uEye, rd, 1.0);
  float t0 = max(outer.x, 0.0), t1 = outer.y;
  if (inner.x < inner.y && inner.x > 0.0) t1 = inner.x;
  float len = t1 - t0;
  // thin blue limb glow (optical path through the shell)
  vec3 mid = uEye + rd * (t0 + 0.5 * len);
  float day = clamp(dot(normalize(mid), uLight) * 1.5 + 0.3, 0.0, 1.0);
  vec3 glow = vec3(0.30, 0.55, 1.0) * (1.0 - exp(-len / uH * 0.10)) * day;
  vec3 acc = vec3(0.0); float T = 1.0;
  if (uCloudOn > 0.5) {
    // steps: enough to resolve the layer depth along the ray (grazing rays cross many columns);
    // segments inside an embedded regional nest are subdivided to its finer grid
    int N = int(clamp(uSteps * sqrt(len / uH), 24.0, 160.0));
    float dt = len / float(N);
    for (int i = 0; i < 160; i++) {
      if (i >= N || T < 0.02) break;
      vec3 q, pm = uEye + rd * (t0 + (float(i) + 0.5) * dt);
      vec3 nm = normalize(pm);
      float lonm = atan(-nm.z, nm.x); if (lonm < 0.0) lonm += 2.0 * PI;
      // subdivide only where the step is much longer than the nest grid spacing
      int sub = nestUV(asin(clamp(nm.y, -1.0, 1.0)), lonm, 0.0, q) ? int(clamp(dt * 6.371e6 / (2.0 * uNestDx), 1.0, float(NEST_SUB))) : 1;
      float ds = dt / float(sub);
      for (int s = 0; s < NEST_SUB; s++) {
        if (s >= sub || T < 0.02) break;
        vec3 p = uEye + rd * (t0 + float(i) * dt + (float(s) + 0.5) * ds);
        float r = length(p);
        float w = (r - 1.0) / uH;
        if (w < 0.0 || w > 1.0) continue;
        vec3 n = p / r;
        float lon = atan(-n.z, n.x); if (lon < 0.0) lon += 2.0 * PI;
        float lat = asin(clamp(n.y, -1.0, 1.0));
        float kc, kp, shade = 1.0;
        if (nestUV(lat, lon, w, q)) {
          // regional nest: bytes are sqrt(mixing ratio / scale); extinction per unit radius
          vec2 d = q.z <= 1.0 ? textureLod(uNest, q, 0.0).rg : vec2(0.0);
          kc = d.r * d.r * uNestK.x * ds; kp = d.g * d.g * uNestK.y * ds;
          if (kc > 1e-3) {
            // self-shadowing: short march toward the sun through the nest (2 km steps)
            vec3 E = vec3(-sin(lon), 0.0, -cos(lon)), Nn = vec3(-sin(lat) * cos(lon), cos(lat), sin(lat) * sin(lon));
            vec3 dq = vec3(dot(uLight, E) / (uNestL * 6.371e6), dot(uLight, Nn) / (uNestL * 6.371e6), dot(uLight, n) / uNestTop) * 2000.0;
            float od = 0.0;
            for (int l = 1; l <= 3; l++) { vec3 ql = q + dq * float(l); if (ql.z > 1.0) break; float c = textureLod(uNest, ql, 0.0).r; od += c * c; }
            shade = 0.45 + 0.55 * exp(-od * uNestK.x / 6.371e6 * 2000.0);
          }
        } else {
          vec2 cp = bspline3(uCloud, vec3(lon / (2.0 * PI) + 0.5 / uNx, 0.5 - lat / PI, w), vec2(uNx, uNy)).rg;
          // grid-box cloud fraction f: a layer of depth D0 (2 km) hides a fraction f of what lies behind it,
          // i.e. extinction -ln(1 - f) / D0; precipitation shafts (blue) hide up to half over 6 km
          kc = -log(1.0 - 0.98 * cp.r) * ds / (uD0 * uH); kp = -log(1.0 - 0.5 * cp.g) * ds / (3.0 * uD0 * uH);
        }
        if (kc + kp < 1e-4) continue;
        float a = 1.0 - exp(-(kc + kp));
        float lit = (0.25 + 0.85 * clamp(dot(n, uLight) * 1.3 + 0.15, 0.0, 1.0)) * shade;
        vec3 col = (kc * vec3(1.0) + kp * vec3(0.35, 0.55, 1.0)) / (kc + kp);
        acc += T * a * col * lit;
        T *= 1.0 - a;
      }
    }
  }
  o = vec4(acc + T * glow, 1.0 - T + T * length(glow) * 0.6);
}`;
// composite of the cached cloud-shell image (premultiplied colour)
const CFS = `#version 300 es
precision mediump float;
in vec2 vQ; uniform sampler2D uImg; out vec4 o;
void main(){ o = textureLod(uImg, vQ * 0.5 + 0.5, 0.0); }`;

function compile(gl: WebGL2RenderingContext, type: number, src: string): WebGLShader {
  const s = gl.createShader(type)!;
  gl.shaderSource(s, src); gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s) ?? 'shader');
  return s;
}
function program(gl: WebGL2RenderingContext, vs: string, fs: string): WebGLProgram {
  const p = gl.createProgram()!;
  gl.attachShader(p, compile(gl, gl.VERTEX_SHADER, vs)); gl.attachShader(p, compile(gl, gl.FRAGMENT_SHADER, fs));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p) ?? 'link');
  return p;
}

export class Globe {
  private readonly gl: WebGL2RenderingContext;
  private readonly prog: WebGLProgram;
  private readonly loc: { pos: number; col: number; mvp: WebGLUniformLocation; scale: WebGLUniformLocation; light: WebGLUniformLocation; alpha: WebGLUniformLocation; shade: WebGLUniformLocation };
  private readonly planet: { prog: WebGLProgram; vao: WebGLVertexArrayObject; count: number };
  private readonly vol: { prog: WebGLProgram; vao: WebGLVertexArrayObject; comp: WebGLProgram; tex: WebGLTexture; fbo: WebGLFramebuffer; w: number; h: number };
  /** the cloud ray march is cached in vol.tex and redone only when clouds, camera or settings change */
  private cloudDirty = true;
  private lastDraw = 0;
  private readonly mapTex: WebGLTexture; private readonly fieldTex: WebGLTexture; private readonly cloudTex: WebGLTexture; private readonly nestTex: WebGLTexture;
  /** embedded regional nest: centre (radians), side (m) and model top (m); null when none */
  private nest: { lat0: number; lon0: number; L: number; top: number } | null = null;
  private nestBuf: Uint8Array | null = null; private nestNx = 1;
  /** nest data waiting for the next (throttled) cloud-shell redraw */
  private nestPending = false; private lastShell = 0;
  private fieldSize: [number, number] = [1, 1];
  private cloudNx = 1; private cloudNy = 1; private hasCloud = false; private landOn = 0;
  private readonly grat: { vao: WebGLVertexArrayObject; count: number };
  private readonly trc: { vao: WebGLVertexArrayObject; pos: WebGLBuffer; col: WebGLBuffer };
  private trcCount = 0;
  private readonly markers = new Map<'pick' | 'nest', { vao: WebGLVertexArrayObject; count: number }>();
  /** called on a click (not a drag) on the sphere with (lat, lon) in radians, lon in [0, 2 pi) */
  onPick: ((lat: number, lon: number) => void) | null = null;
  yaw = -0.4;
  pitch = 0.35;
  /** camera distance from the centre (Earth radii); close in, the view tilts toward the horizon */
  dist = 3.2;
  /** vertical exaggeration of the atmosphere and terrain (x real scale) */
  exaggeration = 1;
  /** opacity of the model field over the map (0..1) */
  fieldAlpha = 0.75;
  /** draw the 3-D cloud shell */
  cloudsOn = true;
  /** top of the cloud texture (m) */
  cloudTop = 16000;

  constructor(private readonly canvas: HTMLCanvasElement) {
    const gl = canvas.getContext('webgl2', { antialias: true });
    if (!gl) throw new Error('WebGL2 不可用 / WebGL2 unavailable');
    this.gl = gl;
    const p = program(gl, VS, FS);
    this.prog = p;
    this.loc = {
      pos: gl.getAttribLocation(p, 'aPos'), col: gl.getAttribLocation(p, 'aCol'),
      mvp: gl.getUniformLocation(p, 'uMVP')!, scale: gl.getUniformLocation(p, 'uScale')!,
      light: gl.getUniformLocation(p, 'uLight')!, alpha: gl.getUniformLocation(p, 'uAlpha')!, shade: gl.getUniformLocation(p, 'uShade')!,
    };
    this.grat = this.buildGraticule();
    const vao = gl.createVertexArray()!, pos = gl.createBuffer()!, col = gl.createBuffer()!;
    gl.bindVertexArray(vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, pos); gl.enableVertexAttribArray(this.loc.pos); gl.vertexAttribPointer(this.loc.pos, 3, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, col); gl.enableVertexAttribArray(this.loc.col); gl.vertexAttribPointer(this.loc.col, 3, gl.FLOAT, false, 0, 0);
    gl.bindVertexArray(null);
    this.trc = { vao, pos, col };
    // planet mesh: regular longitude-latitude grid in texture coordinates
    const pp = program(gl, PVS, PFS), NU = 720, NV = 360, uv = new Float32Array((NU + 1) * (NV + 1) * 2), idx = new Uint32Array(NU * NV * 6);
    for (let j = 0; j <= NV; j++) for (let i = 0; i <= NU; i++) { const o = (j * (NU + 1) + i) * 2; uv[o] = i / NU; uv[o + 1] = j / NV; }
    let n = 0;
    for (let j = 0; j < NV; j++) for (let i = 0; i < NU; i++) { const a = j * (NU + 1) + i, b = a + 1, c = a + NU + 1, d = c + 1; idx[n++] = a; idx[n++] = c; idx[n++] = b; idx[n++] = b; idx[n++] = c; idx[n++] = d; }
    const pv = gl.createVertexArray()!;
    gl.bindVertexArray(pv);
    const ub = gl.createBuffer()!; gl.bindBuffer(gl.ARRAY_BUFFER, ub); gl.bufferData(gl.ARRAY_BUFFER, uv, gl.STATIC_DRAW);
    const la = gl.getAttribLocation(pp, 'aUV'); gl.enableVertexAttribArray(la); gl.vertexAttribPointer(la, 2, gl.FLOAT, false, 0, 0);
    const ib = gl.createBuffer()!; gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ib); gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, idx, gl.STATIC_DRAW);
    gl.bindVertexArray(null);
    this.planet = { prog: pp, vao: pv, count: n };
    // full-screen quad for the cloud shell
    const vp = program(gl, VVS, VFS), vv = gl.createVertexArray()!;
    gl.bindVertexArray(vv);
    const qb = gl.createBuffer()!; gl.bindBuffer(gl.ARRAY_BUFFER, qb); gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
    const lq = gl.getAttribLocation(vp, 'aQ'); gl.enableVertexAttribArray(lq); gl.vertexAttribPointer(lq, 2, gl.FLOAT, false, 0, 0);
    gl.bindVertexArray(null);
    const ct = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, ct);
    for (const [k, v] of [[gl.TEXTURE_MIN_FILTER, gl.LINEAR], [gl.TEXTURE_MAG_FILTER, gl.LINEAR], [gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE], [gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE]] as const) gl.texParameteri(gl.TEXTURE_2D, k, v);
    const fbo = gl.createFramebuffer()!;
    this.vol = { prog: vp, vao: vv, comp: program(gl, VVS, CFS), tex: ct, fbo, w: 0, h: 0 };
    // textures: map (R elevation / 6 km, G land fraction), field colours, 3-D cloud
    const tex2 = (): WebGLTexture => { const t = gl.createTexture()!; gl.bindTexture(gl.TEXTURE_2D, t); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE); gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array(4)); return t; };
    this.mapTex = tex2(); this.fieldTex = tex2();
    this.nestTex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_3D, this.nestTex);
    for (const [k, v] of [[gl.TEXTURE_MIN_FILTER, gl.LINEAR], [gl.TEXTURE_MAG_FILTER, gl.LINEAR], [gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE], [gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE], [gl.TEXTURE_WRAP_R, gl.CLAMP_TO_EDGE]] as const) gl.texParameteri(gl.TEXTURE_3D, k, v);
    gl.texImage3D(gl.TEXTURE_3D, 0, gl.RG8, 1, 1, 1, 0, gl.RG, gl.UNSIGNED_BYTE, new Uint8Array(2));
    this.cloudTex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_3D, this.cloudTex);
    for (const [k, v] of [[gl.TEXTURE_MIN_FILTER, gl.LINEAR], [gl.TEXTURE_MAG_FILTER, gl.LINEAR], [gl.TEXTURE_WRAP_S, gl.REPEAT], [gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE], [gl.TEXTURE_WRAP_R, gl.CLAMP_TO_EDGE]] as const) gl.texParameteri(gl.TEXTURE_3D, k, v);
    gl.texImage3D(gl.TEXTURE_3D, 0, gl.RG8, 1, 1, 1, 0, gl.RG, gl.UNSIGNED_BYTE, new Uint8Array(2));
    this.attachControls();
  }

  /** Load the relief map (PNG: R,G = elevation + 1000 m as 16 bits, B = land fraction x 255). */
  async loadMap(url: string): Promise<void> {
    const img = await createImageBitmap(await (await fetch(url)).blob(), { colorSpaceConversion: 'none', premultiplyAlpha: 'none' });
    const cv = new OffscreenCanvas(img.width, img.height), cx = cv.getContext('2d', { colorSpace: 'srgb' })!;
    cx.drawImage(img, 0, 0);
    const d = cx.getImageData(0, 0, img.width, img.height).data, out = new Uint8Array(img.width * img.height * 4);
    for (let i = 0; i < img.width * img.height; i++) {
      const e = d[4 * i]! * 256 + d[4 * i + 1]! - 1000, land = d[4 * i + 2]!;
      out[4 * i] = Math.max(0, Math.min(255, Math.round(e / 6000 * 255)));
      out[4 * i + 1] = land;
    }
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, this.mapTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, img.width, img.height, 0, gl.RGBA, gl.UNSIGNED_BYTE, out);
    this.dirty = true;
  }

  private attachControls(): void {
    attachOrbit(this.canvas, {
      rotate: (dx, dy) => { const k = 0.006 * Math.min(1, 0.4 + (this.dist - 1)); this.yaw -= dx * k; this.pitch = Math.max(-1.5, Math.min(1.5, this.pitch + dy * k)); },
      zoom: (f) => { this.dist = 1 + Math.max(3e-4, Math.min(7, (this.dist - 1) * f)); },
      tap: (x, y) => { if (this.onPick) { const p = this.pick(x, y); if (p) this.onPick(p.lat, p.lon); } },
    });
  }

  /** Screen point -> (lat, lon) on the unit sphere, or null when the ray misses it. */
  pick(clientX: number, clientY: number): { lat: number; lon: number } | null {
    const r = this.canvas.getBoundingClientRect();
    const x = (clientX - r.left) / r.width * 2 - 1, y = 1 - (clientY - r.top) / r.height * 2;
    const { e, f, s, u } = this.camera();
    const t = Math.tan(0.4), a = r.width / r.height;
    const d = norm([f[0] + (s[0] * x * a + u[0] * y) * t, f[1] + (s[1] * x * a + u[1] * y) * t, f[2] + (s[2] * x * a + u[2] * y) * t]);
    const b = dot(e, d), c = dot(e, e) - 1, disc = b * b - c;
    if (disc < 0) return null;
    const l = -b - Math.sqrt(disc);
    const p: V3 = [e[0] + l * d[0], e[1] + l * d[1], e[2] + l * d[2]];
    let lon = Math.atan2(-p[2], p[0]);
    if (lon < 0) lon += 2 * Math.PI;
    return { lat: Math.asin(Math.max(-1, Math.min(1, p[1]))), lon };
  }

  /** Outline a square region of half-width `half` (radians of arc) centred at (lat, lon); null clears it. */
  setMarker(lat: number | null, lon = 0, half = 0.1, slot: 'pick' | 'nest' = 'pick'): void {
    const gl = this.gl;
    if (lat === null) { this.markers.delete(slot); this.dirty = true; return; }
    const rgb = slot === 'nest' ? [0.3, 0.95, 1] : [1, 0.85, 0.2];
    const v: number[] = [], c: number[] = [];
    const pt = (x: number, y: number): void => {
      const la = lat + y, lo = lon + x / Math.max(0.05, Math.cos(lat));
      v.push(Math.cos(la) * Math.cos(lo), Math.sin(la), -Math.cos(la) * Math.sin(lo)); c.push(rgb[0]!, rgb[1]!, rgb[2]!);
    };
    const n = 24;
    const edge = (x0: number, y0: number, x1: number, y1: number): void => {
      for (let i = 0; i < n; i++) { pt(x0 + (x1 - x0) * i / n, y0 + (y1 - y0) * i / n); pt(x0 + (x1 - x0) * (i + 1) / n, y0 + (y1 - y0) * (i + 1) / n); }
    };
    edge(-half, -half, half, -half); edge(half, -half, half, half); edge(half, half, -half, half); edge(-half, half, -half, -half);
    const vao = gl.createVertexArray()!;
    gl.bindVertexArray(vao);
    const pb = gl.createBuffer()!; gl.bindBuffer(gl.ARRAY_BUFFER, pb); gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(v), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(this.loc.pos); gl.vertexAttribPointer(this.loc.pos, 3, gl.FLOAT, false, 0, 0);
    const cb = gl.createBuffer()!; gl.bindBuffer(gl.ARRAY_BUFFER, cb); gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(c), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(this.loc.col); gl.vertexAttribPointer(this.loc.col, 3, gl.FLOAT, false, 0, 0);
    gl.bindVertexArray(null);
    this.markers.set(slot, { vao, count: v.length / 3 });
    this.dirty = true;
  }

  private buildGraticule(): { vao: WebGLVertexArrayObject; count: number } {
    const gl = this.gl, v: number[] = [], c: number[] = [];
    const push = (lat: number, lon: number): void => {
      v.push(Math.cos(lat) * Math.cos(lon), Math.sin(lat), -Math.cos(lat) * Math.sin(lon));
      c.push(0.85, 0.9, 1);
    };
    for (let latD = -60; latD <= 60; latD += 30) for (let i = 0; i < 180; i++) {
      push(latD * Math.PI / 180, i * 2 * Math.PI / 180); push(latD * Math.PI / 180, (i + 1) * 2 * Math.PI / 180);
    }
    for (let lonD = 0; lonD < 360; lonD += 30) for (let i = 0; i < 90; i++) {
      const a = -Math.PI / 2 + i * Math.PI / 90, b = a + Math.PI / 90;
      push(a, lonD * Math.PI / 180); push(b, lonD * Math.PI / 180);
    }
    const vao = gl.createVertexArray()!;
    gl.bindVertexArray(vao);
    const pb = gl.createBuffer()!; gl.bindBuffer(gl.ARRAY_BUFFER, pb); gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(v), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(this.loc.pos); gl.vertexAttribPointer(this.loc.pos, 3, gl.FLOAT, false, 0, 0);
    const cb = gl.createBuffer()!; gl.bindBuffer(gl.ARRAY_BUFFER, cb); gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(c), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(this.loc.col); gl.vertexAttribPointer(this.loc.col, 3, gl.FLOAT, false, 0, 0);
    gl.bindVertexArray(null);
    return { vao, count: v.length / 3 };
  }

  /** Colour the model field (grid [lat][lon], rows north -> south); f may return an alpha (0 = map shows through). */
  setField(lat: Float64Array, nlon: number, values: Float32Array, f: (v: number, idx: number) => Rgb | Rgba): void {
    const nlat = lat.length, px = new Uint8Array(nlat * nlon * 4);
    for (let p = 0; p < nlat * nlon; p++) {
      const c = f(values[p]!, p);
      px[4 * p] = Math.round(Math.max(0, Math.min(1, c[0])) * 255); px[4 * p + 1] = Math.round(Math.max(0, Math.min(1, c[1])) * 255);
      px[4 * p + 2] = Math.round(Math.max(0, Math.min(1, c[2])) * 255); px[4 * p + 3] = Math.round((c.length > 3 ? (c as Rgba)[3] : 1) * 255);
    }
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, this.fieldTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, nlon, nlat, 0, gl.RGBA, gl.UNSIGNED_BYTE, px);
    this.fieldSize = [nlon, nlat];
    this.dirty = true;
  }

  /** Continents on (Earth experiments) or off (aquaplanet, idealised planets). */
  setOutline(_lat: Float64Array, _nlon: number, mask: Uint8Array | null): void { this.landOn = mask ? 1 : 0; this.dirty = true; }

  /** 3-D cloud and precipitation [k][lat][lon][2] (bytes, k = height level from sea level to cloudTop), or null. */
  setClouds(data: Uint8Array | null, nlon: number, nlat: number, nz: number, top: number): void {
    this.hasCloud = !!data;
    if (!data) return;
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_3D, this.cloudTex);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texImage3D(gl.TEXTURE_3D, 0, gl.RG8, nlon, nlat, nz, 0, gl.RG, gl.UNSIGNED_BYTE, data);
    this.cloudNx = nlon; this.cloudNy = nlat; this.cloudTop = top;
    this.dirty = true; this.cloudDirty = true;
  }

  /** Place (or remove, with null) an embedded regional nest: tangent-plane box of side L (m) centred at
   *  (lat0, lon0) with model top `top` (m), as in regional/nest.ts. */
  setNest(g: { lat0: number; lon0: number; L: number; top: number } | null): void {
    this.nest = g; this.dirty = true; this.cloudDirty = true;
  }

  /** Nest cloud and precipitation bytes [k][j][i] (j northward), as sent by the regional worker. */
  setNestData(cloud: Uint8Array, rain: Uint8Array, nx: number, ny: number, nz: number): void {
    const n = nx * ny * nz;
    if (!this.nestBuf || this.nestBuf.length !== 2 * n) this.nestBuf = new Uint8Array(2 * n);
    const b = this.nestBuf;
    for (let i = 0; i < n; i++) { b[2 * i] = cloud[i]!; b[2 * i + 1] = rain[i]!; }
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_3D, this.nestTex);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texImage3D(gl.TEXTURE_3D, 0, gl.RG8, nx, ny, nz, 0, gl.RG, gl.UNSIGNED_BYTE, b);
    this.nestNx = nx;
    this.nestPending = true;
  }

  /** Tracer streaks: pairs of unit vectors (head, tail) with colours. */
  setTracers(pos: Float32Array, col: Float32Array, count: number): void {
    if (count === 0 && this.trcCount === 0) return;
    const gl = this.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.trc.pos); gl.bufferData(gl.ARRAY_BUFFER, pos.subarray(0, count * 3), gl.DYNAMIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.trc.col); gl.bufferData(gl.ARRAY_BUFFER, col.subarray(0, count * 3), gl.DYNAMIC_DRAW);
    this.trcCount = count;
    this.dirty = true;
  }

  /** redraw needed (new data, camera, settings); the cloud ray march shares the GPU with the model */
  dirty = true;
  private lastSettings = '';

  render(): void {
    const gl = this.gl, c = this.canvas;
    const cssW = Math.max(1, c.clientWidth), cssH = Math.max(1, c.clientHeight);
    const scale = Math.min(Math.min(2, window.devicePixelRatio || 1), Math.sqrt(1.6e6 / (cssW * cssH)));
    const w = Math.max(1, Math.round(cssW * scale)), h = Math.max(1, Math.round(cssH * scale));
    if (c.width !== w || c.height !== h) { c.width = w; c.height = h; this.dirty = true; this.cloudDirty = true; }
    const settings = `${this.exaggeration}|${this.fieldAlpha}|${this.cloudsOn}|${this.yaw}|${this.pitch}|${this.dist}`;
    if (settings !== this.lastSettings) { this.lastSettings = settings; this.dirty = true; this.cloudDirty = true; }
    if (this.nestPending && now0() - this.lastShell > 1500) { this.nestPending = false; this.dirty = true; this.cloudDirty = true; }
    if (!this.dirty) return;
    // data-only redraws (tracers, fields) at most 30 per second; the model shares the GPU
    const now = performance.now();
    if (!this.cloudDirty && now - this.lastDraw < 33) return;
    this.lastDraw = now;
    this.dirty = false;
    gl.viewport(0, 0, w, h);
    gl.clearColor(0.01, 0.015, 0.03, 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.enable(gl.DEPTH_TEST);
    const mvp = this.matrix(w / h), cam = this.camera(), eye = cam.e;
    const ln = Math.hypot(eye[0] + 0.6, eye[1] + 0.8, eye[2]), L: V3 = [(eye[0] + 0.6) / ln, (eye[1] + 0.8) / ln, eye[2] / ln]; // display light: from above-left of the camera
    const H = this.exaggeration * this.cloudTop / 6.371e6, relief = this.exaggeration * 6000 / 6.371e6;
    // planet
    const P = this.planet.prog;
    gl.useProgram(P);
    gl.uniformMatrix4fv(gl.getUniformLocation(P, 'uMVP'), false, mvp);
    gl.uniform3f(gl.getUniformLocation(P, 'uLight'), L[0], L[1], L[2]);
    gl.uniform1f(gl.getUniformLocation(P, 'uFieldAlpha'), this.fieldAlpha);
    gl.uniform1f(gl.getUniformLocation(P, 'uLandOn'), this.landOn);
    gl.uniform1f(gl.getUniformLocation(P, 'uRelief'), relief);
    gl.uniform2f(gl.getUniformLocation(P, 'uFieldSize'), this.fieldSize[0], this.fieldSize[1]);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this.mapTex); gl.uniform1i(gl.getUniformLocation(P, 'uMap'), 0);
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, this.fieldTex); gl.uniform1i(gl.getUniformLocation(P, 'uField'), 1);
    gl.bindVertexArray(this.planet.vao);
    gl.drawElements(gl.TRIANGLES, this.planet.count, gl.UNSIGNED_INT, 0);
    // lines under the clouds
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.useProgram(this.prog);
    gl.uniformMatrix4fv(this.loc.mvp, false, mvp);
    gl.uniform3f(this.loc.light, L[0], L[1], L[2]);
    gl.uniform1f(this.loc.shade, 0);
    gl.uniform1f(this.loc.scale, 1.00002 + relief * 1.05); gl.uniform1f(this.loc.alpha, 0.12);
    gl.bindVertexArray(this.grat.vao);
    gl.drawArrays(gl.LINES, 0, this.grat.count);
    if (this.trcCount > 0) {
      gl.uniform1f(this.loc.scale, 1.0 + 0.5 * H); gl.uniform1f(this.loc.alpha, 0.8);
      gl.bindVertexArray(this.trc.vao);
      gl.drawArrays(gl.LINES, 0, this.trcCount);
    }
    // cloud shell and limb glow (premultiplied colour), cached at reduced resolution
    gl.disable(gl.DEPTH_TEST);
    if (this.cloudDirty) { this.drawShell(w, h, cam, L, H); this.cloudDirty = false; this.lastShell = now0(); gl.viewport(0, 0, w, h); }
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    gl.useProgram(this.vol.comp);
    gl.activeTexture(gl.TEXTURE2); gl.bindTexture(gl.TEXTURE_2D, this.vol.tex); gl.uniform1i(gl.getUniformLocation(this.vol.comp, 'uImg'), 2);
    gl.bindVertexArray(this.vol.vao);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    // marker on top
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    for (const mk of this.markers.values()) {
      gl.useProgram(this.prog);
      gl.uniform1f(this.loc.scale, 1.0 + H); gl.uniform1f(this.loc.alpha, 1);
      gl.bindVertexArray(mk.vao);
      gl.drawArrays(gl.LINES, 0, mk.count);
    }
    gl.disable(gl.BLEND);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindVertexArray(null);
  }

  private drawShell(cw: number, ch: number, cam: { e: V3; f: V3; s: V3; u: V3 }, L: V3, H: number): void {
    const gl = this.gl, vol = this.vol;
    const k = Math.min(1, Math.sqrt((this.nest ? 0.5e6 : 0.8e6) / (cw * ch))), w = Math.max(1, Math.round(cw * k)), h = Math.max(1, Math.round(ch * k));
    if (vol.w !== w || vol.h !== h) {
      vol.w = w; vol.h = h;
      gl.bindTexture(gl.TEXTURE_2D, vol.tex);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
      gl.bindFramebuffer(gl.FRAMEBUFFER, vol.fbo);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, vol.tex, 0);
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, vol.fbo);
    gl.viewport(0, 0, w, h);
    gl.disable(gl.BLEND);
    const V = this.vol.prog, eye = cam.e, f = cam.f, s2 = cam.s, u2 = cam.u;
    gl.useProgram(V);
    gl.uniform3f(gl.getUniformLocation(V, 'uEye'), eye[0], eye[1], eye[2]);
    gl.uniform3f(gl.getUniformLocation(V, 'uF'), f[0], f[1], f[2]);
    gl.uniform3f(gl.getUniformLocation(V, 'uS'), s2[0], s2[1], s2[2]);
    gl.uniform3f(gl.getUniformLocation(V, 'uU'), u2[0], u2[1], u2[2]);
    gl.uniform1f(gl.getUniformLocation(V, 'uTan'), Math.tan(0.4));
    gl.uniform1f(gl.getUniformLocation(V, 'uAspect'), cw / ch);
    gl.uniform1f(gl.getUniformLocation(V, 'uH'), H);
    gl.uniform1f(gl.getUniformLocation(V, 'uCloudOn'), this.cloudsOn && this.hasCloud ? 1 : 0);
    gl.uniform1f(gl.getUniformLocation(V, 'uNx'), this.cloudNx);
    gl.uniform1f(gl.getUniformLocation(V, 'uNy'), this.cloudNy);
    // nest extinction per unit Earth radius for byte value 1: cloud 3 g/kg x 0.04 m^-1 per g/kg
    // (softened from the ~0.15 m^-1 of real cloud so the 3 km grid does not render as solid blocks),
    // precipitation 8 g/kg x 1.5e-3 m^-1 per g/kg
    const ne = this.nest;
    gl.uniform1f(gl.getUniformLocation(V, 'uNestOn'), ne ? 1 : 0);
    gl.uniform1f(gl.getUniformLocation(V, 'uNestLat0'), ne?.lat0 ?? 0);
    gl.uniform1f(gl.getUniformLocation(V, 'uNestLon0'), ne?.lon0 ?? 0);
    gl.uniform1f(gl.getUniformLocation(V, 'uNestL'), (ne?.L ?? 1) / 6.371e6);
    gl.uniform1f(gl.getUniformLocation(V, 'uNestTop'), ne?.top ?? 1);
    gl.uniform1f(gl.getUniformLocation(V, 'uCloudTopM'), this.cloudTop);
    gl.uniform1f(gl.getUniformLocation(V, 'uNestDx'), ne ? ne.L / this.nestNx : 1);
    gl.uniform2f(gl.getUniformLocation(V, 'uNestK'), 0.12 * 6.371e6, 0.012 * 6.371e6);
    gl.activeTexture(gl.TEXTURE3); gl.bindTexture(gl.TEXTURE_3D, this.nestTex); gl.uniform1i(gl.getUniformLocation(V, 'uNest'), 3);
    gl.uniform1f(gl.getUniformLocation(V, 'uSteps'), 20);
    gl.uniform1f(gl.getUniformLocation(V, 'uD0'), 2000 / this.cloudTop);
    gl.uniform3f(gl.getUniformLocation(V, 'uLight'), L[0], L[1], L[2]);
    gl.activeTexture(gl.TEXTURE2); gl.bindTexture(gl.TEXTURE_3D, this.cloudTex); gl.uniform1i(gl.getUniformLocation(V, 'uCloud'), 2);
    gl.bindVertexArray(this.vol.vao);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }


  /** Eye position and view basis. Far out the camera looks at the centre; below ~0.8 Earth radii of
   *  altitude it tilts from straight down toward the horizon (to ~83 deg at the 2 km floor). */
  camera(): { e: V3; f: V3; s: V3; u: V3; h: number } {
    const p = this.pitch, y = this.yaw, h = this.dist - 1;
    const n: V3 = [Math.cos(p) * Math.sin(y), Math.sin(p), Math.cos(p) * Math.cos(y)];
    const t: V3 = [-Math.sin(p) * Math.sin(y), Math.cos(p), -Math.sin(p) * Math.cos(y)];
    const x = Math.max(0, Math.min(1, 1 - Math.log(h / 3e-4) / Math.log(0.8 / 3e-4)));
    const tau = 1.45 * x * x * (3 - 2 * x);
    const e: V3 = [n[0] * this.dist, n[1] * this.dist, n[2] * this.dist];
    const f = norm([-n[0] * Math.cos(tau) + t[0] * Math.sin(tau), -n[1] * Math.cos(tau) + t[1] * Math.sin(tau), -n[2] * Math.cos(tau) + t[2] * Math.sin(tau)]);
    const up: V3 = [n[0] * Math.sin(tau) + t[0] * Math.cos(tau), n[1] * Math.sin(tau) + t[1] * Math.cos(tau), n[2] * Math.sin(tau) + t[2] * Math.cos(tau)];
    const s = norm(cross(f, up)), u = cross(s, f);
    return { e, f, s, u, h };
  }

  private matrix(aspect: number): Float32Array {
    const { e, f, s, u, h } = this.camera();
    const view = [
      s[0], u[0], -f[0], 0,
      s[1], u[1], -f[1], 0,
      s[2], u[2], -f[2], 0,
      -dot(s, e), -dot(u, e), dot(f, e), 1,
    ];
    const fov = 0.8, near = Math.min(0.05, 0.3 * h), far = this.dist + 2, t = 1 / Math.tan(fov / 2);
    const proj = [t / aspect, 0, 0, 0, 0, t, 0, 0, 0, 0, (far + near) / (near - far), -1, 0, 0, 2 * far * near / (near - far), 0];
    const out = new Float32Array(16);
    for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) {
      let acc = 0;
      for (let k = 0; k < 4; k++) acc += proj[k * 4 + j]! * view[i * 4 + k]!;
      out[i * 4 + j] = acc;
    }
    return out;
  }
}

type V3 = [number, number, number];
const now0 = (): number => performance.now();
function cross(a: V3, b: V3): V3 { return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]; }
function dot(a: V3, b: V3): number { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
function norm(a: V3): V3 { const l = Math.hypot(a[0], a[1], a[2]); return [a[0] / l, a[1] / l, a[2] / l]; }
