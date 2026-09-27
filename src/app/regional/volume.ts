// WebGL2 volume renderer for the regional model: front-to-back ray marching through a 3-D texture
// (R = cloud water, G = rain water) with single-scattering sun lighting and self-shadowing,
// a coloured ground plane (surface field) and an orbit camera. Vertical scale is exaggerated.
import { attachOrbit } from '../orbitControls.js';

const VS = `#version 300 es
in vec2 aPos; out vec2 vUv;
void main(){ vUv = aPos*0.5+0.5; gl_Position = vec4(aPos,0.0,1.0); }`;

const FS = `#version 300 es
precision highp float; precision highp sampler3D;
in vec2 vUv; out vec4 o;
uniform sampler3D uVol; uniform sampler2D uGround;
uniform mat4 uInvVP; uniform vec3 uEye; uniform vec3 uBox; uniform vec3 uSun;
uniform float uCloudK; uniform float uRainK;
bool hitBox(vec3 ro, vec3 rd, out float t0, out float t1){
  vec3 inv = 1.0/rd; vec3 a = (vec3(0.0)-ro)*inv; vec3 b = (uBox-ro)*inv;
  vec3 mn = min(a,b), mx = max(a,b);
  t0 = max(max(mn.x,mn.y),mn.z); t1 = min(min(mx.x,mx.y),mx.z);
  return t1 > max(t0,0.0);
}
vec2 dens(vec3 p){ vec2 s = texture(uVol, p/uBox).rg; return vec2(s.r*s.r*uCloudK, s.g*s.g*uRainK); }
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
  if (hitBox(ro, rd, t0, t1)) {
    t0 = max(t0, 0.0);
    const int N = 160;
    float dt = (t1-t0)/float(N);
    for (int i = 0; i < N; i++) {
      float t = t0 + (float(i)+0.5)*dt;
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
  o = vec4(col + trans*bg, 1.0);
}`;

export class VolumeView {
  private readonly gl: WebGL2RenderingContext;
  private readonly prog: WebGLProgram;
  private readonly vol: WebGLTexture;
  private readonly groundTex: WebGLTexture;
  private box: [number, number, number] = [1, 1, 0.3];
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
    const vb = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, vb);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
    const loc = gl.getAttribLocation(p, 'aPos');
    gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
    this.vol = gl.createTexture()!;
    this.groundTex = gl.createTexture()!;
    attachOrbit(canvas, {
      rotate: (dx, dy) => { this.yaw -= dx * 0.006; this.pitch = Math.max(0.05, Math.min(1.5, this.pitch + dy * 0.006)); this.dirty = true; },
      zoom: (f) => { this.dist = Math.max(0.6, Math.min(6, this.dist * f)); this.dirty = true; },
    });
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
    this.box = [ax, ay, aspectZ];
    this.dirty = true;
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
    const ctr: [number, number, number] = [bx / 2, by / 2, bz * 0.3];
    const eye: [number, number, number] = [
      ctr[0] + this.dist * Math.cos(this.pitch) * Math.cos(this.yaw),
      ctr[1] + this.dist * Math.cos(this.pitch) * Math.sin(this.yaw),
      ctr[2] + this.dist * Math.sin(this.pitch),
    ];
    const vp = viewProj(eye, ctr, w / h);
    const inv = invert4(vp);
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
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }
}

type V3 = [number, number, number];
function viewProj(eye: V3, ctr: V3, aspect: number): Float32Array {
  const f = norm([ctr[0] - eye[0], ctr[1] - eye[1], ctr[2] - eye[2]]);
  const s = norm(cross(f, [0, 0, 1])), u = cross(s, f);
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
