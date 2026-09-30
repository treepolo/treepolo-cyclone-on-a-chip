// WebGL2 volume renderer for the regional model: front-to-back ray marching through a 3-D texture of display
// extinctions (R = cloud, G = precipitation, or updraft / vorticity; bytes (beta / EXT_MAX)^(1/3), display.ts), lit by
// sun and sky light that is precomputed on the CPU for every new volume (a sweep toward the sun and one straight up:
// no light marching per pixel), with forward scattering toward the sun (silver lining), cloud shadows on the ground,
// a coloured ground plane (surface field) and a map-style camera. The vertical scale is exaggerated; optical depths use
// the vertical metres per box unit in every direction, so what is drawn is what the displayed geometry would show.
// An eye nest (a finer grid in a cylinder, twoway.ts) has its own volume and light: inside the cylinder it replaces the
// outer volume, blended out toward the rim as its feedback into the outer grid is (cos^2 over Wf), with finer steps there.
// Tracer particles (with short trails) are drawn first into an offscreen buffer that keeps, per pixel, the
// nearest particle colour and its distance from the eye; the ray march composites that colour where the
// ray passes that distance, so clouds in front hide the particles behind them.
import { EXT_MAX } from '../../regional/display.js';
import type { NestFrame } from './protocol.js';

/** The eye nest's volume for the view, with the outer domain's width and height (m). */
export type NestVolume = NestFrame & { Lx: number; Ly: number };

const VS = `#version 300 es
in vec2 aPos; out vec2 vUv;
void main(){ vUv = aPos*0.5+0.5; gl_Position = vec4(aPos,0.0,1.0); }`;

const FS = `#version 300 es
precision highp float; precision highp sampler3D;
in vec2 vUv; out vec4 o;
uniform sampler3D uVol;    // r cloud extinction, g precipitation (or updraft / vorticity)
uniform sampler3D uLight;  // r transmittance to the sun, g transmittance straight up (precomputed per volume)
uniform sampler2D uGround;
uniform mat4 uInvVP; uniform vec3 uEye; uniform vec3 uBox; uniform vec3 uSun;
uniform float uK;     // extinction multiplier (the opacity slider; 1 = physical)
uniform float uMpb;   // metres per box unit (vertical scale)
uniform float uCell;  // grid spacing in box units: horizontal
uniform float uCellZ; // and vertical
uniform vec3 uOut;    // surface colour beyond the domain (sea or land)
uniform int uMode;    // channel 2: 0 precipitation, 1 updraft, 2 cyclonic vorticity
uniform vec4 uCut; uniform int uCutOn;  // cutaway: nothing where dot(uCut.xyz, p) > uCut.w
uniform sampler2D uTrC; uniform sampler2D uTrD; uniform int uTrOn;
// the eye nest: its volume and light, box (x0, y0, width; box units), cylinder (centre, radius, taper width), cells
uniform sampler3D uNVol; uniform sampler3D uNLight; uniform int uNOn;
uniform vec3 uNBox; uniform vec4 uNCyl; uniform float uNCell; uniform float uNCellZ;
// occupancy: per block of cells, 0 where the block and its neighbours hold no cloud or precipitation at all; the
// texture coordinate scale (blocks do not divide the grid exactly) and a clear-air step shorter than one block
uniform sampler3D uOcc; uniform vec3 uOccScale; uniform float uOccStep;
const float EXTMAX = ${EXT_MAX.toFixed(3)};
// weight of the nest at p: 1 inside R - Wf, cos^2 to 0 at R (the feedback's taper), 0 without one
float nestW(vec3 p){
  if (uNOn == 0) return 0.0;
  float d = length(p.xy - uNCyl.xy), a = uNCyl.z - uNCyl.w;
  if (d >= uNCyl.z) return 0.0;
  if (d <= a) return 1.0;
  float c = cos(1.5707963*(d - a)/uNCyl.w); return c*c;
}
vec3 nestQ(vec3 p){ return vec3((p.xy - uNBox.xy)/uNBox.z, p.z/uBox.z); }
vec2 volAt(vec3 p, float w){ vec2 s = texture(uVol, p/uBox).rg; if (w > 0.0) s = mix(s, texture(uNVol, nestQ(p)).rg, w); return s; }
// sun light (r) and sky light (g) at p: looked up one grid cell toward the sun (the sun is 50 degrees up: that is
// also most of a cell up, toward the sky), between the light volume's first two mipmap levels. The light of a voxel is that at its centre, so right at the surface of dense cloud
// it would blend a lit clear voxel with a dark cloudy one by where the surface happens to sit between their centres,
// and whether a voxel is in a cloud's shadow is yes or no, so a shadow edge would follow the grid as a staircase (both
// patterns of the grid, not of the cloud). A cell toward the light it is the light that reaches the surface; between
// the mipmap levels shadow edges are soft over about a cell, as light diffusing inside cloud makes them anyway.
const float LLOD = 0.6;
vec2 lightAt(vec3 p, float w){
  float c = max(uCell, uCellZ);
  vec2 L = textureLod(uLight, (p + uSun*c)/uBox, LLOD).rg;
  if (w > 0.0) L = mix(L, textureLod(uNLight, nestQ(p + uSun*max(uNCell, uNCellZ)), LLOD).rg, w);
  return L;
}
bool hitBox(vec3 ro, vec3 rd, out float t0, out float t1){
  vec3 inv = 1.0/rd; vec3 a = (vec3(0.0)-ro)*inv; vec3 b = (uBox-ro)*inv;
  vec3 mn = min(a,b), mx = max(a,b);
  t0 = max(max(mn.x,mn.y),mn.z); t1 = min(min(mx.x,mx.y),mx.z);
  return t1 > max(t0,0.0);
}
float ext(float v){ return EXTMAX*v*v*v; }
float hash(vec2 p){ return fract(sin(dot(p, vec2(12.9898,78.233)))*43758.5453); }
// Cloud and precipitation extinction (1/m) at p (box units) and the grid's cloud value: exactly the model's field (its
// grid-box means, interpolated between box centres), nothing added. Detail the grid does not resolve is not drawn:
// a partly cloudy box (sub-grid cloud) is a thin, translucent cloud, as its mean is.
vec3 dens(vec3 p, float w){
  if (uCutOn == 1 && dot(uCut.xyz, p) > uCut.w) return vec3(0.0);
  vec2 s = volAt(p, w);
  return vec3(ext(s.r)*uK, uMode == 0 ? ext(s.g)*uK : s.g*s.g*4e-4, s.r);
}
void main(){
  vec4 ndc = vec4(vUv*2.0-1.0, 1.0, 1.0);
  vec4 wp = uInvVP*ndc; vec3 rd = normalize(wp.xyz/wp.w - uEye); vec3 ro = uEye;
  float mu = dot(rd, uSun);
  vec3 sunCol = vec3(1.0, 0.96, 0.90)*1.65;
  vec3 ambCol = vec3(0.46, 0.56, 0.72);
  vec3 horizon = vec3(0.70, 0.79, 0.90);
  vec3 sky = mix(horizon, vec3(0.20,0.40,0.72), clamp(rd.z*1.6,0.0,1.0))
           + vec3(1.0,0.9,0.72)*(0.18*pow(max(mu,0.0),16.0) + 2.0*pow(max(mu,0.0),900.0));
  vec3 bg = sky;
  // ground plane z = 0: the surface field inside the domain, lit by the sun through the clouds (cloud shadows); open
  // sea or land beyond it (clear air: only the far horizon blends into the sky)
  if (rd.z < 0.0) {
    float tg = -ro.z/rd.z; vec3 pg = ro + tg*rd;
    vec3 g;
    if (pg.x >= 0.0 && pg.y >= 0.0 && pg.x <= uBox.x && pg.y <= uBox.y) {
      vec3 p0 = vec3(pg.xy, 0.0);
      vec2 L = lightAt(p0, nestW(p0));
      g = texture(uGround, pg.xy/uBox.xy).rgb * (0.40*(0.35+0.65*L.g) + 0.95*L.r*max(uSun.z,0.0));
    } else g = uOut * (0.40 + 0.95*max(uSun.z,0.0));
    bg = mix(g, horizon, smoothstep(6.0, 30.0, tg));
  }
  vec3 col = vec3(0.0); float trans = 1.0;
  vec4 tc = vec4(0.0); float tdist = 1e9;
  if (uTrOn == 1) { tc = texture(uTrC, vUv); if (tc.a > 0.0) { vec4 d4 = texture(uTrD, vUv); tdist = (d4.r * 65280.0 + d4.g * 255.0) / 65535.0 * 16.0; } }
  bool tdone = tc.a <= 0.0;
  float t0, t1;
  bool hit = hitBox(ro, rd, t0, t1);
  if (hit && uCutOn == 1) {
    // cutaway: only the part of the ray on the kept side of the plane
    float dn = dot(uCut.xyz, rd), s0 = dot(uCut.xyz, ro) - uCut.w;
    if (abs(dn) < 1e-9) { if (s0 > 0.0) hit = false; }
    else if (dn > 0.0) t1 = min(t1, -s0/dn); else t0 = max(t0, -s0/dn);
  }
  t0 = max(t0, 0.0);
  if (hit && t1 > t0) {
    // Clear air (an empty block with empty neighbours) is crossed in steps shorter than a block, so no cloud is ever
    // stepped over. Where there is cloud the steps are half a grid cell, longer (up to two cells) only where the cloud is
    // so thin that a step holds at most a quarter optical depth. Entering a new stretch of cloud the ray goes back to its last clear
    // point. A step that crosses a fixed extinction (0.3 optical depths per half cell) from below holds the surface of
    // denser cloud: it is found by bisection and the ray goes on from there in quarter steps. So the picture of a surface does not depend on where the steps happen to fall (otherwise
    // neighbouring pixels differ: speckle with random starts, contour lines with regular ones).
    float dtP = 0.5*min(uCell, uCellZ), dtN = uNOn == 1 ? 0.5*min(uNCell, uNCellZ) : dtP;
    float dtO = min(uOccStep, (t1 - t0)/40.0);
    float t = t0 + dtO*hash(gl_FragCoord.xy);
    // single scattering toward the sun (silver lining), and multiple scattering: light that has diffused through the
    // cloud (sqrt of the direct transmittance: brighter and softer than the direct beam) and the sky's light
    float phase = 0.85 + 1.8*pow(max(mu,0.0), 6.0);
    // the last sample (the ray's start counts as clear): position, optical depth per box unit, what was composited before it
    float tP = t0, bP = 0.0, trP = trans; vec3 colP = col, cP = vec3(0.0); bool tdP = tdone;
    bool inCloud = false; int hold = 0, sub = 0, refines = 0; float dtSub = 0.0;
    for (int i = 0; i < 640; i++) {
      if (t > t1 || trans < 0.01) break;
      if (!tdone && t > tdist) { col += trans*tc.rgb; trans *= 1.0 - tc.a; tdone = true; }
      vec3 p = ro + t*rd;
      if (hold == 0 && sub == 0 && texture(uOcc, p/uBox*uOccScale).r == 0.0) {
        inCloud = false; tP = t; bP = 0.0; colP = col; trP = trans; tdP = tdone;
        t += dtO; continue;
      }
      if (!inCloud) {
        // a new stretch of cloud: from the last clear point, in fine steps (at least back to here)
        inCloud = true;
        if (t - tP > 1e-6) { hold = int((t - tP)/min(dtP, dtN)) + 2; t = tP; col = colP; trans = trP; tdone = tdP; continue; }
      }
      if (hold > 0) hold--;
      float w = nestW(p);
      vec3 d = dens(p, w);
      float b = (d.x + d.y)*uMpb, dtMin = w > 0.0 ? dtN : dtP, dt;
      if (sub > 0) { dt = dtSub; sub--; }
      else {
        float bThr = 0.3/dtMin;
        if (refines < 6 && bP < bThr && b >= bThr && t - tP > 0.2*dtMin) {
          float lo = tP, hi = t;
          for (int k = 0; k < 8; k++) {
            float mid = 0.5*(lo + hi); vec3 pm = ro + mid*rd; vec3 dm = dens(pm, nestW(pm));
            if ((dm.x + dm.y)*uMpb >= bThr) hi = mid; else lo = mid;
          }
          // the thin stretch up to the surface with the last sample, then on from the surface
          col = colP; trans = trP; tdone = tdP;
          if (bP > 1e-6) { float a = 1.0 - exp(-bP*(hi - tP)); col += trans*a*cP; trans *= 1.0 - a; }
          tP = hi; bP = 0.0; colP = col; trP = trans; tdP = tdone;
          dtSub = 0.25*dtMin; sub = 12; refines++; t = hi; continue;
        }
        dt = clamp(0.25/max(b, 1e-6), dtMin, 4.0*dtMin);
      }
      tP = t; bP = b; colP = col; trP = trans; tdP = tdone; cP = vec3(0.0);
      if (b > 1e-6) {
        vec2 L = lightAt(p, w);
        float a = 1.0 - exp(-b*dt);
        vec3 cc = sunCol*(L.r*phase + 0.3*sqrt(L.r)) + ambCol*(0.36 + 0.5*L.g);
        vec3 cp = uMode == 0 ? vec3(0.46,0.54,0.66)*(0.35 + 0.65*L.r) + ambCol*0.1 : (uMode == 1 ? vec3(1.0,0.45,0.2) : vec3(0.35,0.55,1.0));
        cP = (cc*d.x + cp*d.y)/(d.x + d.y);
        col += trans*a*cP; trans *= 1.0-a;
      }
      t += dt;
    }
  }
  if (!tdone) { col += trans*tc.rgb; trans *= 1.0 - tc.a; }
  col += trans*bg;
  // soft shoulder for the bright sunlit tops
  col = vec3(1.0) - exp(-col*1.3);
  o = vec4(col, 1.0);
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

/** Cutaway of the 3-D view: the plane and which side stays (positions as fractions of the domain; x east, y north). */
export interface Cut {
  /** 'x' north-south plane at x = pos (east side removed), 'y' east-west plane at y = pos (south side removed),
   *  'z' level at pos of the top (above removed), 'line' vertical plane through the section line (right of A to B removed) */
  kind: 'x' | 'y' | 'z' | 'line';
  pos: number;
  /** remove the other side */
  flip: boolean;
  line?: { x0: number; y0: number; x1: number; y1: number };
}

/** orbit (map-style camera), free flight, or the fixed side and top views */
export type CameraMode = 'orbit' | 'fly' | 'side' | 'top';
/** direction toward the sun (box space): from the south-south-west, 50 degrees up */
const SUN: V3 = ((): V3 => { const az = 200 * Math.PI / 180, el = 50 * Math.PI / 180; return [Math.cos(el) * Math.sin(az), Math.cos(el) * Math.cos(az), Math.sin(el)]; })();

export class VolumeView {
  private readonly gl: WebGL2RenderingContext;
  private readonly prog: WebGLProgram;
  private readonly vol: WebGLTexture;
  private readonly light: WebGLTexture;
  private readonly nVol: WebGLTexture;
  private readonly nLight: WebGLTexture;
  private readonly occ: WebGLTexture;
  /** occupancy grid: texture coordinate scale and the clear-air step (box units) */
  private occScale: V3 = [1, 1, 1]; private occStep = 0.01;
  /** the eye nest (box units: x0, y0, width; cylinder centre, radius, taper), or null */
  private nest: { nx: number; nz: number; cloud: Uint8Array; x0: number; y0: number; L: number; cx: number; cy: number; R: number; Wf: number } | null = null;
  /** optical depth toward the sun of every outer voxel ([k][j][i]), from the last lighting (the nest's rays leave its box into it) */
  private odAll: Float32Array | null = null;
  private readonly groundTex: WebGLTexture;
  private box: [number, number, number] = [1, 1, 0.3];
  // tracer particles
  private trProg: WebGLProgram | null = null;
  private trBuf: WebGLBuffer | null = null;
  private trFbo: { fb: WebGLFramebuffer; c: WebGLTexture; d: WebGLTexture; z: WebGLRenderbuffer; w: number; h: number } | null = null;
  private trHist: Float32Array | null = null; private trCnt: Uint8Array | null = null; private trAge: Float32Array | null = null;
  private trLines = 0; private trPoints = 0;
  private quad: WebGLBuffer | null = null;
  /** the current volume (for the lighting sweep), its height (m), the second channel's meaning */
  private data: { nx: number; ny: number; nz: number; cloud: Uint8Array; top: number } | null = null;
  private mode = 0;
  /** extinction multiplier of the opacity slider (1 = physical) and the one the lighting was computed with */
  private kExt = 0.3; private kLight = -1;
  yaw = -Math.PI / 2; pitch = 0.55; dist = 1.35;
  /** vertical field of view (radians) */
  private fov = 0.9;
  /** Set the vertical field of view (degrees, 15-140). */
  setFov(deg: number): void { this.fov = Math.max(15, Math.min(140, deg)) * Math.PI / 180; this.dirty = true; }

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
    this.light = gl.createTexture()!;
    this.nVol = gl.createTexture()!;
    this.nLight = gl.createTexture()!;
    this.occ = gl.createTexture()!;
    this.groundTex = gl.createTexture()!;
    // empty volumes and light until the first frame (the sampler types must be complete)
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    for (const [t, v] of [[this.vol, 0], [this.light, 255], [this.nVol, 0], [this.nLight, 255]] as const) {
      gl.bindTexture(gl.TEXTURE_3D, t);
      gl.texImage3D(gl.TEXTURE_3D, 0, gl.RG8, 1, 1, 1, 0, gl.RG, gl.UNSIGNED_BYTE, new Uint8Array([v, v]));
    }
    gl.bindTexture(gl.TEXTURE_3D, this.occ);
    gl.texImage3D(gl.TEXTURE_3D, 0, gl.R8, 1, 1, 1, 0, gl.RED, gl.UNSIGNED_BYTE, new Uint8Array([255]));
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    this.controls();
  }

  // ---------------------------------------------------------------- camera
  // Map-style camera around a target point: drag to move the ground with the pointer (grab), right- or middle-drag (or
  // Ctrl / Alt + drag) to turn and tilt, wheel to zoom toward the pointer, double-click / double-tap to zoom in there. Touch: one
  // finger moves, two fingers pinch (zoom), twist (turn), move up and down together (tilt) or move together (pan).
  // Keys: arrows / WASD move, Q / E turn, R / F tilt, + / - zoom. The screen's up direction always follows the heading,
  // so straight down is just the steepest tilt (nothing locks). Free flight: WASD / QE or the on-screen buttons move,
  // dragging turns the head.
  /** orbit target (box units); null: the domain centre near the ground */
  private tgt: V3 | null = null;
  /** inverse view-projection, view-projection and eye of the last drawn frame (labels, picking) */
  private lastInv: Float32Array | null = null; private lastEye: V3 = [0, 0, 0]; private lastVP: Float32Array | null = null;
  /** tap on the view (no drag): screen position, for interaction tools; returns true when it used the tap */
  onTap: ((clientX: number, clientY: number) => boolean) | null = null;
  /** an interaction tool is active: one-finger / left drags go to onToolDrag (painting), right drags turn, Shift drags move */
  toolActive = false;
  onToolDrag: ((clientX: number, clientY: number, phase: 'start' | 'move' | 'end') => void) | null = null;
  /** pointer moving over the view without a button (mouse), or leaving it (null): for a tool's preview */
  onHover: ((clientX: number, clientY: number) => void) | null = null;
  onLeave: (() => void) | null = null;

  private target(): V3 { const [bx, by, bz] = this.box; return this.tgt ?? [bx / 2, by / 2, bz * 0.15]; }
  /** Eye and camera axes (forward, right, up) of the current state. */
  private basis(): { eye: V3; f: V3; r: V3; u: V3 } {
    if (this.fly) {
      const fl = this.fly, cp = Math.cos(fl.pitch), f: V3 = [cp * Math.cos(fl.yaw), cp * Math.sin(fl.yaw), Math.sin(fl.pitch)];
      const r: V3 = [Math.sin(fl.yaw), -Math.cos(fl.yaw), 0];
      return { eye: [fl.eye[0], fl.eye[1], fl.eye[2]], f, r, u: cross(r, f) };
    }
    const c = this.target(), cp = Math.cos(this.pitch), sp = Math.sin(this.pitch), cy = Math.cos(this.yaw), sy = Math.sin(this.yaw);
    // (past straight down, cos(pitch) < 0: the eye is on the other side and the right vector keeps the heading's, so the
    // picture turns over smoothly instead of spinning)
    const eye: V3 = [c[0] + this.dist * cp * cy, c[1] + this.dist * cp * sy, c[2] + this.dist * sp];
    const f: V3 = [-cp * cy, -cp * sy, -sp], r: V3 = [-sy, cy, 0];
    return { eye, f, r, u: cross(r, f) };
  }
  /** Ray through a screen point (client coordinates) with the current camera, box units. */
  pickRay(clientX: number, clientY: number): { o: V3; d: V3 } | null {
    const rc = this.canvas.getBoundingClientRect(); if (rc.width < 1 || rc.height < 1) return null;
    const nx = (clientX - rc.left) / rc.width * 2 - 1, ny = 1 - (clientY - rc.top) / rc.height * 2, t = Math.tan(this.fov / 2), asp = rc.width / rc.height;
    const { eye, f, r, u } = this.basis();
    return { o: eye, d: norm([f[0] + r[0] * nx * t * asp + u[0] * ny * t, f[1] + r[1] * nx * t * asp + u[1] * ny * t, f[2] + r[2] * nx * t * asp + u[2] * ny * t]) };
  }
  /** Where the ray through a screen point meets the horizontal plane at height zb (box units), or null. */
  pickPlane(clientX: number, clientY: number, zb = 0): V3 | null {
    const r = this.pickRay(clientX, clientY); if (!r || Math.abs(r.d[2]) < 1e-6) return null;
    const t = (zb - r.o[2]) / r.d[2]; if (t <= 0) return null;
    return [r.o[0] + t * r.d[0], r.o[1] + t * r.d[1], zb];
  }
  /** Ground point under a screen point, or null when the ray misses the ground or meets it very far away. */
  private groundAt(clientX: number, clientY: number): V3 | null {
    const p = this.pickPlane(clientX, clientY, 0); if (!p) return null;
    const e = this.basis().eye;
    return Math.hypot(p[0] - e[0], p[1] - e[1], p[2] - e[2]) < 8 * Math.max(this.dist, 0.2) ? p : null;
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
  /** the box size (box units; x is 1) */
  get boxSize(): V3 { return [this.box[0], this.box[1], this.box[2]]; }
  /** Direction of north on the screen: radians clockwise from straight up. */
  get northAngle(): number {
    const { f, r } = this.basis(), fh = Math.hypot(f[0], f[1]);
    // screen up on the ground is the heading (the forward direction when looking ahead, the camera's up when looking down)
    const u = fh > 1e-3 ? [f[0] / fh, f[1] / fh] : [-r[1], r[0]];
    return Math.atan2(r[1], u[1]!);
  }
  /** Turn to face north (north up on the screen), keeping the tilt and position. */
  faceNorth(): void { if (this.fly) this.fly.yaw = Math.PI / 2; else this.yaw = -Math.PI / 2; this.dirty = true; }

  private clampTarget(): void {
    const [bx, by, bz] = this.box, t = this.target();
    this.tgt = [Math.max(-0.25 * bx, Math.min(1.25 * bx, t[0])), Math.max(-0.25 * by, Math.min(1.25 * by, t[1])), bz * 0.15];
  }
  /** whether a tilt keeps the orbit eye above the ground */
  private eyeAbove(p: number): boolean { return this.target()[2] + this.dist * Math.sin(p) >= 0.003; }
  /** Turn and tilt. No angle limits: the orbit camera can pass over the top (the picture then turns upside down) and
   *  look up from below the target, the flying eye can look straight up or down and beyond; only the orbit eye stays
   *  above the ground. */
  private turn(dx: number, dy: number): void {
    const wrap = (a: number): number => a - 2 * Math.PI * Math.floor((a + Math.PI) / (2 * Math.PI));
    if (this.fly) { this.fly.yaw -= dx * 0.004; this.fly.pitch = wrap(this.fly.pitch - dy * 0.004); }
    else {
      this.yaw -= dx * 0.006;
      const p = wrap(this.pitch + dy * 0.006);
      if (this.eyeAbove(p)) this.pitch = p;
    }
    this.dirty = true;
  }
  /** Zoom by factor f (< 1 closer) toward a screen point: the ground point under it stays put. */
  private zoomAt(clientX: number, clientY: number, f: number): void {
    if (this.fly) { this.moveFly(0.6 * (1 - f) * Math.max(0.05, this.fly.eye[2] * 4), 0, 0); this.dirty = true; return; }
    const d = Math.max(0.03, Math.min(6, this.dist * f)), g = d / this.dist, p = this.groundAt(clientX, clientY), t = this.target();
    if (p) this.tgt = [p[0] + (t[0] - p[0]) * g, p[1] + (t[1] - p[1]) * g, t[2]];
    this.dist = d; this.clampTarget();
    // closing in from below: lift the eye back above the ground
    while (!this.eyeAbove(this.pitch) && this.pitch < 0) this.pitch = Math.min(0, this.pitch + 0.02);
    this.dirty = true;
  }
  /** Move the view so that the ground point `grab` comes under the screen point (grab-to-pan); false when not possible. */
  private dragGround(grab: V3, clientX: number, clientY: number): boolean {
    const p = this.groundAt(clientX, clientY); if (!p) return false;
    const t = this.target();
    this.tgt = [t[0] + grab[0] - p[0], t[1] + grab[1] - p[1], t[2]]; this.clampTarget();
    this.dirty = true;
    return true;
  }
  /** Screen-space move (pixels; used where the ground is not under the pointer, and for the keys). */
  private pan(dx: number, dy: number): void {
    if (this.fly) { this.moveFly(0, -dx * 0.002, dy * 0.002); return; }
    const t = this.target(), k = this.dist * 1.2 / Math.max(1, this.canvas.clientHeight);
    const rx = -Math.sin(this.yaw), ry = Math.cos(this.yaw), fx = -Math.cos(this.yaw), fy = -Math.sin(this.yaw);
    this.tgt = [t[0] - k * dx * rx + k * dy * fx, t[1] - k * dx * ry + k * dy * fy, t[2]]; this.clampTarget();
    this.dirty = true;
  }

  private controls(): void {
    const canvas = this.canvas, pts = new Map<number, { x: number; y: number }>();
    let mode: 'pan' | 'rotate' | 'tool' | 'look' | 'multi' = 'pan', moved = 0, lastTap = 0, grab: V3 | null = null;
    let two: { mx: number; my: number; s: number; a: number } | null = null;
    const twoState = (): { mx: number; my: number; s: number; a: number } => {
      const [a, b] = [...pts.values()];
      return { mx: (a!.x + b!.x) / 2, my: (a!.y + b!.y) / 2, s: Math.hypot(b!.x - a!.x, b!.y - a!.y), a: Math.atan2(b!.y - a!.y, b!.x - a!.x) };
    };
    canvas.addEventListener('contextmenu', (e) => e.preventDefault());
    // the middle button turns the view: no auto-scroll of the page
    canvas.addEventListener('mousedown', (e) => { if (e.button === 1) e.preventDefault(); });
    canvas.addEventListener('auxclick', (e) => { if (e.button === 1) e.preventDefault(); });
    canvas.addEventListener('pointerdown', (e) => {
      if (e.button === 1) e.preventDefault();
      canvas.setPointerCapture(e.pointerId);
      pts.set(e.pointerId, { x: e.clientX, y: e.clientY });
      if (pts.size === 1) {
        moved = 0;
        if (this.fly) mode = 'look';
        else if (this.toolActive && e.button === 0 && !e.shiftKey) mode = 'tool';
        else if (e.button === 1 || e.button === 2 || (e.button === 0 && (e.ctrlKey || e.altKey || e.metaKey))) mode = 'rotate';
        else mode = 'pan';
        if (mode === 'tool') this.onToolDrag?.(e.clientX, e.clientY, 'start');
        grab = mode === 'pan' ? this.groundAt(e.clientX, e.clientY) : null;
      } else if (pts.size === 2) {
        if (mode === 'tool') this.onToolDrag?.(e.clientX, e.clientY, 'end');
        mode = 'multi'; moved = 99; two = twoState();
      }
    });
    const end = (e: PointerEvent): void => {
      const was = pts.get(e.pointerId);
      pts.delete(e.pointerId);
      if (was && mode === 'tool' && pts.size === 0) this.onToolDrag?.(e.clientX, e.clientY, 'end');
      if (pts.size === 1 && mode === 'multi') {
        // one finger left after a two-finger gesture: carry on moving with it
        const [q] = [...pts.values()]; mode = 'pan'; grab = this.groundAt(q!.x, q!.y); two = null;
      }
      if (!was || pts.size > 0 || moved >= 6) return;
      const now = performance.now();
      // double tap / double click: zoom in there (not with a tool: a second tap is a second placement)
      if (!this.toolActive && now - lastTap < 350) { lastTap = 0; this.zoomAt(e.clientX, e.clientY, 0.5); return; }
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
      if (pts.size >= 2 && mode === 'multi') {
        const n = twoState(), o = two ?? n;
        if (o.s > 0 && n.s > 0) this.zoomAt(n.mx, n.my, o.s / n.s);
        let da = n.a - o.a; if (da > Math.PI) da -= 2 * Math.PI; if (da < -Math.PI) da += 2 * Math.PI;
        if (!this.fly) this.yaw += da;
        const mdx = n.mx - o.mx, mdy = n.my - o.my;
        // both fingers moving up or down together, side by side: tilt; otherwise move the map with them
        if (Math.abs(mdy) > 2 * Math.abs(mdx) && Math.abs(Math.sin(n.a)) < 0.65) this.turn(0, mdy);
        else if (!this.fly) {
          const g0 = this.groundAt(o.mx, o.my);
          if (!(g0 && this.dragGround(g0, n.mx, n.my))) this.pan(mdx, mdy);
        }
        two = n; this.dirty = true;
        return;
      }
      moved += Math.abs(dx) + Math.abs(dy);
      if (mode === 'tool') { this.onToolDrag?.(e.clientX, e.clientY, 'move'); return; }
      if (mode === 'rotate' || mode === 'look') { this.turn(dx, dy); return; }
      if (!(grab && this.dragGround(grab, e.clientX, e.clientY))) this.pan(dx, dy);
    });
    canvas.addEventListener('wheel', (e) => { e.preventDefault(); this.zoomAt(e.clientX, e.clientY, Math.exp(e.deltaY * 0.0012)); }, { passive: false });
    // keys (ignored while typing in a form field): orbit - arrows / WASD move, Q / E turn, R / F tilt, + / - zoom;
    // free flight - WASD / arrows fly, Q / E down / up (Shift faster)
    const typing = (): boolean => { const a = document.activeElement; return !!a && (a.tagName === 'INPUT' || a.tagName === 'SELECT' || a.tagName === 'TEXTAREA'); };
    const KEYS = ['w', 'a', 's', 'd', 'q', 'e', 'r', 'f', '+', '=', '-', 'arrowup', 'arrowdown', 'arrowleft', 'arrowright', 'shift'];
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

  /** free-flight camera: eye position (box units) and look direction; null = orbit camera */
  private fly: { eye: [number, number, number]; yaw: number; pitch: number } | null = null;
  private readonly keys = new Set<string>();
  private lastT = 0;
  setCamera(mode: CameraMode): void {
    // fixed views of the whole domain, north up: level from the south, or straight down (dragging turns them freely)
    if (mode === 'side') { this.fly = null; this.tgt = null; this.yaw = -Math.PI / 2; this.pitch = 0.02; this.dist = 1.0; this.dirty = true; return; }
    if (mode === 'top') { this.fly = null; this.tgt = null; this.yaw = -Math.PI / 2; this.pitch = Math.PI / 2; this.dist = 1.05; this.dirty = true; return; }
    if (mode === 'orbit') { this.fly = null; this.tgt = null; this.yaw = -Math.PI / 2; this.pitch = 0.55; this.dist = 1.35; this.dirty = true; return; }
    if (this.fly) return;
    // start where the orbit camera is, looking at the same point
    const { eye, f } = this.basis();
    this.fly = { eye, yaw: Math.atan2(f[1], f[0]), pitch: Math.asin(Math.max(-1, Math.min(1, f[2]))) };
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

  /** The second volume channel's meaning (0 precipitation, 1 updraft, 2 vorticity). */
  setMode(mode: number): void { this.mode = mode | 0; this.dirty = true; }

  /** Upload a volume ([k][j][i] cloud and second-channel bytes) into a 3-D texture (RG, linear, clamped). */
  private upload(tex: WebGLTexture, nx: number, ny: number, nz: number, a: Uint8Array, b: Uint8Array): void {
    const gl = this.gl, rg = new Uint8Array(nx * ny * nz * 2);
    for (let i = 0; i < a.length; i++) { rg[2 * i] = a[i]!; rg[2 * i + 1] = b[i]!; }
    gl.bindTexture(gl.TEXTURE_3D, tex);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texImage3D(gl.TEXTURE_3D, 0, gl.RG8, nx, ny, nz, 0, gl.RG, gl.UNSIGNED_BYTE, rg);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    for (const w of [gl.TEXTURE_WRAP_S, gl.TEXTURE_WRAP_T, gl.TEXTURE_WRAP_R]) gl.texParameteri(gl.TEXTURE_3D, w, gl.CLAMP_TO_EDGE);
  }

  /** Upload the display volume [k][j][i] (cloud and precipitation bytes, display.ts) of a domain `top` metres high, set
   *  the box aspect (x, y normalised to 1) and light it; with the eye nest's volume, or none. */
  setVolume(nx: number, ny: number, nz: number, cloud: Uint8Array, rain: Uint8Array, aspectZ: number, top: number, nest: NestVolume | null = null): void {
    this.upload(this.vol, nx, ny, nz, cloud, rain);
    const ax = 1, ay = ny / nx;
    // a new domain shape: back to the domain centre (a new vertical exaggeration keeps the view)
    if (ax !== this.box[0] || ay !== this.box[1]) this.tgt = null;
    this.box = [ax, ay, aspectZ];
    this.data = { nx, ny, nz, cloud, top };
    this.nest = null;
    if (nest) {
      const u = 1 / nest.Lx;
      this.upload(this.nVol, nest.nx, nest.nx, nest.nz, nest.cloud, nest.rain);
      this.nest = { nx: nest.nx, nz: nest.nz, cloud: nest.cloud, x0: nest.x0 * u, y0: nest.y0 * u, L: nest.L * u, cx: nest.cx * u, cy: nest.cy * u, R: nest.R * u, Wf: nest.Wf * u };
    }
    this.buildOccupancy(nx, ny, nz, cloud, rain, nest);
    this.computeLight();
    this.dirty = true;
  }
  /**
   * Occupancy grid for skipping clear air: blocks of B x B cells across and Bz levels (about as tall as wide in the
   * box), 255 where the block or any of its 26 neighbours holds any cloud or second-channel value (outer grid or nest),
   * 0 elsewhere. A ray in an empty block may step up to a block length without passing any cloud (the trilinear
   * sampling reaches one cell into a neighbour, which is empty as well).
   */
  private buildOccupancy(nx: number, ny: number, nz: number, cloud: Uint8Array, rain: Uint8Array, nest: NestVolume | null): void {
    const gl = this.gl, [bx, by, bz] = this.box, B = 8, Bz = Math.max(1, Math.min(16, Math.round(B * (bx / nx) / (bz / nz))));
    const mx = Math.ceil(nx / B), my = Math.ceil(ny / B), mz = Math.ceil(nz / Bz), raw = new Uint8Array(mx * my * mz);
    for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) {
      const o = (k * ny + j) * nx, ob = (Math.floor(k / Bz) * my + Math.floor(j / B)) * mx;
      for (let i = 0; i < nx; i++) if (cloud[o + i] || rain[o + i]) raw[ob + Math.floor(i / B)] = 1;
    }
    if (nest) {
      const cn = nest.nx, cz = nest.nz, u = 1 / nest.Lx;
      for (let k = 0; k < cz; k++) {
        const kb = Math.min(mz - 1, Math.floor(((k + 0.5) / cz * nz) / Bz));
        for (let j = 0; j < cn; j++) {
          const jb = Math.max(0, Math.min(my - 1, Math.floor(((nest.y0 + (j + 0.5) / cn * nest.L) * u / by * ny) / B))), o = (k * cn + j) * cn;
          for (let i = 0; i < cn; i++) if (nest.cloud[o + i] || nest.rain[o + i]) {
            const ib = Math.max(0, Math.min(mx - 1, Math.floor(((nest.x0 + (i + 0.5) / cn * nest.L) * u / bx * nx) / B)));
            raw[(kb * my + jb) * mx + ib] = 1;
          }
        }
      }
    }
    const occ = new Uint8Array(mx * my * mz);
    for (let k = 0; k < mz; k++) for (let j = 0; j < my; j++) for (let i = 0; i < mx; i++) {
      let any = 0;
      for (let c = Math.max(0, k - 1); c <= Math.min(mz - 1, k + 1) && !any; c++) for (let b = Math.max(0, j - 1); b <= Math.min(my - 1, j + 1) && !any; b++)
        for (let a = Math.max(0, i - 1); a <= Math.min(mx - 1, i + 1); a++) if (raw[(c * my + b) * mx + a]) { any = 1; break; }
      occ[(k * my + j) * mx + i] = any ? 255 : 0;
    }
    gl.bindTexture(gl.TEXTURE_3D, this.occ);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texImage3D(gl.TEXTURE_3D, 0, gl.R8, mx, my, mz, 0, gl.RED, gl.UNSIGNED_BYTE, occ);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    for (const w of [gl.TEXTURE_WRAP_S, gl.TEXTURE_WRAP_T, gl.TEXTURE_WRAP_R]) gl.texParameteri(gl.TEXTURE_3D, w, gl.CLAMP_TO_EDGE);
    this.occScale = [nx / (mx * B), ny / (my * B), nz / (mz * Bz)];
    this.occStep = 0.95 * Math.min(B * bx / nx, B * by / ny, Bz * bz / nz);
  }

  /** Drop the eye nest's volume (the nest stopped). */
  clearNest(): void { if (this.nest) { this.nest = null; this.dirty = true; } }

  /**
   * Sun and sky light of every voxel (bytes: transmittance to the sun, transmittance straight up) from the cloud
   * extinction: one sweep down the levels toward the sun (the light of the level above, shifted toward the sun, times
   * this layer's transmittance) and one straight down. Optical depths in physical vertical metres.
   */
  private computeLight(): void {
    const d = this.data, gl = this.gl; if (!d) return;
    const { nx, ny, nz, top } = d, np = nx * ny, k = this.kExt, [bx, by, bz] = this.box;
    // a cutaway: the removed part casts no shadow
    let cloud = d.cloud;
    const cp = this.cutPlane();
    if (cp) {
      cloud = Uint8Array.from(cloud);
      for (let kk = 0; kk < nz; kk++) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
        if (cp.n[0] * (i + 0.5) / nx * bx + cp.n[1] * (j + 0.5) / ny * by + cp.n[2] * (kk + 0.5) / nz * bz > cp.d) cloud[(kk * ny + j) * nx + i] = 0;
      }
    }
    const lut = new Float32Array(256); for (let v = 0; v < 256; v++) lut[v] = EXT_MAX * (v / 255) ** 3 * k;
    const dzm = top / nz, s = SUN;
    // horizontal shift toward the sun per level (cells) and the slant path per level (m)
    const di = (bz / nz) * s[0] / s[2] / (bx / nx), dj = (bz / nz) * s[1] / s[2] / (by / ny), slant = dzm / s[2];
    let od = new Float32Array(np), prev = new Float32Array(np);
    const sky = new Float32Array(np), out = new Uint8Array(2 * nz * np), all = this.nest ? new Float32Array(nz * np) : null;
    const put = (kk: number): void => {
      const o = kk * np;
      for (let c = 0; c < np; c++) { out[2 * (o + c)] = Math.round(255 * Math.exp(-od[c]!)); out[2 * (o + c) + 1] = Math.round(255 * Math.exp(-sky[c]!)); }
      all?.set(od, o);
    };
    { const o = (nz - 1) * np; for (let c = 0; c < np; c++) { const b = lut[cloud[o + c]!]!; od[c] = 0.5 * b * slant; sky[c] = 0.5 * b * dzm; } put(nz - 1); }
    for (let kk = nz - 2; kk >= 0; kk--) {
      [od, prev] = [prev, od];
      const o = kk * np, oa = (kk + 1) * np;
      for (let j = 0; j < ny; j++) {
        const y = Math.max(0, Math.min(ny - 1, j + dj)), j0 = Math.min(ny - 2, Math.floor(y)), fy = ny > 1 ? y - j0 : 0;
        for (let i = 0; i < nx; i++) {
          const x = Math.max(0, Math.min(nx - 1, i + di)), i0 = Math.min(nx - 2, Math.floor(x)), fx = nx > 1 ? x - i0 : 0;
          const a = j0 * nx + i0, above = (prev[a]! * (1 - fx) + prev[a + 1]! * fx) * (1 - fy) + (prev[a + nx]! * (1 - fx) + prev[a + nx + 1]! * fx) * fy;
          const c = j * nx + i, b = 0.5 * (lut[cloud[o + c]!]! + lut[cloud[oa + c]!]!);
          od[c] = above + b * slant; sky[c] = sky[c]! + b * dzm;
        }
      }
      put(kk);
    }
    this.uploadLight(this.light, nx, ny, nz, out);
    this.odAll = all;
    this.kLight = k;
    this.computeNestLight();
  }
  /** Upload a light volume with its mipmaps (the shader reads it between the first two levels: soft shadow edges). */
  private uploadLight(tex: WebGLTexture, nx: number, ny: number, nz: number, out: Uint8Array): void {
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_3D, tex);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texImage3D(gl.TEXTURE_3D, 0, gl.RG8, nx, ny, nz, 0, gl.RG, gl.UNSIGNED_BYTE, out);
    gl.generateMipmap(gl.TEXTURE_3D);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    for (const w of [gl.TEXTURE_WRAP_S, gl.TEXTURE_WRAP_T, gl.TEXTURE_WRAP_R]) gl.texParameteri(gl.TEXTURE_3D, w, gl.CLAMP_TO_EDGE);
  }

  /**
   * The eye nest's sun and sky light, as computeLight on its finer grid: the sweep toward the sun runs through the nest's
   * own cloud while the path stays in its box and takes the outer grid's optical depth where it leaves the box; the
   * sky light is its own column (same top).
   */
  private computeNestLight(): void {
    const n = this.nest, d = this.data, all = this.odAll; if (!n || !d || !all) return;
    const { nx: cn, nz: cz } = n, np = cn * cn, [bx, by, bz] = this.box, k = this.kExt, s = SUN;
    let cloud = n.cloud;
    const cp = this.cutPlane();
    if (cp) {
      cloud = Uint8Array.from(cloud);
      for (let kk = 0; kk < cz; kk++) for (let j = 0; j < cn; j++) for (let i = 0; i < cn; i++) {
        if (cp.n[0] * (n.x0 + (i + 0.5) / cn * n.L) + cp.n[1] * (n.y0 + (j + 0.5) / cn * n.L) + cp.n[2] * (kk + 0.5) / cz * bz > cp.d) cloud[(kk * cn + j) * cn + i] = 0;
      }
    }
    const lut = new Float32Array(256); for (let v = 0; v < 256; v++) lut[v] = EXT_MAX * (v / 255) ** 3 * k;
    const dzm = d.top / cz, cell = n.L / cn, di = (bz / cz) * s[0] / s[2] / cell, dj = (bz / cz) * s[1] / s[2] / cell, slant = dzm / s[2];
    // the outer grid's optical depth at box point (x, y) and height fraction zf (bilinear across, linear between levels)
    const onx = d.nx, ony = d.ny, onz = d.nz, onp = onx * ony;
    const outer = (x: number, y: number, zf: number): number => {
      const fx = Math.max(0, Math.min(onx - 1, x / bx * onx - 0.5)), fy = Math.max(0, Math.min(ony - 1, y / by * ony - 0.5)), fz = Math.max(0, Math.min(onz - 1, zf * onz - 0.5));
      const i0 = Math.min(onx - 2, Math.floor(fx)), j0 = Math.min(ony - 2, Math.floor(fy)), k0 = Math.min(onz - 2, Math.floor(fz));
      const wx = onx > 1 ? fx - i0 : 0, wy = ony > 1 ? fy - j0 : 0, wz = onz > 1 ? fz - k0 : 0;
      const at = (kk: number): number => {
        const o = Math.max(0, kk) * onp + Math.max(0, j0) * onx + Math.max(0, i0);
        return (all[o]! * (1 - wx) + all[o + (onx > 1 ? 1 : 0)]! * wx) * (1 - wy) + (all[o + (ony > 1 ? onx : 0)]! * (1 - wx) + all[o + (ony > 1 ? onx : 0) + (onx > 1 ? 1 : 0)]! * wx) * wy;
      };
      return onz > 1 ? at(k0) * (1 - wz) + at(k0 + 1) * wz : at(0);
    };
    let od = new Float32Array(np), prev = new Float32Array(np);
    const sky = new Float32Array(np), out = new Uint8Array(2 * cz * np);
    const put = (kk: number): void => { const o = kk * np; for (let c = 0; c < np; c++) { out[2 * (o + c)] = Math.round(255 * Math.exp(-od[c]!)); out[2 * (o + c) + 1] = Math.round(255 * Math.exp(-sky[c]!)); } };
    { const o = (cz - 1) * np; for (let c = 0; c < np; c++) { const b = lut[cloud[o + c]!]!; od[c] = 0.5 * b * slant; sky[c] = 0.5 * b * dzm; } put(cz - 1); }
    for (let kk = cz - 2; kk >= 0; kk--) {
      [od, prev] = [prev, od];
      const o = kk * np, oa = (kk + 1) * np, zfa = (kk + 1.5) / cz;
      for (let j = 0; j < cn; j++) {
        const y = j + dj, inY = y >= 0 && y <= cn - 1, j0 = Math.min(cn - 2, Math.max(0, Math.floor(y))), fy = y - j0;
        for (let i = 0; i < cn; i++) {
          const x = i + di;
          let above: number;
          if (inY && x >= 0 && x <= cn - 1) {
            const i0 = Math.min(cn - 2, Math.max(0, Math.floor(x))), fx = x - i0, a = j0 * cn + i0;
            above = (prev[a]! * (1 - fx) + prev[a + 1]! * fx) * (1 - fy) + (prev[a + cn]! * (1 - fx) + prev[a + cn + 1]! * fx) * fy;
          } else above = outer(n.x0 + (x + 0.5) * cell, n.y0 + (y + 0.5) * cell, zfa);
          const c = j * cn + i, b = 0.5 * (lut[cloud[o + c]!]! + lut[cloud[oa + c]!]!);
          od[c] = above + b * slant; sky[c] = sky[c]! + b * dzm;
        }
      }
      put(kk);
    }
    this.uploadLight(this.nLight, cn, cn, cz, out);
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

  private outside: V3 = [0.10, 0.17, 0.30];
  private cut: Cut | null = null;
  /** time (ms) at which the light is recomputed for a changed cutaway (0: none due) */
  private lightDue = 0;
  /** Cutaway (null: whole volume). The light is recomputed without the removed part (shortly after the last change). */
  setCut(c: Cut | null): void { this.cut = c; this.dirty = true; this.lightDue = performance.now() + 120; }
  /** the cut plane in box space: nothing where dot(n, p) > d */
  private cutPlane(): { n: V3; d: number } | null {
    const c = this.cut; if (!c) return null;
    const [bx, by, bz] = this.box;
    let n: V3, d: number;
    if (c.kind === 'x') { n = [1, 0, 0]; d = c.pos * bx; }
    else if (c.kind === 'y') { n = [0, -1, 0]; d = -c.pos * by; }
    else if (c.kind === 'z') { n = [0, 0, 1]; d = c.pos * bz; }
    else {
      const l = c.line; if (!l) return null;
      const tx = (l.x1 - l.x0) * bx, ty = (l.y1 - l.y0) * by, len = Math.hypot(tx, ty);
      if (len < 1e-9) return null;
      n = [ty / len, -tx / len, 0]; d = n[0] * l.x0 * bx + n[1] * l.y0 * by;
    }
    return c.flip ? { n: [-n[0], -n[1], -n[2]], d: -d } : { n, d };
  }

  /** Surface colour beyond the domain (the plain sea or land colour of the run). */
  setOutside(rgb: V3): void { this.outside = rgb; this.dirty = true; }

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

  /** Draw (when anything changed). `opacity`: the slider value 2-80; 80 is the physical extinction, each 30 below a
   *  tenth of it. */
  render(opacity: number): void {
    const gl = this.gl, c = this.canvas;
    const k = Math.pow(10, (Math.min(80, Math.max(2, opacity)) - 80) / 30);
    if (Math.abs(k - this.kExt) > 1e-9) { this.kExt = k; this.dirty = true; }
    if (this.lightDue && performance.now() >= this.lightDue) { this.lightDue = 0; if (this.data) this.computeLight(); this.dirty = true; }
    if (this.kLight !== this.kExt && this.data) this.computeLight();
    // keys: free flight moves (0.25 box widths per second, 4x with Shift); the orbit camera moves, turns, tilts, zooms
    const now = performance.now(), dts = Math.min(0.1, (now - (this.lastT || now)) / 1000); this.lastT = now;
    if (this.keys.size) {
      const kk = this.keys, fb = (kk.has('w') || kk.has('arrowup') ? 1 : 0) - (kk.has('s') || kk.has('arrowdown') ? 1 : 0);
      const lr = (kk.has('d') || kk.has('arrowright') ? 1 : 0) - (kk.has('a') || kk.has('arrowleft') ? 1 : 0), ud = (kk.has('e') ? 1 : 0) - (kk.has('q') ? 1 : 0);
      if (this.fly) {
        // speed grows with the height above the ground (slow near the surface, fast high up)
        const v = dts * (kk.has('shift') ? 4 : 1) * Math.max(0.03, Math.min(0.4, this.fly.eye[2] * 2));
        this.moveFly(fb * v, lr * v, ud * v);
      } else {
        const px = 600 * dts * (kk.has('shift') ? 3 : 1);
        if (fb || lr) this.pan(-lr * px, fb * px);
        if (ud) { this.yaw += ud * 1.2 * dts; this.dirty = true; }
        const tilt = (kk.has('r') ? 1 : 0) - (kk.has('f') ? 1 : 0);
        if (tilt) this.turn(0, tilt * 150 * dts);
        const z = (kk.has('-') ? 1 : 0) - (kk.has('+') || kk.has('=') ? 1 : 0);
        if (z) { const r = this.canvas.getBoundingClientRect(); this.zoomAt(r.left + r.width / 2, r.top + r.height / 2, Math.exp(z * 1.5 * dts)); }
      }
    }
    // cap the ray-marched pixel count (about 0.9 megapixels): the ray march is the costly part
    const cssW = Math.max(1, c.clientWidth), cssH = Math.max(1, c.clientHeight);
    const scale = Math.min(Math.min(1.5, window.devicePixelRatio || 1), Math.sqrt(9e5 / (cssW * cssH)));
    const w = Math.max(1, Math.round(cssW * scale)), h = Math.max(1, Math.round(cssH * scale));
    if (c.width !== w || c.height !== h) { c.width = w; c.height = h; this.dirty = true; }
    if (!this.dirty) return;
    this.dirty = false;
    gl.viewport(0, 0, w, h);
    gl.useProgram(this.prog);
    const [bx, by, bz] = this.box;
    const { eye, f, r, u: up } = this.basis();
    const vp = viewProj(eye, f, r, up, w / h, this.fov);
    const inv = invert4(vp);
    this.lastInv = inv; this.lastEye = eye; this.lastVP = vp;
    const trOn = this.tracerPass(vp, eye, w, h);
    gl.viewport(0, 0, w, h);
    gl.useProgram(this.prog);
    const u = (n: string): WebGLUniformLocation => gl.getUniformLocation(this.prog, n)!;
    gl.uniformMatrix4fv(u('uInvVP'), false, inv);
    gl.uniform3f(u('uEye'), eye[0], eye[1], eye[2]);
    gl.uniform3f(u('uBox'), bx, by, bz);
    gl.uniform3f(u('uSun'), SUN[0], SUN[1], SUN[2]);
    gl.uniform1f(u('uK'), this.kExt);
    gl.uniform1f(u('uMpb'), this.data ? this.data.top / bz : 1e5);
    gl.uniform1f(u('uCell'), this.data ? bx / this.data.nx : 0.01);
    gl.uniform1f(u('uCellZ'), this.data ? bz / this.data.nz : 0.01);
    gl.uniform3f(u('uOut'), this.outside[0], this.outside[1], this.outside[2]);
    const cp = this.cutPlane();
    gl.uniform1i(u('uCutOn'), cp ? 1 : 0);
    gl.uniform4f(u('uCut'), cp ? cp.n[0] : 0, cp ? cp.n[1] : 0, cp ? cp.n[2] : 1, cp ? cp.d : 1e9);
    gl.uniform1i(u('uMode'), this.mode);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_3D, this.vol); gl.uniform1i(u('uVol'), 0);
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, this.groundTex); gl.uniform1i(u('uGround'), 1);
    gl.activeTexture(gl.TEXTURE4); gl.bindTexture(gl.TEXTURE_3D, this.light); gl.uniform1i(u('uLight'), 4);
    const nt = this.nest;
    gl.uniform1i(u('uNOn'), nt ? 1 : 0);
    gl.activeTexture(gl.TEXTURE5); gl.bindTexture(gl.TEXTURE_3D, this.nVol); gl.uniform1i(u('uNVol'), 5);
    gl.activeTexture(gl.TEXTURE6); gl.bindTexture(gl.TEXTURE_3D, this.nLight); gl.uniform1i(u('uNLight'), 6);
    gl.uniform3f(u('uNBox'), nt ? nt.x0 : 0, nt ? nt.y0 : 0, nt ? nt.L : 1);
    gl.uniform4f(u('uNCyl'), nt ? nt.cx : 0, nt ? nt.cy : 0, nt ? nt.R : 0, nt ? Math.max(nt.Wf, 1e-6) : 1);
    gl.uniform1f(u('uNCell'), nt ? nt.L / nt.nx : 0.01);
    gl.uniform1f(u('uNCellZ'), nt ? bz / nt.nz : 0.01);
    gl.activeTexture(gl.TEXTURE7); gl.bindTexture(gl.TEXTURE_3D, this.occ); gl.uniform1i(u('uOcc'), 7);
    gl.uniform3f(u('uOccScale'), this.occScale[0], this.occScale[1], this.occScale[2]);
    gl.uniform1f(u('uOccStep'), this.occStep);
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
/** View-projection matrix (column-major) of an eye with forward f, right s and up u axes. */
function viewProj(eye: V3, f: V3, s: V3, u: V3, aspect: number, fov: number): Float32Array {
  const view = [s[0], u[0], -f[0], 0, s[1], u[1], -f[1], 0, s[2], u[2], -f[2], 0, -dot(s, eye), -dot(u, eye), dot(f, eye), 1];
  const n = 0.002, fa = 100, t = 1 / Math.tan(fov / 2);
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
