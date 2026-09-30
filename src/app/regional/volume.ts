// WebGL2 volume renderer for the regional model: front-to-back ray marching through a 3-D texture of display
// extinctions (R = cloud, G = precipitation, or updraft / vorticity; bytes (beta / EXT_MAX)^(1/3), display.ts), lit by
// sun and sky light that is precomputed on the CPU for every new volume (a sweep toward the sun and one straight up:
// no light marching per pixel), with forward scattering toward the sun (silver lining), cloud shadows on the ground,
// a coloured ground plane (surface field) and a map-style camera. The vertical scale is exaggerated; optical depths use
// the vertical metres per box unit in every direction, so what is drawn is what the displayed geometry would show.
// Tracer particles (with short trails) are drawn first into an offscreen buffer that keeps, per pixel, the
// nearest particle colour and its distance from the eye; the ray march composites that colour where the
// ray passes that distance, so clouds in front hide the particles behind them.
import { EXT_MAX } from '../../regional/display.js';

const VS = `#version 300 es
in vec2 aPos; out vec2 vUv;
void main(){ vUv = aPos*0.5+0.5; gl_Position = vec4(aPos,0.0,1.0); }`;

const FS = `#version 300 es
precision highp float; precision highp sampler3D;
in vec2 vUv; out vec4 o;
uniform sampler3D uVol;    // r cloud extinction, g precipitation (or updraft / vorticity), b vigour of the convection, a ice fraction
uniform sampler3D uLight;  // r transmittance to the sun, g transmittance straight up (precomputed per volume)
uniform sampler3D uNoise;  // r bubbles, g large lumpy pattern, b small lumpy pattern, a flat sheets (tileable)
uniform sampler2D uGround;
uniform mat4 uInvVP; uniform vec3 uEye; uniform vec3 uBox; uniform vec3 uSun;
uniform float uK;     // extinction multiplier (the opacity slider; 1 = physical)
uniform float uMpb;   // metres per box unit (vertical scale)
uniform float uCell;  // grid spacing in box units
uniform float uDx;    // grid spacing (m)
uniform float uPix;   // size of a screen pixel per unit distance from the eye
uniform int uDetail;  // 1: lumps of vigorous cloud, their surfaces found exactly and lit by the way they face
uniform vec2 uAnchor; // where the cloud detail is anchored (m; moves with the fields when they jump in the domain)
uniform float uRise;  // how far the lumps of vigorous cloud have risen (m)
uniform vec3 uOut;    // surface colour beyond the domain (sea or land)
uniform int uMode;    // channel 2: 0 precipitation, 1 updraft, 2 cyclonic vorticity
uniform vec4 uCut; uniform int uCutOn;  // cutaway: nothing where dot(uCut.xyz, p) > uCut.w
uniform sampler2D uTrC; uniform sampler2D uTrD; uniform int uTrOn;
const float EXTMAX = ${EXT_MAX.toFixed(3)};
bool hitBox(vec3 ro, vec3 rd, out float t0, out float t1){
  vec3 inv = 1.0/rd; vec3 a = (vec3(0.0)-ro)*inv; vec3 b = (uBox-ro)*inv;
  vec3 mn = min(a,b), mx = max(a,b);
  t0 = max(max(mn.x,mn.y),mn.z); t1 = min(min(mx.x,mx.y),mx.z);
  return t1 > max(t0,0.0);
}
float ext(float v){ return EXTMAX*v*v*v; }
float hash(vec2 p){ return fract(sin(dot(p, vec2(12.9898,78.233)))*43758.5453); }
// Cloud and precipitation extinction at p (box units) (dens below); pf is the size of a screen pixel there (box units).
// The grid box holds the model's mean extinction, and how vigorous the convection it belongs to is (convectionVigour,
// computed by the page from the model's vertical velocity and cloud). Weak, sinking or dissipating cloud is drawn as
// the part of the box that is cloudy: with an in-cloud extinction for the kind of cloud (liquid water 0.2 g/m3 in
// layer cloud, 0.6 g/m3 in convective cloud, ice 0.02 g/m3) the cloud fraction is mean / in-cloud, and a noise value
// (uniformly distributed) picks where that fraction is (cloud where the noise exceeds 1 - fraction), so the box mean
// stays the model's; the noise has real sizes whatever the grid, groups of clouds (2.5-30 km) plus single clouds
// (0.4-1.5 km), at unrelated tile sizes and angles so nothing repeats, and detail smaller than a pixel enters as its
// exact average. Vigorous cloud (a rising tower, liquid or glaciating) is drawn full: a solid body whose surface
// bulges into turrets and bubbles, found exactly and lit by the way it faces, so the sun lights the lumps on one side
// and they shade one another (cauliflower; there the box mean is not kept exactly). Ice that hardly rises away from
// updrafts (anvils, cirrus) spreads into flat sheets with soft edges. The model's field decides where and how much
// cloud there is.
const float B_LAYER = 0.03, B_CONV = 0.09, B_ICE = 0.0011;
// display value (cube root of extinction / EXTMAX) of the convective in-cloud extinction
const float S_CONV = ${Math.cbrt(0.09 / EXT_MAX).toFixed(4)};
// uniform value of a*x + (1 - a)*y for independent uniform x, y (the distribution function of the weighted sum)
float u2(float x, float y, float a){
  float b = 1.0 - a, lo = min(a, b), hi = max(a, b), s = a*x + b*y;
  if (lo < 1e-3) return s;
  return s < lo ? s*s/(2.0*a*b) : (s <= hi ? (s - 0.5*lo)/hi : 1.0 - (1.0 - s)*(1.0 - s)/(2.0*a*b));
}
// quantile of a*x + (1 - a)*y for independent uniform x, y (inverse of the distribution function of the weighted sum)
float invU2(float q, float a){
  float b = 1.0 - a, lo = min(a, b), hi = max(a, b);
  if (lo < 1e-3) return q;
  float q1 = lo/(2.0*hi);
  return q < q1 ? sqrt(2.0*a*b*q) : (q <= 1.0 - q1 ? q*hi + 0.5*lo : 1.0 - sqrt(2.0*a*b*(1.0 - q)));
}
// cloudy part (0..1) for small-scale noise x, large-scale y (weight a of x), clear fraction th, edge width e; wx, wy:
// how much of each scale a pixel resolves (unresolved noise enters as its average)
float shape(float x, float y, float a, float th, float e, float wx, float wy){
  float st = invU2(th, a);
  float full = smoothstep(st - e, st + e, a*x + (1.0 - a)*y);
  float noX = clamp((a + (1.0 - a)*y - st)/a, 0.0, 1.0);
  return mix(1.0 - th, mix(noX, full, wx), wy);
}
// c cloud and pr precipitation extinction (1/m), g the grid's cloud value, vig how vigorous (full-body) the cloud is,
// m how far inside the full body (box units; its gradient gives the way the surface faces)
struct Cl { float c; float pr; float g; float vig; float m; };
Cl dens(vec3 p, float pf, bool detail){
  Cl r = Cl(0.0, 0.0, 0.0, 0.0, -1.0);
  if (uCutOn == 1 && dot(uCut.xyz, p) > uCut.w) return r;
  vec4 s = textureLod(uVol, p/uBox, 0.0);
  float bp = uMode == 0 ? ext(s.g)*uK : s.g*s.g*4e-4;
  r.pr = bp; r.g = s.r;
  if (s.r < 0.004) return r;
  float bm = ext(s.r);
  // how vigorous the convection this cloud belongs to is (computed by the page for every box, see setVolume)
  float conv = s.b;
  float ice = s.a*(1.0 - conv);
  float bc = mix(mix(B_LAYER, B_CONV, conv), B_ICE, ice);
  float th = 1.0 - min(1.0, bm/bc);
  float vig = smoothstep(0.1, 0.5, conv);
  // metres (horizontal scale of the box; the view's heights are stretched alike, so bubbles look round)
  float mpb = uDx/uCell;
  // weak, short-lived or dissipating cloud: the cloudy part of the box, in pieces; ice that hardly rises: sheets
  float dl = 0.0, di = 0.0;
  // metres, anchored so the detail stays with the clouds from frame to frame
  vec3 q = p*mpb + vec3(uAnchor, 0.0);
  if (vig < 0.999 || ice > 0.001) {
    // groups: two tiles (40 km turned 27 degrees, 64.7 km turned -71 degrees) whose sum never repeats in a domain;
    // clouds: tile of 6 km turned -38 degrees
    vec4 A1 = texture(uNoise, vec3(0.891*q.x - 0.454*q.y, 0.454*q.x + 0.891*q.y, q.z)/40000.0 + vec3(0.37, 0.11, 0.0));
    vec4 A2 = texture(uNoise, vec3(0.326*q.x + 0.946*q.y, -0.946*q.x + 0.326*q.y, q.z)/64700.0 + vec3(0.71, 0.29, 0.5));
    vec4 B = texture(uNoise, vec3(0.788*q.x + 0.616*q.y, -0.616*q.x + 0.788*q.y, q.z)/6000.0);
    float Ag = u2(A1.g, A2.g, 0.5), Aa = u2(A1.a, A2.a, 0.5);
    // how much of each scale a pixel resolves (details need about two pixels)
    float wB = smoothstep(pf, 2.5*pf, 1500.0/mpb), wA = smoothstep(pf, 2.5*pf, 5000.0/mpb);
    dl = shape(B.b, Ag, 0.5, th, 0.07, wB, wA);
    di = shape(B.a, Aa, 0.5, th, 0.18, wB, wA);
  }
  // vigorous cloud: a full body (where the model's field exceeds the cloud-edge value, about 0.04 g/m3 of water) whose
  // surface bulges into lumps on lumps, like a fractal: bubble noise of lobes about 4, 2 and 1 km across on one tile and
  // 0.5, 0.25 and 0.13 km on another (unrelated sizes and angles), each size bulging in proportion to its size (so every
  // size shows alike in the light and shade); each tile shows once its middle size spans a couple of pixels, smaller
  // lobes give way to the smooth body. The bulges push
  // mostly outward and only next to the body, so it stays full and nothing floats free; they rise with the updraft from
  // frame to frame (uRise, in the view's stretched heights). m: how far inside (box units, from the field's typical
  // slope S_CONV per cell)
  float dv = 0.0;
  if (vig > 0.0) {
    float s0 = 0.4*S_CONV, disp = 0.0;
    if (detail) {
      vec3 qb = q - vec3(0.0, 0.0, uRise*mpb/uMpb);
      float T = texture(uNoise, vec3(0.891*qb.x - 0.454*qb.y, 0.454*qb.x + 0.891*qb.y, qb.z)/17000.0 + vec3(0.37, 0.11, 0.0)).r;
      float Bb = texture(uNoise, vec3(0.788*qb.x + 0.616*qb.y, -0.616*qb.x + 0.788*qb.y, qb.z)/2120.0).r;
      float aT = min(0.24, 0.5*(4250.0/uDx)*S_CONV)*smoothstep(pf, 2.5*pf, 4250.0/mpb);
      float aB = min(0.1, 0.5*(530.0/uDx)*S_CONV)*smoothstep(pf, 2.5*pf, 265.0/mpb);
      disp = (aT*(T - 0.3) + aB*(Bb - 0.3))*smoothstep(s0 - 0.6*(aT + aB), s0, s.r);
    }
    float m = (s.r - s0 + disp)*uCell/S_CONV;
    dv = smoothstep(-0.7*pf, 0.7*pf, m);
    r.m = m;
  }
  float d = mix(mix(dl, dv, vig), di, ice);
  // precipitation inside cloud (graupel in a tower, rain in a raining cloud) is in its cloudy part (same box mean); below
  // the cloud it falls as a smooth shaft
  float f = 1.0 - th;
  if (uMode == 0) bp *= mix(1.0, d/max(f, 1e-3), smoothstep(0.1, 0.4, f));
  r.c = max(bm, bc)*d*uK; r.pr = bp; r.vig = vig*(1.0 - ice);
  return r;
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
  // sea beyond it (clear air: only the far horizon blends into the sky)
  if (rd.z < 0.0) {
    float tg = -ro.z/rd.z; vec3 pg = ro + tg*rd;
    vec3 g;
    if (pg.x >= 0.0 && pg.y >= 0.0 && pg.x <= uBox.x && pg.y <= uBox.y) {
      vec2 L = texture(uLight, vec3(pg.xy/uBox.xy, 0.0)).rg;
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
    // coarse steps through clear air; entering cloud, one step back and fine steps (the billows) until the ray is
    // spent or out again (opaque cloud ends the ray within a few fine steps)
    // fine steps: a quarter of the largest bubble near the eye, about a pixel far away, at most half a grid cell
    float dtC = (t1 - t0)/120.0, lamB = 1500.0*uCell/uDx;
    float t = t0 + dtC*hash(gl_FragCoord.xy);
    float phase = 0.85 + 1.8*pow(max(mu,0.0), 6.0);
    int fine = 0, budget = 170;
    float prevC = 0.0;
    for (int i = 0; i < 320; i++) {
      if (t > t1 || trans < 0.004) break;
      if (!tdone && t > tdist) { col += trans*tc.rgb; trans *= 1.0 - tc.a; tdone = true; }
      vec3 p = ro + t*rd;
      float pf = t*uPix, dtF = min(dtC, min(0.5*uCell, max(0.8*pf, 0.35*lamB)));
      Cl d = dens(p, pf, fine > 0 && uDetail == 1);
      if (fine == 0 && d.g >= 0.004 && budget > 0 && dtF < dtC*0.99) { t = max(t0, t - dtC); fine = 24; prevC = 0.0; continue; }
      float dt = fine > 0 ? dtF : dtC;
      // sunlight and skylight factors of the surface of a full cloud body (1: no surface shading)
      float sunF = 1.0, skyF = 1.0;
      if (uDetail == 1 && fine > 0 && d.vig > 0.05 && d.c*dt*uMpb > 1.0 && prevC*dt*uMpb < 0.3) {
        // entering an opaque body: find its surface (bisection) and light it by the way it faces
        float ta = t - dt, tb = t;
        for (int k = 0; k < 4; k++) { float tm = 0.5*(ta + tb); if (dens(ro + tm*rd, pf, true).c*dt*uMpb > 1.0) tb = tm; else ta = tm; }
        t = tb; p = ro + t*rd; d = dens(p, pf, true);
        // the way the surface faces: down the slope of the body's margin (lobes included)
        float hn = max(0.05*lamB, pf);
        vec3 g = vec3(dens(p + vec3(hn, 0.0, 0.0), pf, true).m, dens(p + vec3(0.0, hn, 0.0), pf, true).m, dens(p + vec3(0.0, 0.0, hn), pf, true).m) - d.m;
        if (dot(g, g) > 1e-12) {
          vec3 n = -normalize(g);
          // light wraps well round a lobe (it scatters through the cloud), so the shaded side is soft; faces turned up
          // see more sky
          sunF = mix(1.0, clamp((dot(n, uSun) + 0.45)/1.45, 0.0, 1.0)*1.25, d.vig);
          skyF = mix(1.0, 0.5 + 0.5*clamp(0.5 + 0.5*n.z, 0.0, 1.0), d.vig);
        }
      }
      prevC = d.c;
      float b = d.c + d.pr;
      if (b > 1e-9) {
        vec2 L = texture(uLight, p/uBox).rg;
        // the cloud a short way toward the sun shades the bubbles (the precomputed light has no small-scale shape), with a
        // multiple-scattering tail (light diffusing into the shaded side of a bubble)
        float ls = 1.0;
        if (fine > 0 && d.c > 0.0 && trans > 0.02) {
          float h = min(0.5*uCell, max(0.6*lamB, 2.0*pf)), tl = dens(p + uSun*h, pf, false).c*h*uMpb;
          ls = (exp(-tl) + 0.5*exp(-0.25*tl))/1.5;
        }
        float a = 1.0 - exp(-b*dt*uMpb);
        vec3 cc = sunCol*(L.r*ls*phase*sunF + 0.25*sqrt(L.r)) + ambCol*(0.32 + 0.5*L.g)*skyF;
        vec3 cp = uMode == 0 ? vec3(0.46,0.54,0.66)*(0.35 + 0.65*L.r) + ambCol*0.1 : (uMode == 1 ? vec3(1.0,0.45,0.2) : vec3(0.35,0.55,1.0));
        vec3 c = (cc*d.c + cp*d.pr)/b;
        col += trans*a*c; trans *= 1.0-a;
      }
      if (fine > 0) { fine--; budget--; if (d.g >= 0.004) fine = max(fine, 8); }
      t += dt;
    }
  }
  if (!tdone) { col += trans*tc.rgb; trans *= 1.0 - tc.a; }
  col += trans*bg;
  // soft shoulder for the bright sunlit tops
  col = vec3(1.0) - exp(-col*1.3);
  o = vec4(col, 1.0);
}`;

/**
 * Tileable 3-D noise (n^3 RGBA bytes) for the cloud shapes, each channel equalised to a uniform distribution (so that
 * "noise above 1 - f" covers the fraction f): r bubbles (inverted Worley distance, 4, 8 and 16 cells per tile: round
 * lobes on lobes), g lumpy value noise with most variance at the largest scale (2 to 16 cells), b lumpy value noise
 * (4 to 16 cells), a flat sheets (value noise twice as fine vertically as horizontally).
 */
function cloudNoise(n: number): Uint8Array {
  const out = new Uint8Array(4 * n * n * n), N = n * n * n;
  const h = (a: number): number => { let x = Math.imul(a ^ 0x9e3779b9, 0x85ebca6b); x ^= x >>> 13; x = Math.imul(x, 0xc2b2ae35); x ^= x >>> 16; return (x >>> 0) / 4294967296; };
  const worley = (C: number): Float32Array => {
    const pts = new Float32Array(3 * C * C * C);
    for (let c = 0; c < C * C * C; c++) for (let d = 0; d < 3; d++) pts[3 * c + d] = h(c * 3 + d + C * 7919);
    const f = new Float32Array(N), s = C / n;
    for (let k = 0; k < n; k++) for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
      const x = (i + 0.5) * s, y = (j + 0.5) * s, z = (k + 0.5) * s, ci = Math.floor(x), cj = Math.floor(y), ck = Math.floor(z);
      let best = 9;
      for (let c = -1; c <= 1; c++) for (let b = -1; b <= 1; b++) for (let a = -1; a <= 1; a++) {
        const ii = ci + a, jj = cj + b, kk = ck + c, w = ((((kk % C) + C) % C) * C + (((jj % C) + C) % C)) * C + (((ii % C) + C) % C);
        const dx = ii + pts[3 * w]! - x, dy = jj + pts[3 * w + 1]! - y, dz = kk + pts[3 * w + 2]! - z, d = dx * dx + dy * dy + dz * dz;
        if (d < best) best = d;
      }
      f[(k * n + j) * n + i] = Math.max(0, 1 - Math.sqrt(best));
    }
    return f;
  };
  // value noise with P x P x Pz lattice cells per tile
  const value = (P: number, Pz: number, seed: number): Float32Array => {
    const f = new Float32Array(N), s = P / n, sz = Pz / n;
    const g = (a: number, b: number, c: number): number => h(((((c % Pz) + Pz) % Pz) * P + (((b % P) + P) % P)) * P + (((a % P) + P) % P) + seed);
    const sm = (t: number): number => t * t * (3 - 2 * t);
    for (let k = 0; k < n; k++) for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
      const x = i * s, y = j * s, z = k * sz, a = Math.floor(x), b = Math.floor(y), c = Math.floor(z), u = sm(x - a), v = sm(y - b), w = sm(z - c);
      const l = (cc: number): number => (1 - v) * ((1 - u) * g(a, b, cc) + u * g(a + 1, b, cc)) + v * ((1 - u) * g(a, b + 1, cc) + u * g(a + 1, b + 1, cc));
      f[(k * n + j) * n + i] = (1 - w) * l(c) + w * l(c + 1);
    }
    return f;
  };
  // rank of each value among all (histogram equalisation), as a byte
  const put = (ch: number, v: (q: number) => number): void => {
    const B = 4096, x = new Float32Array(N); let lo = Infinity, hi = -Infinity;
    for (let q = 0; q < N; q++) { const y = v(q); x[q] = y; if (y < lo) lo = y; if (y > hi) hi = y; }
    const cnt = new Float64Array(B + 1), sc = (B - 1) / ((hi - lo) || 1);
    for (let q = 0; q < N; q++) { const b = Math.floor((x[q]! - lo) * sc) + 1; cnt[b] = cnt[b]! + 1; }
    for (let b = 1; b <= B; b++) cnt[b] = cnt[b]! + cnt[b - 1]!;
    for (let q = 0; q < N; q++) { const b = Math.floor((x[q]! - lo) * sc); out[4 * q + ch] = Math.round(255 * (cnt[b]! + 0.5 * (cnt[b + 1]! - cnt[b]!)) / N); }
  };
  const w4 = worley(4), w8 = worley(8), w16 = worley(16);
  put(0, (q) => w4[q]! + 0.5 * w8[q]! + 0.25 * w16[q]!);
  // large lumpy pattern: most of its variance in the largest features (clusters and gaps)
  const g2 = value(2, 2, 5), g4 = value(4, 4, 7), g8 = value(8, 8, 13), g16 = value(16, 16, 17);
  put(1, (q) => g2[q]! + 0.6 * g4[q]! + 0.36 * g8[q]! + 0.22 * g16[q]!);
  const v4 = value(4, 4, 11), v8 = value(8, 8, 23), v16 = value(16, 16, 37);
  put(2, (q) => v4[q]! + 0.5 * v8[q]! + 0.25 * v16[q]!);
  const s4 = value(4, 8, 41), s8 = value(8, 16, 53), s16 = value(16, 32, 67);
  put(3, (q) => s4[q]! + 0.5 * s8[q]! + 0.25 * s16[q]!);
  return out;
}

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

/**
 * How vigorous the convection each box's cloud belongs to is (0..1 as a byte), which decides how the 3-D view draws it
 * (full and bulging, or in pieces). From the display bytes: vertical velocity (wByte) and cloud (extByte) per box, on
 * a grid of spacing dx (m). A box counts as vigorous when the air there or in the boxes around rises (the strength
 * that counts is scale-aware: a plume fills less of a larger box, 1.5 m/s at 3 km or finer, 0.3 m/s at 15 km), when a
 * strong updraft rises within about 4 km (a tower's sides and overshooting top, where the air itself hardly rises),
 * or when the boxes around are mostly liquid cloud that is not sinking. Sinking, thin cloud away from updrafts is
 * not (dissipating: drawn in pieces).
 */
export function convectionVigour(nx: number, ny: number, nz: number, cloud: Uint8Array, aux: Uint8Array, dx: number): Uint8Array {
  const n = nx * ny * nz, np = nx * ny, wc = 1.5 * Math.min(1, 3000 / Math.max(dx, 1));
  const W = new Float32Array(256);
  for (let b = 0; b < 256; b++) { const x = (b - 128) / 127; W[b] = Math.sign(x) * x * x * 40; }
  const ss = (a: number, b: number, x: number): number => { const t = x <= a ? 0 : x >= b ? 1 : (x - a) / (b - a); return t * t * (3 - 2 * t); };
  // vertical velocity and cloud (0..1), then their means over the 3 x 3 x 3 boxes around (separable, fewer at the edges)
  const w = new Float32Array(n), c = new Float32Array(n), wa = new Float32Array(n), ca = new Float32Array(n);
  for (let q = 0; q < n; q++) { w[q] = W[aux[2 * q]!]!; c[q] = cloud[q]! / 255; }
  for (let r = 0; r < nz * ny; r++) {
    const o = r * nx;
    for (let i = 0; i < nx; i++) {
      const q = o + i, lo = i > 0 ? q - 1 : q, hi = i < nx - 1 ? q + 1 : q, cn = 1 + (i > 0 ? 1 : 0) + (i < nx - 1 ? 1 : 0);
      wa[q] = (w[q]! + (lo !== q ? w[lo]! : 0) + (hi !== q ? w[hi]! : 0)) / cn;
      ca[q] = (c[q]! + (lo !== q ? c[lo]! : 0) + (hi !== q ? c[hi]! : 0)) / cn;
    }
  }
  for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) {
    const o = (k * ny + j) * nx, dn = j > 0 ? -nx : 0, up = j < ny - 1 ? nx : 0, cn = 1 + (dn ? 1 : 0) + (up ? 1 : 0);
    for (let i = 0; i < nx; i++) {
      const q = o + i;
      w[q] = (wa[q]! + (dn ? wa[q + dn]! : 0) + (up ? wa[q + up]! : 0)) / cn;
      c[q] = (ca[q]! + (dn ? ca[q + dn]! : 0) + (up ? ca[q + up]! : 0)) / cn;
    }
  }
  for (let k = 0; k < nz; k++) {
    const o = k * np, dn = k > 0 ? -np : 0, up = k < nz - 1 ? np : 0, cn = 1 + (dn ? 1 : 0) + (up ? 1 : 0);
    for (let q2 = 0; q2 < np; q2++) {
      const q = o + q2;
      wa[q] = (w[q]! + (dn ? w[q + dn]! : 0) + (up ? w[q + up]! : 0)) / cn;
      ca[q] = (c[q]! + (dn ? c[q + dn]! : 0) + (up ? c[q + up]! : 0)) / cn;
    }
  }
  // the strongest updraft of each column, then the largest within about 4 km (a square of that half width)
  const col = new Float32Array(np).fill(-99), tmp = new Float32Array(np), wCol = new Float32Array(np);
  for (let q = 0; q < n; q++) { const v = W[aux[2 * q]!]!, q2 = q % np; if (v > col[q2]!) col[q2] = v; }
  const R = Math.max(1, Math.round(4000 / Math.max(dx, 1)));
  for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
    let m = -99; for (let a = Math.max(0, i - R); a <= Math.min(nx - 1, i + R); a++) m = Math.max(m, col[j * nx + a]!);
    tmp[j * nx + i] = m;
  }
  for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
    let m = -99; for (let b = Math.max(0, j - R); b <= Math.min(ny - 1, j + R); b++) m = Math.max(m, tmp[b * nx + i]!);
    wCol[j * nx + i] = m;
  }
  const out = new Uint8Array(n);
  for (let q = 0; q < n; q++) {
    const cN = ca[q]!;
    if (cloud[q] === 0 && cN === 0) continue;
    const wN = wa[q]!, wl = W[aux[2 * q]!]!;
    const v = Math.max(ss(0.15 * wc, 2 * wc, wl > wN ? wl : wN), ss(0.5 * wc, 3 * wc, wCol[q % np]!), ss(0.35, 0.6, cN) * ss(-wc, 0, wN) * (1 - aux[2 * q + 1]! / 255));
    out[q] = Math.round(255 * v);
  }
  return out;
}

/**
 * Cloud detail that carries on from frame to frame: the vigour of each box remembers the last frames (a tower that stops
 * rising stays full for a while and then breaks up, e-folding 15 min; the memory follows the fields when they jump in
 * the domain), and the lumps of vigorous cloud rise with the updraft (half the mean vertical velocity of the vigorous
 * cloudy boxes, at most 8 m/s), so a turret seen in one frame is higher up in the next. anchor: the frame's anchor
 * (m; the fields keep their place relative to it).
 */
export class CloudDetail {
  private prev: { nx: number; ny: number; nz: number; dx: number; t: number; ax: number; ay: number; vig: Uint8Array } | null = null;
  /** how far (m) the lumps of vigorous cloud have risen since the run began */
  rise = 0;
  reset(): void { this.prev = null; this.rise = 0; }
  /** Display bytes for the renderer: per box [vigour, ice fraction]; aux holds [w byte, ice fraction]. */
  update(t: number, nx: number, ny: number, nz: number, dx: number, cloud: Uint8Array, aux: Uint8Array, anchor: { x: number; y: number }): Uint8Array {
    const vig = convectionVigour(nx, ny, nz, cloud, aux, dx), n = nx * ny * nz, np = nx * ny, p = this.prev;
    if (p && t < p.t - 1e-6) this.reset();
    const q = this.prev;
    if (q && t > q.t) {
      const dt = t - q.t;
      // the lumps rise with the vigorous clouds' updraft
      let sw = 0, cn = 0;
      for (let c = 0; c < n; c++) if (cloud[c]! > 0 && vig[c]! > 128) { const x = (aux[2 * c]! - 128) / 127; sw += Math.sign(x) * x * x * 40; cn++; }
      this.rise += Math.max(0, Math.min(8, cn ? 0.5 * sw / cn : 0)) * dt;
      // the last frames' vigour, moved with the fields, fading
      if (q.nx === nx && q.ny === ny && q.nz === nz && Math.abs(q.dx - dx) < 1e-6 && dt < 3 * 3600) {
        const f = Math.exp(-dt / 900), di = Math.round((anchor.x - q.ax) / dx), dj = Math.round((anchor.y - q.ay) / dx);
        for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) {
          const jj = j + dj; if (jj < 0 || jj >= ny) continue;
          for (let i = 0; i < nx; i++) {
            const ii = i + di; if (ii < 0 || ii >= nx) continue;
            const o = k * np + j * nx + i, v = Math.round(q.vig[k * np + jj * nx + ii]! * f);
            if (v > vig[o]!) vig[o] = v;
          }
        }
      }
    }
    if (!q || t !== q.t) this.prev = { nx, ny, nz, dx, t, ax: anchor.x, ay: anchor.y, vig: Uint8Array.from(vig) };
    const out = new Uint8Array(2 * n);
    for (let c = 0; c < n; c++) { out[2 * c] = vig[c]!; out[2 * c + 1] = aux[2 * c + 1]!; }
    return out;
  }
}

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
  private readonly noise: WebGLTexture;
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
  private data: { nx: number; ny: number; nz: number; cloud: Uint8Array; top: number; dx: number } | null = null;
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
    this.noise = gl.createTexture()!;
    this.groundTex = gl.createTexture()!;
    // the cloud texture's noise (once; tiles, repeats)
    gl.bindTexture(gl.TEXTURE_3D, this.noise);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texImage3D(gl.TEXTURE_3D, 0, gl.RGBA8, 64, 64, 64, 0, gl.RGBA, gl.UNSIGNED_BYTE, cloudNoise(64));
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    for (const w of [gl.TEXTURE_WRAP_S, gl.TEXTURE_WRAP_T, gl.TEXTURE_WRAP_R]) gl.texParameteri(gl.TEXTURE_3D, w, gl.REPEAT);
    // empty volume and light until the first frame (the sampler types must be complete)
    gl.bindTexture(gl.TEXTURE_3D, this.vol);
    gl.texImage3D(gl.TEXTURE_3D, 0, gl.RGBA8, 1, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([0, 0, 128, 0]));
    gl.bindTexture(gl.TEXTURE_3D, this.light);
    gl.texImage3D(gl.TEXTURE_3D, 0, gl.RG8, 1, 1, 1, 0, gl.RG, gl.UNSIGNED_BYTE, new Uint8Array([255, 255]));
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

  /** Upload the display volume [k][j][i] (bytes, display.ts; aux: vigour and ice-fraction bytes per cell, CloudDetail)
   *  of a domain `top` metres high with grid spacing dx (m), set the box aspect (x, y normalised to 1) and light it;
   *  anchor and rise place the cloud detail (CloudDetail). */
  setVolume(nx: number, ny: number, nz: number, cloud: Uint8Array, rain: Uint8Array, aux: Uint8Array | null, aspectZ: number, top: number, dx: number,
    anchor: { x: number; y: number } = { x: 0, y: 0 }, rise = 0): void {
    const gl = this.gl, rg = new Uint8Array(nx * ny * nz * 4);
    for (let i = 0; i < cloud.length; i++) { rg[4 * i] = cloud[i]!; rg[4 * i + 1] = rain[i]!; rg[4 * i + 2] = aux ? aux[2 * i]! : 0; rg[4 * i + 3] = aux ? aux[2 * i + 1]! : 0; }
    this.anchor = anchor; this.rise = rise;
    gl.bindTexture(gl.TEXTURE_3D, this.vol);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texImage3D(gl.TEXTURE_3D, 0, gl.RGBA8, nx, ny, nz, 0, gl.RGBA, gl.UNSIGNED_BYTE, rg);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    for (const w of [gl.TEXTURE_WRAP_S, gl.TEXTURE_WRAP_T, gl.TEXTURE_WRAP_R]) gl.texParameteri(gl.TEXTURE_3D, w, gl.CLAMP_TO_EDGE);
    const ax = 1, ay = ny / nx;
    // a new domain shape: back to the domain centre (a new vertical exaggeration keeps the view)
    if (ax !== this.box[0] || ay !== this.box[1]) this.tgt = null;
    this.box = [ax, ay, aspectZ];
    this.data = { nx, ny, nz, cloud, top, dx };
    this.computeLight();
    this.dirty = true;
  }

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
    const sky = new Float32Array(np), out = new Uint8Array(2 * nz * np);
    const put = (kk: number): void => { const o = kk * np; for (let c = 0; c < np; c++) { out[2 * (o + c)] = Math.round(255 * Math.exp(-od[c]!)); out[2 * (o + c) + 1] = Math.round(255 * Math.exp(-sky[c]!)); } };
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
    gl.bindTexture(gl.TEXTURE_3D, this.light);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texImage3D(gl.TEXTURE_3D, 0, gl.RG8, nx, ny, nz, 0, gl.RG, gl.UNSIGNED_BYTE, out);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    for (const w of [gl.TEXTURE_WRAP_S, gl.TEXTURE_WRAP_T, gl.TEXTURE_WRAP_R]) gl.texParameteri(gl.TEXTURE_3D, w, gl.CLAMP_TO_EDGE);
    this.kLight = k;
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
  private anchor = { x: 0, y: 0 };
  private rise = 0;
  private detail = true;
  /** Lumps of vigorous cloud and their lighting (off: a lighter picture for slower devices). */
  setDetail(on: boolean): void { this.detail = on; this.dirty = true; }
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
    gl.uniform1f(u('uDx'), this.data ? this.data.dx : 3000);
    gl.uniform1f(u('uPix'), 2 * Math.tan(this.fov / 2) / h);
    gl.uniform1i(u('uDetail'), this.detail ? 1 : 0);
    gl.uniform2f(u('uAnchor'), this.anchor.x % 1e6, this.anchor.y % 1e6);
    gl.uniform1f(u('uRise'), this.rise % 1e6);
    gl.uniform3f(u('uOut'), this.outside[0], this.outside[1], this.outside[2]);
    const cp = this.cutPlane();
    gl.uniform1i(u('uCutOn'), cp ? 1 : 0);
    gl.uniform4f(u('uCut'), cp ? cp.n[0] : 0, cp ? cp.n[1] : 0, cp ? cp.n[2] : 1, cp ? cp.d : 1e9);
    gl.uniform1i(u('uMode'), this.mode);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_3D, this.vol); gl.uniform1i(u('uVol'), 0);
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, this.groundTex); gl.uniform1i(u('uGround'), 1);
    gl.activeTexture(gl.TEXTURE4); gl.bindTexture(gl.TEXTURE_3D, this.light); gl.uniform1i(u('uLight'), 4);
    gl.activeTexture(gl.TEXTURE5); gl.bindTexture(gl.TEXTURE_3D, this.noise); gl.uniform1i(u('uNoise'), 5);
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
