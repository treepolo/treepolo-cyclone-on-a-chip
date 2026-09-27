// WebGL2 globe: colour-shaded model field on a sphere mesh whose rows are the model's
// Gaussian latitudes, plus wind tracer streaks and a graticule. Orbit camera.

export type Rgb = [number, number, number];

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

function compile(gl: WebGL2RenderingContext, type: number, src: string): WebGLShader {
  const s = gl.createShader(type)!;
  gl.shaderSource(s, src); gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s) ?? 'shader');
  return s;
}

export class Globe {
  private readonly gl: WebGL2RenderingContext;
  private readonly prog: WebGLProgram;
  private readonly loc: { pos: number; col: number; mvp: WebGLUniformLocation; scale: WebGLUniformLocation; light: WebGLUniformLocation; alpha: WebGLUniformLocation; shade: WebGLUniformLocation };
  private mesh: { vao: WebGLVertexArrayObject; col: WebGLBuffer; count: number; rows: number; cols: number; lat: Float64Array } | null = null;
  private readonly grat: { vao: WebGLVertexArrayObject; count: number };
  private readonly trc: { vao: WebGLVertexArrayObject; pos: WebGLBuffer; col: WebGLBuffer };
  private trcCount = 0;
  private coast: { vao: WebGLVertexArrayObject; count: number } | null = null;
  private marker: { vao: WebGLVertexArrayObject; count: number } | null = null;
  /** called on a click (not a drag) on the sphere with (lat, lon) in radians, lon in [0, 2 pi) */
  onPick: ((lat: number, lon: number) => void) | null = null;
  yaw = -0.4;
  pitch = 0.35;
  dist = 3.2;

  constructor(private readonly canvas: HTMLCanvasElement) {
    const gl = canvas.getContext('webgl2', { antialias: true });
    if (!gl) throw new Error('WebGL2 不可用 / WebGL2 unavailable');
    this.gl = gl;
    const p = gl.createProgram()!;
    gl.attachShader(p, compile(gl, gl.VERTEX_SHADER, VS));
    gl.attachShader(p, compile(gl, gl.FRAGMENT_SHADER, FS));
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p) ?? 'link');
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
    this.attachControls();
  }

  private attachControls(): void {
    let drag = false, lx = 0, ly = 0, moved = 0;
    const c = this.canvas;
    c.addEventListener('pointerdown', (e) => { drag = true; moved = 0; lx = e.clientX; ly = e.clientY; c.setPointerCapture(e.pointerId); });
    c.addEventListener('pointerup', (e) => {
      drag = false;
      if (moved < 5 && this.onPick) { const p = this.pick(e.clientX, e.clientY); if (p) this.onPick(p.lat, p.lon); }
    });
    c.addEventListener('pointermove', (e) => {
      if (!drag) return;
      moved += Math.abs(e.clientX - lx) + Math.abs(e.clientY - ly);
      this.yaw -= (e.clientX - lx) * 0.006;
      this.pitch = Math.max(-1.5, Math.min(1.5, this.pitch + (e.clientY - ly) * 0.006));
      lx = e.clientX; ly = e.clientY;
    });
    c.addEventListener('wheel', (e) => { e.preventDefault(); this.dist = Math.max(1.4, Math.min(8, this.dist * Math.exp(e.deltaY * 0.001))); }, { passive: false });
  }

  /** Screen point -> (lat, lon) on the unit sphere, or null when the ray misses it. */
  pick(clientX: number, clientY: number): { lat: number; lon: number } | null {
    const r = this.canvas.getBoundingClientRect();
    const x = (clientX - r.left) / r.width * 2 - 1, y = 1 - (clientY - r.top) / r.height * 2;
    const e = this.eye(), f = norm([-e[0], -e[1], -e[2]]), s = norm(cross(f, [0, 1, 0])), u = cross(s, f);
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
  setMarker(lat: number | null, lon = 0, half = 0.1): void {
    const gl = this.gl;
    if (lat === null) { this.marker = null; return; }
    const v: number[] = [], c: number[] = [];
    const pt = (x: number, y: number): void => {
      const la = lat + y, lo = lon + x / Math.max(0.05, Math.cos(lat));
      v.push(Math.cos(la) * Math.cos(lo), Math.sin(la), -Math.cos(la) * Math.sin(lo)); c.push(1, 0.85, 0.2);
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
    this.marker = { vao, count: v.length / 3 };
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

  /** (Re)build the sphere mesh for a Gaussian grid: rows = north pole, lat[0..nlat-1], south pole. */
  private ensureMesh(lat: Float64Array, nlon: number): void {
    if (this.mesh && this.mesh.cols === nlon + 1 && this.mesh.lat.length === lat.length) return;
    const gl = this.gl, rows = lat.length + 2, cols = nlon + 1;
    const pos = new Float32Array(rows * cols * 3);
    for (let r = 0; r < rows; r++) {
      const la = r === 0 ? Math.PI / 2 : r === rows - 1 ? -Math.PI / 2 : lat[r - 1]!;
      for (let c = 0; c < cols; c++) {
        const lo = c * 2 * Math.PI / nlon, o = (r * cols + c) * 3;
        pos[o] = Math.cos(la) * Math.cos(lo); pos[o + 1] = Math.sin(la); pos[o + 2] = -Math.cos(la) * Math.sin(lo);
      }
    }
    const idx = new Uint32Array((rows - 1) * (cols - 1) * 6);
    let n = 0;
    for (let r = 0; r < rows - 1; r++) for (let c = 0; c < cols - 1; c++) {
      const a = r * cols + c, b = a + 1, d = a + cols, e = d + 1;
      idx[n++] = a; idx[n++] = d; idx[n++] = b; idx[n++] = b; idx[n++] = d; idx[n++] = e;
    }
    const vao = gl.createVertexArray()!;
    gl.bindVertexArray(vao);
    const pb = gl.createBuffer()!; gl.bindBuffer(gl.ARRAY_BUFFER, pb); gl.bufferData(gl.ARRAY_BUFFER, pos, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(this.loc.pos); gl.vertexAttribPointer(this.loc.pos, 3, gl.FLOAT, false, 0, 0);
    const cb = gl.createBuffer()!; gl.bindBuffer(gl.ARRAY_BUFFER, cb); gl.bufferData(gl.ARRAY_BUFFER, rows * cols * 12, gl.DYNAMIC_DRAW);
    gl.enableVertexAttribArray(this.loc.col); gl.vertexAttribPointer(this.loc.col, 3, gl.FLOAT, false, 0, 0);
    const ib = gl.createBuffer()!; gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ib); gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, idx, gl.STATIC_DRAW);
    gl.bindVertexArray(null);
    this.mesh = { vao, col: cb, count: n, rows, cols, lat: Float64Array.from(lat) };
  }

  /** Colour the sphere from a [lat][lon] field using colour function f. */
  /** Colour the sphere from a [lat][lon] field; f receives the value and its grid index (-1 at the poles). */
  setField(lat: Float64Array, nlon: number, values: Float32Array, f: (v: number, idx: number) => Rgb): void {
    this.ensureMesh(lat, nlon);
    const m = this.mesh!, nlat = lat.length, col = new Float32Array(m.rows * m.cols * 3);
    const put = (r: number, c: number, rgb: Rgb): void => { const o = (r * m.cols + c) * 3; col[o] = rgb[0]; col[o + 1] = rgb[1]; col[o + 2] = rgb[2]; };
    let n = 0, s = 0;
    for (let i = 0; i < nlon; i++) n += values[i]!;
    for (let i = 0; i < nlon; i++) s += values[(nlat - 1) * nlon + i]!;
    const np = f(n / nlon, -1), sp = f(s / nlon, -1);
    for (let c = 0; c < m.cols; c++) { put(0, c, np); put(m.rows - 1, c, sp); }
    for (let j = 0; j < nlat; j++) for (let c = 0; c < m.cols; c++) put(j + 1, c, f(values[j * nlon + (c % nlon)]!, j * nlon + (c % nlon)));
    const gl = this.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, m.col);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, col);
  }

  /** Coastlines: 0.5 contour of a [lat][lon] land mask (marching squares), or null to clear. */
  setOutline(lat: Float64Array, nlon: number, mask: Uint8Array | null): void {
    if (!mask) { this.coast = null; return; }
    const gl = this.gl, nlat = lat.length, v: number[] = [], c: number[] = [];
    const P = (la: number, lo: number): void => { v.push(Math.cos(la) * Math.cos(lo), Math.sin(la), -Math.cos(la) * Math.sin(lo)); c.push(0.05, 0.05, 0.05); };
    for (let j = 0; j < nlat - 1; j++) for (let i = 0; i < nlon; i++) {
      const i1 = (i + 1) % nlon;
      const lo0 = i * 2 * Math.PI / nlon, lo1 = (i + 1) * 2 * Math.PI / nlon, la0 = lat[j]!, la1 = lat[j + 1]!;
      const corners: [number, number, number][] = [
        [la0, lo0, mask[j * nlon + i]!], [la0, lo1, mask[j * nlon + i1]!], [la1, lo1, mask[(j + 1) * nlon + i1]!], [la1, lo0, mask[(j + 1) * nlon + i]!],
      ];
      const pts: [number, number][] = [];
      for (let e = 0; e < 4; e++) {
        const a = corners[e]!, b = corners[(e + 1) % 4]!;
        if (a[2] !== b[2]) pts.push([(a[0] + b[0]) / 2, (a[1] + b[1]) / 2]);
      }
      if (pts.length >= 2) { P(pts[0]![0], pts[0]![1]); P(pts[1]![0], pts[1]![1]); }
      if (pts.length === 4) { P(pts[2]![0], pts[2]![1]); P(pts[3]![0], pts[3]![1]); }
    }
    const vao = gl.createVertexArray()!;
    gl.bindVertexArray(vao);
    const pb = gl.createBuffer()!; gl.bindBuffer(gl.ARRAY_BUFFER, pb); gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(v), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(this.loc.pos); gl.vertexAttribPointer(this.loc.pos, 3, gl.FLOAT, false, 0, 0);
    const cb = gl.createBuffer()!; gl.bindBuffer(gl.ARRAY_BUFFER, cb); gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(c), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(this.loc.col); gl.vertexAttribPointer(this.loc.col, 3, gl.FLOAT, false, 0, 0);
    gl.bindVertexArray(null);
    this.coast = { vao, count: v.length / 3 };
  }

  /** Tracer streaks: pairs of unit vectors (head, tail) with colours. */
  setTracers(pos: Float32Array, col: Float32Array, count: number): void {
    const gl = this.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.trc.pos); gl.bufferData(gl.ARRAY_BUFFER, pos.subarray(0, count * 3), gl.DYNAMIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.trc.col); gl.bufferData(gl.ARRAY_BUFFER, col.subarray(0, count * 3), gl.DYNAMIC_DRAW);
    this.trcCount = count;
  }

  render(): void {
    const gl = this.gl, c = this.canvas;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const w = Math.max(1, Math.round(c.clientWidth * dpr)), h = Math.max(1, Math.round(c.clientHeight * dpr));
    if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
    gl.viewport(0, 0, w, h);
    gl.clearColor(0.02, 0.03, 0.05, 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.enable(gl.DEPTH_TEST);
    gl.useProgram(this.prog);
    const mvp = this.matrix(w / h);
    gl.uniformMatrix4fv(this.loc.mvp, false, mvp);
    const eye = this.eye();
    const ln = Math.hypot(eye[0] + 0.6, eye[1] + 0.8, eye[2]);
    gl.uniform3f(this.loc.light, (eye[0] + 0.6) / ln, (eye[1] + 0.8) / ln, eye[2] / ln);
    if (this.mesh) {
      gl.uniform1f(this.loc.scale, 1); gl.uniform1f(this.loc.alpha, 1); gl.uniform1f(this.loc.shade, 1);
      gl.bindVertexArray(this.mesh.vao);
      gl.drawElements(gl.TRIANGLES, this.mesh.count, gl.UNSIGNED_INT, 0);
    }
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.uniform1f(this.loc.shade, 0);
    gl.uniform1f(this.loc.scale, 1.002); gl.uniform1f(this.loc.alpha, 0.18);
    gl.bindVertexArray(this.grat.vao);
    gl.drawArrays(gl.LINES, 0, this.grat.count);
    if (this.coast) {
      gl.uniform1f(this.loc.scale, 1.003); gl.uniform1f(this.loc.alpha, 0.9);
      gl.bindVertexArray(this.coast.vao);
      gl.drawArrays(gl.LINES, 0, this.coast.count);
    }
    if (this.marker) {
      gl.uniform1f(this.loc.scale, 1.005); gl.uniform1f(this.loc.alpha, 1);
      gl.bindVertexArray(this.marker.vao);
      gl.drawArrays(gl.LINES, 0, this.marker.count);
    }
    if (this.trcCount > 0) {
      gl.uniform1f(this.loc.scale, 1.004); gl.uniform1f(this.loc.alpha, 0.85);
      gl.bindVertexArray(this.trc.vao);
      gl.drawArrays(gl.LINES, 0, this.trcCount);
    }
    gl.disable(gl.BLEND);
    gl.bindVertexArray(null);
  }

  private eye(): [number, number, number] {
    const d = this.dist;
    return [d * Math.cos(this.pitch) * Math.sin(this.yaw), d * Math.sin(this.pitch), d * Math.cos(this.pitch) * Math.cos(this.yaw)];
  }

  private matrix(aspect: number): Float32Array {
    const e = this.eye();
    // view (lookAt origin, up = +y)
    const f = norm([-e[0], -e[1], -e[2]]);
    const s = norm(cross(f, [0, 1, 0]));
    const u = cross(s, f);
    const view = [
      s[0], u[0], -f[0], 0,
      s[1], u[1], -f[1], 0,
      s[2], u[2], -f[2], 0,
      -dot(s, e), -dot(u, e), dot(f, e), 1,
    ];
    const fov = 0.8, near = 0.05, far = 50, t = 1 / Math.tan(fov / 2);
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
function cross(a: V3, b: V3): V3 { return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]; }
function dot(a: V3, b: V3): number { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
function norm(a: V3): V3 { const l = Math.hypot(a[0], a[1], a[2]); return [a[0] / l, a[1] / l, a[2] / l]; }
