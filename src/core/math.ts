export type Vec3 = readonly [number, number, number];

export const add3 = (a: Vec3, b: Vec3): Vec3 => [a[0]+b[0], a[1]+b[1], a[2]+b[2]];
export const sub3 = (a: Vec3, b: Vec3): Vec3 => [a[0]-b[0], a[1]-b[1], a[2]-b[2]];
export const scale3 = (a: Vec3, s: number): Vec3 => [a[0]*s, a[1]*s, a[2]*s];
export const dot3 = (a: Vec3, b: Vec3): number => a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
export const cross3 = (a: Vec3, b: Vec3): Vec3 => [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
export const norm3 = (a: Vec3): number => Math.hypot(a[0],a[1],a[2]);
export const normalize3 = (a: Vec3): Vec3 => { const n=norm3(a); if (!(n>0)) throw new Error('zero vector'); return scale3(a,1/n); };
export const clamp = (x:number,a:number,b:number):number => Math.max(a,Math.min(b,x));
export const angle3 = (a:Vec3,b:Vec3):number => Math.acos(clamp(dot3(a,b),-1,1));

export function sphericalTriangleAreaUnit(a:Vec3,b:Vec3,c:Vec3):number {
  const numerator = Math.abs(dot3(a, cross3(b,c)));
  const denominator = 1 + dot3(a,b) + dot3(b,c) + dot3(c,a);
  return 2*Math.atan2(numerator, denominator);
}

export function mat4Perspective(fovy:number, aspect:number, near:number, far:number):Float32Array {
  const f=1/Math.tan(fovy/2), nf=1/(near-far);
  return new Float32Array([f/aspect,0,0,0, 0,f,0,0, 0,0,(far+near)*nf,-1, 0,0,2*far*near*nf,0]);
}
export function mat4LookAt(eye:Vec3, center:Vec3, up:Vec3):Float32Array {
  const z=normalize3(sub3(eye,center)); const x=normalize3(cross3(up,z)); const y=cross3(z,x);
  return new Float32Array([x[0],y[0],z[0],0, x[1],y[1],z[1],0, x[2],y[2],z[2],0, -dot3(x,eye),-dot3(y,eye),-dot3(z,eye),1]);
}
export function mat4Mul(a:Float32Array,b:Float32Array):Float32Array {
  const o=new Float32Array(16);
  for(let c=0;c<4;c++) for(let r=0;r<4;r++) {
    let s=0; for(let k=0;k<4;k++) s += a[k*4+r]! * b[c*4+k]!; o[c*4+r]=s;
  }
  return o;
}
