export function assert(cond:unknown,msg:string):asserts cond { if(!cond) throw new Error(msg); }
export function near(a:number,b:number,tol:number,msg:string):void { if(Math.abs(a-b)>tol) throw new Error(`${msg}: got ${a}, expected ${b}, |err|=${Math.abs(a-b)} > ${tol}`); }
export function relative(a:number,b:number,tol:number,msg:string):void { const d=Math.abs(a-b)/Math.max(Math.abs(b),1e-300); if(d>tol)throw new Error(`${msg}: rel=${d} > ${tol}`); }
