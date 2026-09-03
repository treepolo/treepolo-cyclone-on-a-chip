export function solveTridiagonal(lower:Float64Array,diag:Float64Array,upper:Float64Array,rhs:Float64Array,out:Float64Array):void {
  const n=diag.length; if(rhs.length!==n||out.length!==n||lower.length!==n||upper.length!==n) throw new Error('tridiagonal size mismatch');
  if(n===0) return;
  const cp=new Float64Array(n), dp=new Float64Array(n);
  let d=diag[0]!; if(Math.abs(d)<1e-30) throw new Error('singular tridiagonal'); cp[0]=upper[0]!/d; dp[0]=rhs[0]!/d;
  for(let i=1;i<n;i++){ d=diag[i]!-lower[i]!*cp[i-1]!; if(Math.abs(d)<1e-30) throw new Error('singular tridiagonal'); cp[i]=i===n-1?0:upper[i]!/d; dp[i]=(rhs[i]!-lower[i]!*dp[i-1]!)/d; }
  out[n-1]=dp[n-1]!; for(let i=n-2;i>=0;i--) out[i]=dp[i]!-cp[i]!*out[i+1]!;
}
