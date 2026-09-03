export interface Rk3SplitStage {
  /** Large-step target time as a fraction of dt: 1/3, 1/2, 1. */
  targetFraction:number;
  /** Number of acoustic small steps used in this RK stage. */
  acousticSteps:number;
  /** Each acoustic small-step size as a fraction of the large dt. */
  acousticDtFraction:number;
}

/**
 * Wicker-Skamarock / ARW-style three-stage schedule.
 *
 * For ns=4:
 *   stage 1 -> target dt/3 using one modified acoustic step dt/3
 *   stage 2 -> target dt/2 using 2 steps of dt/4
 *   stage 3 -> target dt   using 4 steps of dt/4
 *
 * ns must be even so stage 2 can use ns/2 small steps. The first stage is
 * intentionally a single dt/3 acoustic step, independent of ns.
 */
export function buildRk3SplitSchedule(ns=4):readonly Rk3SplitStage[] {
  if(!Number.isInteger(ns)||ns<2||ns%2!==0)throw new Error('RK3 acoustic step ratio ns must be an even integer >= 2');
  return [
    {targetFraction:1/3,acousticSteps:1,acousticDtFraction:1/3},
    {targetFraction:1/2,acousticSteps:ns/2,acousticDtFraction:1/ns},
    {targetFraction:1,acousticSteps:ns,acousticDtFraction:1/ns},
  ] as const;
}

/**
 * Pure large-step RK3 predictor algebra for a scalar ODE x'=R(x).
 * Every predictor is formed from the SAME large-step base x^t.
 * This helper deliberately contains no acoustic integration; it exists so the
 * restart semantics can be regression-tested independently of the atmosphere.
 */
export function wickerSkamarockRk3Scalar(x0:number,dt:number,rhs:(x:number)=>number):number {
  if(!(dt>0))throw new Error('RK3 dt must be positive');
  const x1=x0+(dt/3)*rhs(x0);
  const x2=x0+(dt/2)*rhs(x1);
  return x0+dt*rhs(x2);
}

/** Form one RK predictor from the immutable large-step base. */
export function rkPredictorFromBase(base:number,dt:number,targetFraction:number,rhsAtPredictor:number):number {
  if(!(dt>0))throw new Error('RK3 dt must be positive');
  if(!(targetFraction>0&&targetFraction<=1))throw new Error('RK3 target fraction must be in (0,1]');
  return base+targetFraction*dt*rhsAtPredictor;
}
