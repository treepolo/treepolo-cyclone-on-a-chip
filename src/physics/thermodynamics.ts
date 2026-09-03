import { AtmosphereConfig, DRY_AIR } from '../core/constants.js';

export function pressureFromRhoTheta(rhoTheta:number, a:AtmosphereConfig=DRY_AIR):number {
  if (!(rhoTheta>0)) return NaN;
  return a.pRef*Math.pow(a.rd*rhoTheta/a.pRef,a.gamma);
}
export function thetaFromTP(T:number,p:number,a:AtmosphereConfig=DRY_AIR):number { return T*Math.pow(a.pRef/p,a.kappa); }
export function temperatureFromThetaP(theta:number,p:number,a:AtmosphereConfig=DRY_AIR):number { return theta*Math.pow(p/a.pRef,a.kappa); }
export function rhoFromPT(p:number,T:number,a:AtmosphereConfig=DRY_AIR):number { return p/(a.rd*T); }

export interface ThermoDiagnostic { p:number; theta:number; T:number; }
export function diagnoseDry(rho:number,rhoTheta:number,a:AtmosphereConfig=DRY_AIR):ThermoDiagnostic {
  const p=pressureFromRhoTheta(rhoTheta,a), theta=rhoTheta/rho, T=temperatureFromThetaP(theta,p,a);
  return {p,theta,T};
}
