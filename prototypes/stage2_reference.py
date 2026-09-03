import argparse, collections, json, math, time
import numpy as np

C=np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]],float)
A=np.array([[0,1,0],[0,-1,0],[-1,0,0],[1,0,0],[1,0,0],[-1,0,0]],float)
B=np.array([[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,1,0],[0,1,0]],float)

def fxyz(f,a,b):
    x=C[f]+math.tan(a)*A[f]+math.tan(b)*B[f]
    return x/np.linalg.norm(x)

def tri_area(a,b,c):
    return 2*math.atan2(abs(np.dot(a,np.cross(b,c))),1+np.dot(a,b)+np.dot(b,c)+np.dot(c,a))

def build_grid(n):
    q=np.linspace(-math.pi/4,math.pi/4,n+1); centers=[]; areas=[]; faces=[]; edges=collections.defaultdict(list)
    key=lambda p: tuple(np.round(p,10))
    for f in range(6):
      for j in range(n):
       for i in range(n):
        cs=[fxyz(f,q[i],q[j]),fxyz(f,q[i+1],q[j]),fxyz(f,q[i+1],q[j+1]),fxyz(f,q[i],q[j+1])]
        centers.append(fxyz(f,(q[i]+q[i+1])/2,(q[j]+q[j+1])/2)); faces.append(f)
        areas.append(tri_area(cs[0],cs[1],cs[2])+tri_area(cs[0],cs[2],cs[3])); idx=len(centers)-1
        for a,b in [(0,1),(1,2),(2,3),(3,0)]:
            ka,kb=key(cs[a]),key(cs[b]); edges[(ka,kb) if ka<kb else (kb,ka)].append(idx)
    return np.array(centers),np.array(areas),np.array(faces),edges

def p2(n,cfl=.35):
    centers,areas,faces,raw=build_grid(n); axis=np.array([.3,.5,.8124]); axis/=np.linalg.norm(axis); es=[]
    for k,pair in raw.items():
        assert len(pair)==2
        a,b=pair; p0=np.array(k[0]);p1=np.array(k[1]);p0/=np.linalg.norm(p0);p1/=np.linalg.norm(p1)
        mid=p0+p1;mid/=np.linalg.norm(mid); length=math.acos(np.clip(np.dot(p0,p1),-1,1))
        d=centers[b]-centers[a];d-=np.dot(d,mid)*mid;d/=np.linalg.norm(d)
        s=float(np.dot(np.cross(axis,mid),d))*length; es.append((a,b,s))
    tc=np.array([.9,-.1,.42]);tc/=np.linalg.norm(tc); qi=np.exp(-(np.arccos(np.clip(centers@tc,-1,1))/.35)**2)
    mass=qi*areas;m0=mass.sum();rates=np.zeros(len(mass))
    for a,b,s in es: rates[a if s>=0 else b]+=abs(s)
    dt=cfl*np.min(areas/np.maximum(rates,1e-12));steps=math.ceil(2*math.pi/dt);dt=2*math.pi/steps
    for _ in range(steps):
        q=mass/areas;dm=np.zeros_like(mass)
        for a,b,s in es:
            flux=s*(q[a] if s>=0 else q[b])*dt;dm[a]-=flux;dm[b]+=flux
        mass+=dm
    qf=mass/areas;l2=math.sqrt(np.sum(areas*(qf-qi)**2)/np.sum(areas*qi**2))
    return {'N':n,'cells':len(mass),'edges':len(es),'steps':steps,'mass_rel':float((mass.sum()-m0)/m0),'qmin':float(qf.min()),'qmax':float(qf.max()),'l2':l2,'seam_edges':int(sum(faces[a]!=faces[b] for a,b,_ in es))}

def acoustic_explicit(cfl,steps=300):
    nz,H,c,rho=80,20000.,340.,1.2;dz=H/nz;dt=cfl*dz/c;zc=(np.arange(nz)+.5)*dz;p=np.exp(-((zc-H/2)/1200)**2)*100;w=np.zeros(nz+1)
    def energy(): return np.sum(p*p/(2*rho*c*c))*dz+np.sum(.5*rho*w*w)*dz
    e0=energy();e=e0
    for n in range(steps):
        w[1:-1]-=dt/rho*(p[1:]-p[:-1])/dz;p-=dt*rho*c*c*(w[1:]-w[:-1])/dz;e=energy()
        if not np.isfinite(e) or e/e0>1e90:return {'cfl':cfl,'stable':False,'steps_completed':n+1,'energy_ratio':float(e/e0)}
    return {'cfl':cfl,'stable':True,'steps_completed':steps,'energy_ratio':float(e/e0)}

def acoustic_cn(cfl,steps=300):
    nz,H,c,rho=80,20000.,340.,1.2;dz=H/nz;dt=cfl*dz/c;zc=(np.arange(nz)+.5)*dz;p=np.exp(-((zc-H/2)/1200)**2)*100;w=np.zeros(nz+1);m=nz-1
    L=np.diag(np.full(m,-2/dz**2))+np.diag(np.full(m-1,1/dz**2),1)+np.diag(np.full(m-1,1/dz**2),-1);mu=dt*dt*c*c/4;AI=np.eye(m)-mu*L;BI=np.eye(m)+mu*L
    def energy(): return np.sum(p*p/(2*rho*c*c))*dz+np.sum(.5*rho*w*w)*dz
    e0=energy()
    for _ in range(steps):
        rhs=BI@w[1:-1]-dt/rho*(p[1:]-p[:-1])/dz;wn=np.zeros_like(w);wn[1:-1]=np.linalg.solve(AI,rhs);p-=dt*rho*c*c*.5*((wn[1:]+w[1:])-(wn[:-1]+w[:-1]))/dz;w=wn
    return {'cfl':cfl,'stable':True,'steps_completed':steps,'energy_ratio':float(energy()/e0)}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--full',action='store_true');args=ap.parse_args();t=time.time()
    ns=[12,24,48] if args.full else [12,24]
    result={'P2':[p2(n) for n in ns],'P3':{'explicit':[acoustic_explicit(x) for x in [.25,1,3,10]],'implicit':[acoustic_cn(x) for x in [.25,1,3,10]]},'seconds':time.time()-t}
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
