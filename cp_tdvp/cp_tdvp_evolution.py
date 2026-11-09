"""Integrated canonical-purification TDVP, with square-root-time startup.

Two current-state force backends: a structured single-excitation closure and
a generic dense Sylvester solve. Exact trajectories are diagnostics only.
"""
import tdvp_core as tn
import argparse
import csv
import time
from math import comb
from pathlib import Path
import numpy as np
from scipy.linalg import expm
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import expm_multiply
import single_excitation_study as single


def save(path,rows):single.save(path,rows)


def column_and_vacuum(state,anchor):
    n=state.n;left=[np.ones(1,complex)];right=[None]*(n+1);right[n]=np.ones(1,complex)
    for j,a in enumerate(state.tensors):left.append(left[-1]@a[:,1 if j==anchor else 0,:])
    for j in range(n-1,-1,-1):right[j]=state.tensors[j][:,1 if j==anchor else 0,:]@right[j+1]
    column=np.array([left[j]@a[:,3 if j==anchor else 2,:]@right[j+1] for j,a in enumerate(state.tensors)])
    vacuum=np.ones(1,complex)
    for a in state.tensors:vacuum=vacuum@a[:,0,:]
    return column,vacuum.item()


class SectorForce:
    def __init__(self,onsite,hopping,rates,anchor,clock='sqrt'):
        self.onsite,self.hopping,self.rates=onsite,hopping,rates;self.anchor=anchor;self.clock=clock

    def recover(self,state):
        col,b=column_and_vacuum(state,self.anchor)
        best=int(np.argmax(abs(col)))
        if best!=self.anchor:
            self.anchor=best;col,b=column_and_vacuum(state,best)
        diagonal=float(col[self.anchor].real);length=np.linalg.norm(col)
        if diagonal<1e-12:raise RuntimeError('Occupied column is too small for the rank-one closure.')
        v=col*(length/diagonal)
        return v,float(b.real)

    def __call__(self,u,state):
        v,b=self.recover(state);a=np.linalg.norm(v)
        dv=single.generator_action(v[:,None],self.onsite,self.hopping,self.rates)[:,0]
        ds=2*float(np.vdot(v,dv).real)
        speed=2*u if self.clock=='sqrt' else 1.
        if b<=1e-12:
            if u!=0 or self.clock!='sqrt':raise RuntimeError('Rank opening requires a finite startup step.')
            force=tn.scale(tn.vacuum(state.n),np.sqrt(max(0,-ds)))
        else:
            first=tn.dyad(dv,v,occupied=speed/a)
            second=tn.dyad(v,dv,occupied=speed/a)
            third=tn.dyad(v,v,occupied=-speed*ds/(2*a**3),vacuum_amplitude=-speed*ds/(2*b))
            force,_=tn.compress(tn.add(tn.add(first,second),third),12,normalize=False)
        return force,dict(closure_normalization=abs(a*a+b*b-1))

    def closure_error(self,state):
        v,b=self.recover(state);root=tn.dyad(v,v,occupied=1/np.linalg.norm(v),vacuum_amplitude=b)
        return tn.distance(state,root)


def sector(outdir,quick=False):
    rows=[];configs=[(32,.2),(32,.1),(32,.05),(32,.025),(128,.1),(128,.05),(512,.05)]
    if quick:configs=[(16,.2),(16,.1),(16,.05)]
    for n,du in configs:
        started=time.perf_counter();ons,hops,rates=single.model(n)
        initial=single.wavepacket(n,n//2-5,3.,.6)[:,None]
        total=1.;steps=round(np.sqrt(total)/du);du=np.sqrt(total)/steps
        # Positive symmetric local-channel startup, t_seed=du^2.
        v=single.local_integrate(initial,ons,hops,rates,du*du,du*du)
        s=float(np.linalg.norm(v)**2);state=tn.dyad(v[:,0],v[:,0],occupied=1/np.sqrt(s),vacuum_amplitude=np.sqrt(max(0,1-s)))
        provider=SectorForce(ons,hops,rates,int(np.argmax(abs(initial))))
        max_discard=0.;max_norm=0.;closure=[]
        for step in range(1,steps):
            state,info=tn.midpoint_step(state,step*du,du,provider,4)
            max_discard=max(max_discard,info['mid_retraction']['discarded_bound'],info['end_retraction']['discarded_bound'])
            max_norm=max(max_norm,info['force1']['closure_normalization'],info['force2']['closure_normalization'])
            if step==steps-1:closure.append(provider.closure_error(state))
        reference=single.reference_integrate(initial,ons,hops,rates,total)
        target=single.cp_mps(reference);error=tn.distance(state,target)
        vac=column_and_vacuum(state,provider.anchor)[1]
        exact_vac=np.sqrt(1-np.linalg.norm(reference)**2)
        row=dict(n=n,du=du,time=total,steps=steps,bond=state.bond_dimension(),cp_error=error,
                 vacuum_error=abs(vac-exact_vac),closure_error=max(closure,default=0),
                 max_closure_normalization=max_norm,max_compression_bound=max_discard,
                 seconds=time.perf_counter()-started)
        rows.append(row);save(outdir/'sector.csv',rows);print('sector',row,flush=True)
    return rows


def spin_model(n,kind):
    eye=np.eye(2);x=np.array([[0.,1.],[1.,0.]]);z=np.diag([1.,-1.]);sm=np.array([[0.,1.],[0.,0.]])
    def embed(op,j):return single.base.kron_all([op if k==j else eye for k in range(n)])
    xs=[embed(x,j) for j in range(n)];zs=[embed(z,j) for j in range(n)];sms=[embed(sm,j) for j in range(n)]
    d=2**n
    if kind=='pair':
        h=np.zeros((d,d),complex);jumps=[np.sqrt(.4+.1*np.sin(j)**2)*zs[j]@zs[j+1] for j in range(n-1)]
    else:
        h=.7*sum(zs[j]@zs[j+1] for j in range(n-1))+.43*sum(xs)+.19*sum(zs)
        jumps=[np.sqrt(.3+.12*np.cos(j)**2)*sms[j] for j in range(n)]
    identity=np.eye(d);liouv=-1j*(np.kron(identity,h)-np.kron(h.T,identity))
    for op in jumps:
        number=op.conj().T@op;liouv+=np.kron(op.conj(),op)-.5*np.kron(identity,number)-.5*np.kron(number.T,identity)
    psi=np.ones(d,complex)/np.sqrt(d)
    return h,jumps,csr_matrix(liouv),np.outer(psi,psi.conj())


def positive_root(rho):
    val,vec=np.linalg.eigh((rho+rho.conj().T)/2)
    if min(val)<-1e-10:raise RuntimeError('Reference/channel state is not positive.')
    # Do not turn density-level roundoff in a known kernel into sqrt(eps)
    # spurious CP amplitudes. The discarded spectrum is below 1e-14 relative.
    val=np.where(val>1e-14*max(float(max(val)),1e-300),val,0.)
    return (vec*np.sqrt(np.maximum(val,0)))@vec.conj().T


def positive_start(rho,h,jumps,elapsed):
    """CPTP Strang step: Hamiltonian halves and exact commuting dissipators.

    In the supplied models onsite losses commute with each other; pair-ZZ
    dephasing terms also commute. For other models the dissipative layer itself
    must be symmetrically split or otherwise supplied as a positive channel.
    """
    unitary=expm(-1j*h*elapsed/2);out=unitary@rho@unitary.conj().T
    for jump in jumps:
        number=jump.conj().T@jump
        rate=float(np.linalg.norm(jump,ord=2)**2)
        if np.linalg.norm(number-rate*np.eye(len(h)))<1e-12:
            p=-np.expm1(-2*rate*elapsed)/2;u=jump/np.sqrt(rate);out=(1-p)*out+p*u@out@u.conj().T
        else:
            survival=np.exp(-rate*elapsed);k0=np.eye(len(h))+(np.sqrt(survival)-1)*number/rate
            k1=np.sqrt((1-survival)/rate)*jump;out=k0@out@k0.conj().T+k1@out@k1.conj().T
    return unitary@out@unitary.conj().T


def positive_start_root(amplitude,h,jumps,elapsed,substeps=1):
    """Positive startup through Kraus factors, retaining sqrt-scale singular values.

    Forming rho before its root loses eigenvalues below density roundoff.
    Thin SVDs of the Kraus amplitude retain much smaller root eigenvalues.
    """
    factor=amplitude.copy();step=elapsed/substeps;unitary=expm(-1j*h*step/2)
    for _ in range(substeps):
        factor=unitary@factor
        for jump in jumps:
            number=jump.conj().T@jump;rate=float(np.linalg.norm(jump,ord=2)**2)
            if np.linalg.norm(number-rate*np.eye(len(h)))<1e-12:
                p=-np.expm1(-2*rate*step)/2
                factor=np.column_stack([np.sqrt(1-p)*factor,np.sqrt(p/rate)*jump@factor])
            else:
                survival=np.exp(-rate*step);k0=np.eye(len(h))+(np.sqrt(survival)-1)*number/rate
                k1=np.sqrt(-np.expm1(-rate*step)/rate)*jump
                factor=np.column_stack([k0@factor,k1@factor])
            u,s,_=np.linalg.svd(factor,full_matrices=False);factor=u*s
        factor=unitary@factor
    u,s,_=np.linalg.svd(factor,full_matrices=False)
    return (u*s)@u.conj().T


class DenseForce:
    def __init__(self,n,liouv,clock='sqrt'):self.n=n;self.liouv=liouv;self.clock=clock
    def __call__(self,u,state):
        amplitude=tn.to_matrix(state.dense(),self.n)
        basis,values,_=np.linalg.svd(amplitude,full_matrices=False)
        root=(basis*values)@basis.conj().T;rho=amplitude@amplitude.conj().T
        g=(self.liouv@rho.reshape(-1,order='F')).reshape(rho.shape,order='F')
        g=(g+g.conj().T)/2;coordinates=basis.conj().T@g@basis
        denominator=values[:,None]+values[None,:];keep=denominator>1e-12*values[0]
        inverse=np.zeros_like(coordinates);inverse[keep]=coordinates[keep]/denominator[keep]
        speed=2*u if self.clock=='sqrt' else 1.;force=speed*basis@inverse@basis.conj().T
        info=dict(branch_defect=np.linalg.norm(amplitude-root),null_forcing=np.linalg.norm(coordinates[~keep]),
                  smallest_root_eigenvalue=float(values[-1]))
        return tn.from_dense(tn.to_interleaved(force,self.n),self.n),info


def polar_retractor(n,chi):
    def retract(state):
        result,info=tn.compress(state,chi,normalize=True)
        a=tn.to_matrix(result.dense(),n);u,s,_=np.linalg.svd(a,full_matrices=False);root=(u*s)@u.conj().T
        info['polar_correction']=float(np.linalg.norm(a-root))
        result,second=tn.compress(tn.from_dense(tn.to_interleaved(root,n),n),chi,normalize=True)
        info['discarded_bound']+=second['discarded_bound']
        return result,info
    return retract


def density_residual_certificate(points,velocities,coordinates,liouv):
    """A posteriori full trace-norm bound from a cubic amplitude interpolant.

    rho=A A^dagger is a degree-six Bernstein polynomial; its residual against
    d rho/du=2u L(rho) has degree seven. Integral <= h/8 sum ||R_k||_1.
    This is an exact-arithmetic bound, evaluated in ordinary floating point.
    """
    total=0.
    for a,b,va,vb,u0,u1 in zip(points[:-1],points[1:],velocities[:-1],velocities[1:],coordinates[:-1],coordinates[1:]):
        step=u1-u0;controls=[a,a+step*va/3,b-step*vb/3,b]
        c=[]
        for k in range(7):
            block=np.zeros_like(a)
            for i in range(4):
                j=k-i
                if 0<=j<=3:block+=comb(3,i)*comb(3,j)/comb(6,k)*(controls[i]@controls[j].conj().T)
            c.append((block+block.conj().T)/2)
        derivative=[6*(c[k+1]-c[k])/step for k in range(6)]
        lc=[(liouv@x.reshape(-1,order='F')).reshape(x.shape,order='F') for x in c]
        for k in range(8):
            residual=np.zeros_like(a)
            for i in range(6):
                if 0<=k-i<=2:residual+=comb(5,i)*comb(2,k-i)/comb(7,k)*derivative[i]
            if k<=6:residual-=2*(7-k)/7*u0*lc[k]
            if k>=1:residual-=2*k/7*u1*lc[k-1]
            total+=step/8*float(np.sum(abs(np.linalg.eigvalsh((residual+residual.conj().T)/2))))
    return total


def projected_velocity(provider,u,state,mode='one'):
    force,_=provider(u,state)
    projector=tn.tangent_project if mode=='one' else tn.tangent_project_two
    velocity=projector(state,force)
    velocity=tn.add(velocity,state,beta=-tn.overlap(state,force).real)
    return tn.to_matrix(velocity.dense(),state.n)


def cp_step_defect_certificate(points,coordinates,liouv):
    """Contractivity certificate using exact-in-time dense channel actions.

    This is a verification backend, not an efficient large-system certificate.
    Spectral truncation in the verification root is bounded explicitly.
    """
    roots=[]
    for amplitude in points:
        u,s,_=np.linalg.svd(amplitude,full_matrices=False);roots.append((u*s)@u.conj().T)
    total=float(np.linalg.norm(points[-1]-roots[-1]))
    for a,b,u0,u1 in zip(roots[:-1],roots[1:],coordinates[:-1],coordinates[1:]):
        rho=a@a
        evolved=expm_multiply((u1*u1-u0*u0)*liouv,rho.reshape(-1,order='F')).reshape(rho.shape,order='F')
        val,vec=np.linalg.eigh((evolved+evolved.conj().T)/2)
        keep=val>1e-14*max(float(max(val)),1e-300)
        clipped=np.where(keep,np.maximum(val,0),0.)
        target=(vec*np.sqrt(clipped))@vec.conj().T
        root_discard_bound=np.sqrt(float(np.sum(np.maximum(val[~keep],0))))
        total+=float(np.linalg.norm(b-target))+root_discard_bound
    return total


def dense(outdir,quick=False):
    rows=[];configs=[]
    for kind in ('pair','ising_loss'):
        for du in (.2,.1,.05,.025,.0125):configs.append((4,kind,16,du,'polar','sqrt'))
    for chi in (2,4,8):
        for du in (.1,.05):configs.append((4,'ising_loss',chi,du,'polar','sqrt'))
    configs.extend([(4,'ising_loss',16,.05,'raw','sqrt'),(4,'pair',2,.05,'polar','sqrt'),
                    (5,'ising_loss',16,.05,'polar','sqrt')])
    if quick:configs=[(3,'pair',4,.1,'polar','sqrt'),(3,'ising_loss',4,.1,'polar','sqrt')]
    for n,kind,chi,du,branch,clock in configs:
        started=time.perf_counter();h,jumps,liouv,rho0=spin_model(n,kind);total=1.
        steps=round(np.sqrt(total)/du);du=np.sqrt(total)/steps
        root=positive_start_root(positive_root(rho0),h,jumps,du*du)
        state,_=tn.compress(tn.from_dense(tn.to_interleaved(root,n),n),chi,normalize=True)
        provider=DenseForce(n,liouv,clock);retract=polar_retractor(n,chi) if branch=='polar' else None
        x0=positive_root(rho0);q=np.eye(len(rho0))-rho0
        g0=(liouv@rho0.reshape(-1,order='F')).reshape(rho0.shape,order='F')
        points=[x0,tn.to_matrix(state.dense(),n)];velocities=[positive_root(q@g0@q),projected_velocity(provider,du,state)]
        coordinates=[0.,du]
        maximum_branch=maximum_null=maximum_correction=maximum_discard=0.
        for step in range(1,steps):
            state,info=tn.midpoint_step(state,step*du,du,provider,chi,retract)
            points.append(tn.to_matrix(state.dense(),n));velocities.append(projected_velocity(provider,(step+1)*du,state));coordinates.append((step+1)*du)
            for entry in (info['force1'],info['force2']):
                maximum_branch=max(maximum_branch,entry['branch_defect']);maximum_null=max(maximum_null,entry['null_forcing'])
            for entry in (info['mid_retraction'],info['end_retraction']):
                maximum_correction=max(maximum_correction,entry.get('polar_correction',0.));maximum_discard=max(maximum_discard,entry['discarded_bound'])
        reference=(expm_multiply(total*liouv,rho0.reshape(-1,order='F'))).reshape(rho0.shape,order='F')
        exact=positive_root(reference);amplitude=tn.to_matrix(state.dense(),n)
        ou,ov,_=np.linalg.svd(amplitude,full_matrices=False);own_root=(ou*ov)@ou.conj().T
        density_bound=density_residual_certificate(points,velocities,coordinates,liouv)
        cp_bound=float(np.linalg.norm(amplitude-own_root)+np.sqrt(density_bound))
        assert np.linalg.norm(amplitude-exact)<=cp_bound+1e-10
        row=dict(n=n,model=kind,chi=chi,du=du,branch=branch,cp_error=np.linalg.norm(amplitude-exact),
                 physical_error=.5*np.sum(abs(np.linalg.eigvalsh(amplitude@amplitude.conj().T-reference))),
                 final_branch_defect=np.linalg.norm(amplitude-own_root),hermiticity_error=np.linalg.norm(amplitude-amplitude.conj().T),
                 min_hermitian_eigenvalue=float(np.linalg.eigvalsh((amplitude+amplitude.conj().T)/2)[0]),
                 density_trace_norm_bound=density_bound,cp_error_bound=cp_bound,
                 max_stage_branch=maximum_branch,max_null_forcing=maximum_null,max_polar_correction=maximum_correction,
                 max_compression_bound=maximum_discard,seconds=time.perf_counter()-started)
        rows.append(row);save(outdir/'dense.csv',rows);print('dense',row,flush=True)
    return rows


def onset(outdir):
    rows=[]
    for kind in ('pair','ising_loss'):
        n=4;h,jumps,liouv,rho0=spin_model(n,kind);x0=positive_root(rho0)
        projector=np.eye(len(rho0))-rho0
        g=(liouv@rho0.reshape(-1,order='F')).reshape(rho0.shape,order='F')
        k=positive_root(projector@g@projector)
        for t in (1e-2,1e-3,1e-4,1e-5):
            rho=(expm_multiply(t*liouv,rho0.reshape(-1,order='F'))).reshape(rho0.shape,order='F')
            if kind=='pair':
                rates=.4+.1*np.sin(np.arange(n-1))**2
                root=tn.to_matrix(single.pair_cp(-np.expm1(-2*rates*t)/2).dense(),n)
                refinement=0.
            else:
                # A density eigensolve cannot resolve the smallest newborn
                # populations at these times. Validate in root coordinates.
                root=positive_start_root(x0,h,jumps,t,substeps=64)
                coarse=positive_start_root(x0,h,jumps,t,substeps=32)
                refinement=float(np.linalg.norm(root-coarse))
            rows.append(dict(model=kind,t=t,root_change=np.linalg.norm(root-x0),
                             onset_relative_error=np.linalg.norm((root-x0)/np.sqrt(t)-k),
                             leading_norm=np.linalg.norm(k),seed_error=np.linalg.norm(positive_start_root(x0,h,jumps,t)-root),
                             reference_root_refinement=refinement,reference_density_error=np.linalg.norm(root@root-rho)))
    save(outdir/'onset.csv',rows)


def clock_compare(outdir):
    rows=[];n=16;initial=np.ones(n,complex)/np.sqrt(n);rates=np.full(n,.7)
    reference=tn.dyad(initial,initial,occupied=np.exp(-.35),vacuum_amplitude=np.sqrt(-np.expm1(-.7)))
    for clock in ('physical','sqrt'):
        for step_size in (.2,.1,.05,.025,.0125):
            steps=round(1/step_size);step_size=1/steps
            seed_time=step_size if clock=='physical' else step_size**2
            state=tn.dyad(initial,initial,occupied=np.exp(-.35*seed_time),vacuum_amplitude=np.sqrt(-np.expm1(-.7*seed_time)))
            provider=SectorForce(np.zeros(n),np.zeros(n-1,complex),rates,n//2,clock)
            for step in range(1,steps):state,_=tn.midpoint_step(state,step*step_size,step_size,provider,4)
            rows.append(dict(clock=clock,step=step_size,steps=steps,seed_time=seed_time,cp_error=tn.distance(state,reference)))
    save(outdir/'clock.csv',rows);print('clock comparison done',flush=True)


def adaptive(outdir):
    rows=[]
    for du in (.1,.05,.025):
        n=4;h,jumps,liouv,rho0=spin_model(n,'ising_loss')
        seed=positive_start_root(positive_root(rho0),h,jumps,du*du)
        state,_=tn.compress(tn.from_dense(tn.to_interleaved(seed,n),n),2,normalize=True)
        provider=DenseForce(n,liouv);history=[state.bond_dimension()]
        maximum_branch=0.
        for step in range(1,round(1/du)):
            state,info=tn.midpoint_step(state,step*du,du,provider,16,polar_retractor(n,16),mode='two')
            history.append(state.bond_dimension());maximum_branch=max(maximum_branch,info['force1']['branch_defect'],info['force2']['branch_defect'])
        exact=positive_root(expm_multiply(liouv,rho0.reshape(-1,order='F')).reshape(rho0.shape,order='F'))
        rows.append(dict(n=n,du=du,initial_bond=history[0],final_bond=history[-1],bond_history=str(history),
                         cp_error=np.linalg.norm(tn.to_matrix(state.dense(),n)-exact),max_stage_branch=maximum_branch))
    save(outdir/'adaptive.csv',rows);print('two-site adaptive done',flush=True)


def adaptive_time(outdir):
    rows=[]
    for tolerance in (1e-3,1e-4):
        n=4;chi=16;ham,jumps,liouv,rho0=spin_model(n,'ising_loss');started=time.perf_counter()
        seed_u=.1
        while True:
            full=positive_start_root(positive_root(rho0),ham,jumps,seed_u**2)
            fine=positive_start_root(positive_root(rho0),ham,jumps,seed_u**2,substeps=2)
            seed_estimate=np.linalg.norm(full-fine)/3
            if seed_estimate<tolerance*.05:break
            seed_u/=2
        state=tn.from_dense(tn.to_interleaved(fine,n),n,chi);state,_=tn.compress(state,chi,normalize=True)
        provider=DenseForce(n,liouv);retract=polar_retractor(n,chi)
        q=np.eye(len(rho0))-rho0;g0=(liouv@rho0.reshape(-1,order='F')).reshape(rho0.shape,order='F')
        points=[positive_root(rho0),fine];velocities=[positive_root(q@g0@q),projected_velocity(provider,seed_u,state)]
        coordinates=[0.,seed_u];u=seed_u;step=.05;accepted=rejected=0;estimated_sum=seed_estimate;largest_polar=0.
        while u<1-1e-14:
            step=min(step,1-u)
            full,info=tn.midpoint_step(state,u,step,provider,chi,retract)
            half,_=tn.midpoint_step(state,u,step/2,provider,chi,retract)
            fine,info=tn.midpoint_step(half,u+step/2,step/2,provider,chi,retract)
            estimate=tn.distance(full,fine)/3;budget=.5*tolerance*step
            if (accepted+rejected)%100==0:
                print('adaptive trial',tolerance,accepted,rejected,u,step,estimate,budget,flush=True)
            multiplier=float(np.clip(.85*(budget/max(estimate,1e-16))**(1/3),.2,2.))
            if estimate<=budget:
                state=fine;u+=step;accepted+=1;estimated_sum+=estimate
                largest_polar=max(largest_polar,info['mid_retraction']['polar_correction'],info['end_retraction']['polar_correction'])
                points.append(tn.to_matrix(state.dense(),n));velocities.append(projected_velocity(provider,u,state));coordinates.append(u)
            else:rejected+=1
            step*=multiplier
            if accepted+rejected>2000:raise RuntimeError('Adaptive integration exceeded its trial budget.')
        exact=positive_root(expm_multiply(liouv,rho0.reshape(-1,order='F')).reshape(rho0.shape,order='F'))
        amplitude=tn.to_matrix(state.dense(),n);ou,ov,_=np.linalg.svd(amplitude,full_matrices=False)
        branch=np.linalg.norm(amplitude-(ou*ov)@ou.conj().T)
        certificate=branch+np.sqrt(density_residual_certificate(points,velocities,coordinates,liouv))
        step_defect_bound=cp_step_defect_certificate(points,coordinates,liouv)
        error=np.linalg.norm(amplitude-exact);assert error<certificate
        assert error<step_defect_bound+1e-10
        rows.append(dict(tolerance=tolerance,seed_u=seed_u,seed_error_estimate=seed_estimate,accepted=accepted,rejected=rejected,
                         cp_error=error,accumulated_step_estimates=estimated_sum,cp_error_bound=certificate,
                         cp_step_defect_bound=step_defect_bound,
                         max_accepted_polar_correction=largest_polar,seconds=time.perf_counter()-started))
    save(outdir/'adaptive_time.csv',rows);print('adaptive time done',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--part',choices=['all','sector','dense','onset','clock','adaptive','adaptive_time'],default='all')
    p.add_argument('--quick',action='store_true');p.add_argument('--outdir',type=Path,default=Path(__file__).parent/'cp_tdvp_results')
    args=p.parse_args();args.outdir.mkdir(parents=True,exist_ok=True)
    if args.part in ('all','onset'):onset(args.outdir)
    if args.part in ('all','sector'):sector(args.outdir,args.quick)
    if args.part in ('all','dense'):dense(args.outdir,args.quick)
    if args.part in ('all','clock'):clock_compare(args.outdir)
    if args.part in ('all','adaptive'):adaptive(args.outdir)
    if args.part in ('all','adaptive_time'):adaptive_time(args.outdir)


if __name__=='__main__':main()
