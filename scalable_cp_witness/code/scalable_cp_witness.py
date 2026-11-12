"""Positive-factor CP evolution with an independent global density-residual audit."""
import argparse
import copy
import csv
import json
from pathlib import Path
from time import perf_counter
from collections import deque

import tdvp_core as tn
import mpo_cp_force as mpo
import numpy as np
from cp_accuracy_refinement import hermitian_round, matrix
from guarded_cp_tdvp import MPOAlgebra


def prepared_power(n, exponent, polarization=.12, angle=.35):
    """Exact MPO for a power of the circuit-prepared density, with a two-bit memory."""
    p = np.array([(1+polarization)/2,(1-polarization)/2])
    return circuit_diagonal(n,p**exponent,angle)


def circuit_diagonal(n, values, angle):
    h = np.array([[1.,1.],[1.,-1.]])/np.sqrt(2)
    local = ((h*values) @ h).ravel()
    labels = np.arange(4)
    ket, bra = labels//2, labels%2
    phase = np.exp(1j*angle*(np.outer(ket,ket)-np.outer(bra,bra)))
    state = tn.MPS(n,max_bond=4)
    state.tensors = []
    rotate = np.kron(h,h)
    for j in range(n):
        coeff = local[None,:] if j==0 else phase*local
        a = (coeff[:,:,None]*np.eye(4)[None,:,:]) if j<n-1 else coeff[:,:,None]
        state.tensors.append(np.einsum('ps,asb->apb',rotate,a))
    state.center = n-1
    return state


def parity_power(n, exponent, parity=.6, transverse=.4):
    """Power of (I + parity Z^N + transverse X_1)/2^N using anticommutation."""
    radius=np.hypot(parity,transverse)
    a,b=(1+radius)**exponent,(1-radius)**exponent
    avg=(a+b)/2
    slope=(a-b)/(2*radius) if radius else 0.
    def string(first,rest):
        s=tn.MPS(n,max_bond=1)
        local=2.**(-exponent)
        s.tensors=[(local*x).reshape(1,4,1).copy() for x in [first]+[rest]*(n-1)]
        return s
    ident=string(mpo.I,mpo.I)
    p=string(mpo.Z,mpo.Z)
    q=string(mpo.X,mpo.I)
    return tn.add(tn.add(ident,p,avg,slope*parity),q,beta=slope*transverse)


def channel(density, t, noise='dephasing'):
    out = copy.deepcopy(density)
    if noise=='dephasing':
        weights = np.array([1.,np.exp(-t),np.exp(-t),1.])
        out.tensors = [a*weights[None,:,None] for a in out.tensors]
    elif noise=='loss':
        survive = np.exp(-t)
        layer = np.diag([1.,np.sqrt(survive),np.sqrt(survive),survive])
        layer[0,3] = -np.expm1(-t)
        out.tensors = [np.einsum('ij,ajb->aib',layer,a) for a in out.tensors]
    else:
        raise ValueError('Unknown channel')
    return out


def factor_to_root(factor, bond):
    positive = mpo.multiply(factor, mpo.adjoint(factor))
    root = hermitian_round(positive, bond, normalize=True)
    norm = tn.norm(positive)
    factor = tn.scale(factor,1/np.sqrt(norm))
    positive = tn.scale(positive,1/norm)
    return factor, root, tn.distance(root, positive)


def center_norm(residual):
    center=residual.tensors[-1]
    peak=float(np.max(np.abs(center)))
    return 0. if peak==0 else peak*float(np.linalg.norm(center/peak))


def weighted_norm(residual, metric):
    """Contract Tr(R* R M) with a positive metric M, without a dense matrix.

    R is left canonical with its norm on the last site. Scale transfers
    locally to avoid overflow from the inverse-density MPO.
    """
    length=center_norm(residual)
    if length==0:
        return 0.
    env = np.ones((1,1,1),complex)
    log_scale = np.log(length)
    for j,(a,m) in enumerate(zip(residual.tensors,metric.tensors)):
        r = a.reshape(a.shape[0],2,2,a.shape[-1])
        if j==residual.n-1:
            r = r/length
        local = m.reshape(m.shape[0],2,2,m.shape[-1])
        transfer = np.einsum('abc,bpje->acpje',env,r,optimize=True)
        transfer = np.einsum('acpje,cjif->apeif',transfer,local,optimize=True)
        env = np.einsum('apid,apeif->def',r.conj(),transfer,optimize=True)
        scale = np.max(np.abs(env))
        if scale == 0:
            raise ArithmeticError('Zero weighted contraction on a nonzero residual')
        env /= scale
        log_scale += .5*np.log(scale)
    scalar = env.item()
    if scalar.real <= 0 or abs(scalar.imag)>1e-8*scalar.real:
        raise ArithmeticError('Lost positivity in the weighted contraction')
    value=float(np.exp(log_scale)*np.sqrt(scalar.real))
    if not np.isfinite(value) or value==0:
        raise ArithmeticError('Weighted residual is outside the floating-point range')
    return value


def compress_residual(defect,bond):
    """Scale before SVD so discarded singular values can be squared safely."""
    defect.center=defect.n-1
    defect.move_center(0)
    peak=float(np.max(np.abs(defect.tensors[0])))
    if peak == 0:
        return tn.compress(defect,bond)
    if not np.isfinite(peak):
        raise ArithmeticError('Nonfinite density residual')
    normalized=copy.deepcopy(defect)
    normalized.tensors[0]/=peak
    residual,info=tn.compress(normalized,bond)
    residual.tensors[-1]*=peak
    info['discarded_bound']*=peak
    return residual,info


def checked_weighted_defect(defect, residual, discarded, metric, sqrt_metric_norm, tolerance):
    remainder = sqrt_metric_norm*discarded
    weighted = weighted_norm(residual,metric)
    if remainder > max(.1*tolerance,.25*weighted):
        # Restore the untruncated defect before applying a large weight.
        defect.center = defect.n-1
        defect.move_center(0)
        defect.move_center(defect.n-1)
        weighted = weighted_norm(defect,metric)
        remainder = 0.
    return weighted,remainder


def certify_candidate(candidate, factor, density, metric, sqrt_metric_norm, *,
                      residual_bond=24, tolerance=1e-5, trace_error=0., gap_free=False):
    """Audit any spin MPO against a density, paying for Hermitian rounding.

    For a linear bound supply M >= rho_true^{-1} and sqrt_metric_norm >=
    sqrt(||M||op). In gap-free mode M may instead dominate the inverse of
    any full-rank trace-one auxiliary density. trace_error bounds the trace
    norm of the physical density error. The caller supplies the metric proof.
    """
    if not (np.isfinite(sqrt_metric_norm) and sqrt_metric_norm>0 and
            np.isfinite(trace_error) and trace_error>=0 and residual_bond>=1):
        raise ValueError('Invalid norm, reference error or bond')
    h = hermitian_round(candidate,candidate.bond_dimension())
    branch = tn.distance(h,mpo.multiply(factor,mpo.adjoint(factor)))
    defect = tn.add(density,mpo.multiply(h,h),beta=-1.)
    residual,info = compress_residual(defect,residual_bond)
    weighted,remainder = checked_weighted_defect(defect,residual,info['discarded_bound'],
                                                  metric,sqrt_metric_norm,tolerance)
    magnitude = (np.sqrt(weighted+remainder+trace_error) if gap_free
                 else weighted+remainder+sqrt_metric_norm*trace_error)
    return tn.distance(h,candidate)+2*branch+magnitude


class WitnessAudit:
    def __init__(self,n,polarization=.12,angle=.35,residual_bond=24,jensen=True,noise='dephasing',
                 local_correction=False):
        self.n = n
        self.noise = noise
        self.polarization = polarization
        self.angle = angle
        self.local_correction = local_correction
        self.physical = prepared_power(n,1,polarization,angle)
        self.inverse = prepared_power(n,-.5,polarization,angle)
        self.inverse_density = prepared_power(n,-1.,polarization,angle)
        self.preconditioner = prepared_power(n,-.375,polarization,angle)
        self.root_preconditioner = prepared_power(n,-.25,polarization,angle)
        self.factor_preconditioner = prepared_power(n,-.125,polarization,angle)
        self.inverse_norm = ((1-abs(polarization))/2)**(-n/2)
        self.residual_bond = residual_bond
        self.jensen = jensen
        self.metric_time = None

    @classmethod
    def parity_chain(cls,n,jensen=True):
        engine=cls.__new__(cls)
        engine.n,engine.noise,engine.jensen=n,'dephasing',jensen
        engine.local_correction=False
        engine.physical=parity_power(n,1.)
        engine.inverse=parity_power(n,-.5)
        engine.inverse_density=parity_power(n,-1.)
        engine.preconditioner=parity_power(n,-.375)
        engine.root_preconditioner=parity_power(n,-.25)
        engine.factor_preconditioner=parity_power(n,-.125)
        engine.inverse_norm=2.**(n/2)/np.sqrt(1-np.hypot(.6,.4))
        engine.residual_bond,engine.metric_time=24,None
        return engine

    def endpoint(self, root, witness_distance, target, time, tolerance=1e-5):
        # root is Hermitian by construction. Charge the full truncation
        # allowance of the residual before multiplying by the inverse anchor.
        square = mpo.multiply(root,root)
        defect = tn.add(target,square,beta=-1.)
        residual, info = compress_residual(defect, self.residual_bond)
        discarded = info['discarded_bound']
        # For loss, Phi(I) >= exp(-nt) I and rho_t >= lambda_min(rho_0) Phi(I).
        metric_norm = self.inverse_norm*(np.exp(.5*self.n*time) if self.noise=='loss' else 1.)
        scalar = metric_norm*(center_norm(residual)+discarded)
        mass = ((1+np.exp(-time))/2)**self.n if self.noise=='dephasing' else None
        remainder = metric_norm*discarded
        if self.jensen:
            if self.metric_time != time:
                self.metric = channel(self.inverse_density,time,self.noise)
                if self.noise=='loss':
                    # Normalize the CP map by Phi(I). The Schur complement gives
                    # M=Phi(I)^(-1) Phi(rho_0^(-1)) Phi(I)^(-1) >= rho_t^(-1).
                    survive = np.exp(-time)
                    correction = np.diag([1/(2-survive),1/survive])
                    local = np.kron(correction,correction)
                    self.metric.tensors = [np.einsum('ij,ajb->aib',local,a) for a in self.metric.tensors]
                    # Optional proposal only. The independently tracked metric stays exact.
                    p = 1-(1-self.polarization)*survive
                    self.preconditioner = prepared_power(self.n,-.375,p,self.angle)
                    self.root_preconditioner = prepared_power(self.n,-.25,p,self.angle)
                    self.factor_preconditioner = prepared_power(self.n,-.125,p,self.angle)
                self.metric_time = time
            weighted,remainder=checked_weighted_defect(defect,residual,discarded,self.metric,
                                                        metric_norm,tolerance)
            magnitude = min(scalar, weighted+remainder)
        else:
            weighted = tn.norm(mpo.multiply(residual,self.inverse)) + remainder
            magnitude = min(scalar, weighted/np.sqrt(mass), np.sqrt(weighted))
        return dict(bound=2*witness_distance+magnitude, witness=witness_distance,
                    weighted_residual=weighted, scalar_bound=scalar, no_error_mass=mass,
                    residual_rounding=remainder), defect if self.local_correction else residual

    def correction(self,residual,work_bond):
        if self.local_correction:
            p = self.polarization
            if self.noise=='loss':
                p = 1-(1-p)*np.exp(-self.metric_time)
            d = (np.array([1+p,1-p])/2)**(-.375)
            weight = np.outer(d,d).ravel()
            weighted = copy.deepcopy(residual)
            weighted.tensors = [a*weight[None,:,None] for a in weighted.tensors]
            return tn.scale(tn.compress(weighted,work_bond)[0],.25)
        return congruence(self.preconditioner,residual,work_bond,.25)

    def transport(self,factor,t0,t1,bond,work):
        """Move a product reference with loss, using its preparation circuit as a proposal."""
        def pops(t):
            excited = (1-self.polarization)*np.exp(-t)/2
            return np.array([1-excited,excited])
        ratio = circuit_diagonal(self.n,(pops(t1)/pops(t0))**.125,self.angle)
        return hermitian_round(congruence(ratio,factor,work,1.),bond)


def congruence(anchor,source,bond,coefficient):
    algebra = MPOAlgebra(bond)
    left = algebra.product(anchor,source)
    return tn.scale(algebra.product(left,anchor),coefficient)


def fit_witness(factor,root,anchor,bond,work_bond,iterations=3):
    """Fit the factor to any Hermitian CP trial and return the attained distance."""
    difference = tn.add(root,mpo.multiply(factor,mpo.adjoint(factor)),beta=-1.)
    distance = tn.norm(difference)
    for _ in range(iterations):
        source = tn.compress(difference,work_bond)[0]
        update = congruence(anchor,source,work_bond,.5)
        trial = hermitian_round(tn.add(factor,update),bond)
        new_diff = tn.add(root,mpo.multiply(trial,mpo.adjoint(trial)),beta=-1.)
        next_distance = tn.norm(new_diff)
        if next_distance>=distance:
            break
        factor, difference, distance = trial,new_diff,next_distance
    return factor,distance


def tdvp_prediction(state,t0,t1,engine,bond,work_bond):
    def force(t,current):
        square = tn.compress(mpo.multiply(current,current),work_bond)[0]
        rate = 1. if engine.noise=='loss' else .5
        rhs = tn.compress(mpo.onsite_dissipation(square,np.full(current.n,rate),engine.noise),
                          work_bond)[0]
        alg = MPOAlgebra(work_bond)
        solution = congruence(engine.root_preconditioner,rhs,work_bond,.5)
        applied = alg.combine(alg.product(current,solution),alg.product(solution,current),1.,1.)
        residual = alg.combine(rhs,applied,1.,-1.)
        solution = alg.combine(solution,congruence(engine.root_preconditioner,residual,work_bond,.5),1.,1.)
        return solution, {}
    def retract(candidate):
        return hermitian_round(candidate,bond,normalize=True), {}
    return tn.midpoint_step(state,t0,t1-t0,force,bond,retract=retract,mode='two')[0]


def save_table(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f,fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def run(n,times,levels,tolerance,out,passes,polarization=.12,angle=.35,jensen=True,predict=False,
        noise='dephasing',transport=False,local_correction=False,max_halvings=0,parity=False):
    out.mkdir(parents=True,exist_ok=True)
    base = f'n{n}_t{times[-1]:g}_s{len(times)}_p{polarization:g}_a{angle:g}'
    if jensen:
        base += '_jensen'
    if predict:
        base += '_tdvp'
    if noise!='dephasing':
        base += '_'+noise
    if transport:
        base += '_transport'
    if local_correction:
        base += '_local'
    if max_halvings:
        base += f'_adapt{max_halvings}'
    if parity:
        base=base.replace(f'p{polarization:g}_a{angle:g}','parity0.6_x0.4')
    with (out/(base+'_settings.json')).open('w') as f:
        json.dump(dict(n=n,times=list(times),levels=levels,tolerance=tolerance,passes=passes,
                    family='parity' if parity else 'correlated',noise=noise,tdvp=predict,
                    jensen=jensen,transport=transport,local_correction=local_correction,
                    max_halvings=max_halvings,polarization=None if parity else polarization,
                    angle=None if parity else angle),f,indent=2)
    started = perf_counter()
    if parity:
        engine=WitnessAudit.parity_chain(n,jensen)
        factor=parity_power(n,.25)
    else:
        engine = WitnessAudit(n,polarization,angle,jensen=jensen,noise=noise,local_correction=local_correction)
        factor = prepared_power(n,.25,polarization,angle)
    factor, root, eta = factor_to_root(factor,levels[0][1])
    history, endpoints = [], []
    queue=deque((t,0) for t in [0.]+list(times))
    accepted_time=0.
    while queue:
        t,depth=queue.popleft()
        previous=factor,root,eta
        target = channel(engine.physical,t,noise)
        info, residual = engine.endpoint(root,eta,target,t,tolerance)
        previous_factor=factor
        prediction_bound = ''
        used_prediction = False
        if t and predict:
            fb = max(factor.bond_dimension(),levels[0][0])
            rb = max(root.bond_dimension(),levels[0][1])
            wb = max(levels[0][2],fb,rb)
            prediction = tdvp_prediction(root,accepted_time,t,engine,rb,wb)
            fitted, d = fit_witness(factor,prediction,engine.factor_preconditioner,fb,wb)
            proposal, r = engine.endpoint(prediction,d,target,t,tolerance)
            prediction_bound=proposal['bound']
            if prediction_bound < info['bound']:
                factor,root,eta = fitted,prediction,d
                info,residual = proposal,r
                used_prediction=True
        used_transport=False
        if t and transport:
            fb=max(previous_factor.bond_dimension(),levels[0][0])
            rb=max(root.bond_dimension(),levels[0][1])
            transported=engine.transport(previous_factor,accepted_time,t,fb,max(12,fb))
            transported,x,d=factor_to_root(transported,rb)
            proposal,r=engine.endpoint(x,d,target,t,tolerance)
            if proposal['bound'] < info['bound']:
                factor,root,eta=transported,x,d
                info,residual=proposal,r
                used_transport,used_prediction=True,False
        print(base,'start',t,info['bound'],flush=True)
        for fb,rb,wb in levels:
            if info['bound'] <= tolerance:
                break
            for k in range(passes):
                update = engine.correction(residual,wb)
                improved = False
                for step_length in (1.,.5,.25):
                    trial = hermitian_round(tn.add(factor,update,beta=step_length),fb)
                    trial, candidate, distance = factor_to_root(trial,rb)
                    trial_info, trial_residual = engine.endpoint(candidate,distance,target,t,tolerance)
                    better = trial_info['bound'] < info['bound']
                    history.append(dict(time=t, level_factor=fb, level_root=rb, work=wb,
                                        iteration=k+1, step_length=step_length,
                                        improved=better, **trial_info))
                    if better:
                        factor, root, eta = trial, candidate, distance
                        info, residual = trial_info,trial_residual
                        improved = True
                        break
                if not improved or info['bound']<=tolerance:
                    break
        if history:
            save_table(out/(base+'_trials.csv'),history)
        actual = ''
        if parity:
            actual=tn.distance(root,parity_power(n,.5,.6,.4*np.exp(-t)))
            assert actual<=info['bound']+1e-11
        elif n<=8:
            rho = matrix(target)
            val,vec = np.linalg.eigh(rho)
            actual = np.linalg.norm(matrix(root) - (vec*np.sqrt(np.maximum(val,0)))@vec.conj().T)
            assert actual <= info['bound']+5e-12
        passed = info['bound']<=tolerance
        row = dict(n=n,time=t,accepted=passed,depth=depth,last_accepted=t if passed else accepted_time,
                   actual=actual,factor_bond=factor.bond_dimension(),
                   root_bond=root.bond_dimension(),prediction_bound=prediction_bound,
                   used_prediction=used_prediction,used_transport=used_transport,
                   seconds=perf_counter()-started,**info)
        endpoints.append(row)
        save_table(out/(base+'.csv'),endpoints)
        print(base,'endpoint',t,passed,info['bound'],actual,'seconds',row['seconds'],flush=True)
        if not passed:
            if t>accepted_time and depth<max_halvings:
                factor,root,eta=previous
                queue.appendleft((t,depth+1))
                queue.appendleft(((accepted_time+t)/2,depth+1))
                continue
            return False
        accepted_time=t
        np.savez_compressed(out/(base+'_last_accepted.npz'),time=t,bound=info['bound'],
                            **{f'g{j}':a for j,a in enumerate(factor.tensors)},
                            **{f'x{j}':a for j,a in enumerate(root.tensors)})
    return True


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n',type=int,default=8)
    parser.add_argument('--times',type=float,nargs='+',default=[.005,.02,.05,.1])
    parser.add_argument('--tol',type=float,default=1e-5)
    parser.add_argument('--levels',default='4,6,10:6,8,12:8,12,16')
    parser.add_argument('--passes',type=int,default=4)
    parser.add_argument('--polarization',type=float,default=.12)
    parser.add_argument('--angle',type=float,default=.35)
    parser.add_argument('--mass-bound',action='store_true')
    parser.add_argument('--tdvp',action='store_true')
    parser.add_argument('--noise',choices=['dephasing','loss'],default='dephasing')
    parser.add_argument('--transport',action='store_true')
    parser.add_argument('--local-correction',action='store_true')
    parser.add_argument('--max-halvings',type=int,default=0)
    parser.add_argument('--parity',action='store_true',help='Noncommuting full-rank global parity example')
    here=Path(__file__).resolve().parent
    default_out=here.parent if here.name=='code' else here.parent/'outputs'/'scalable_cp_witness'
    parser.add_argument('--out',type=Path,default=default_out)
    args=parser.parse_args()
    levels=[tuple(map(int,s.split(','))) for s in args.levels.split(':')]
    if (args.n<1 or not 0<=abs(args.polarization)<1 or args.tol<=0 or args.passes<1
        or any(t<=0 for t in args.times) or np.any(np.diff(args.times)<=0)
        or any(len(s)!=3 or min(s)<1 for s in levels) or (args.noise=='loss' and args.mass_bound)
        or (args.transport and args.noise!='loss') or args.max_halvings<0
        or (args.parity and (args.noise!='dephasing' or args.transport or args.local_correction))):
        parser.error('Invalid chain, time grid, tolerance or bond budget.')
    success=run(args.n,args.times,levels,args.tol,args.out,args.passes,args.polarization,args.angle,
                not args.mass_bound,args.tdvp,args.noise,args.transport,args.local_correction,args.max_halvings,
                args.parity)
    raise SystemExit(0 if success else 2)
