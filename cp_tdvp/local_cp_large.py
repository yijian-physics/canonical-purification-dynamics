"""Bounded-depth local CP updates and large-chain validation.

Only local physical marginals enter gate construction. For onsite noise they
evolve independently; a finite-range benchmark obtains them from bounded
backward causal cones. Dense whole-chain states occur only in small validations.
All CSV trace distances use one half of the trace norm. A CP residual against
a finite reference window becomes a certificate only after adding a justified
exterior tail bound; see report.md in the accompanying deliverable.
"""
from __future__ import annotations
import argparse
import copy
import csv
from dataclasses import dataclass
import json
import os
from pathlib import Path
import time
os.environ["OPENBLAS_NUM_THREADS"]="1"
os.environ["OMP_NUM_THREADS"]="1"
os.environ["MKL_NUM_THREADS"]="1"
import numpy as np
from scipy.linalg import eigh, expm
import cluster_dephasing as base
import ising_cp_study as previous


def trace_distance(a,b):
    return float(np.sum(np.abs(np.linalg.eigvalsh((a-b+(a-b).conj().T)/2)))/2)


def normalize_center(state):
    norm=float(np.linalg.norm(state.tensors[state.center]))
    state.tensors[state.center]/=norm
    return norm


@dataclass
class CompactGate:
    basis: np.ndarray
    rotation: np.ndarray
    residual: float

    def act(self,block):
        left,physical,right=block.shape
        flat=block.transpose(1,0,2).reshape(physical,-1)
        coordinates=self.basis.conj().T@flat
        flat=flat+self.basis@(self.rotation@coordinates-coordinates)
        return flat.reshape(physical,left,right).transpose(1,0,2)


def fit_gate(physical,sites,core_start,width,kraus):
    length=int(np.log2(len(physical)))
    source=previous.small_positive_root(physical)
    noisy=physical
    for site in sites:
        noisy=onsite_update(noisy,length,site,kraus)
    target=previous.small_positive_root(noisy)
    unfold=lambda x:previous.branch.unfold(base.to_interleaved(x,length),length,core_start,width)
    initial,desired=unfold(source),unfold(target)
    basis,values,_=np.linalg.svd(np.column_stack([initial,desired]),full_matrices=False)
    basis=basis[:,values>1e-13*values[0]]
    a,b=basis.conj().T@initial,basis.conj().T@desired
    u,_,vh=np.linalg.svd(b@a.conj().T+1e-12*np.eye(len(a)),full_matrices=False)
    rotation=u@vh
    residual=np.linalg.norm(initial+basis@(rotation@a-a)-desired)
    return CompactGate(basis,rotation,float(residual))


def apply_compact(state,start,width,gate):
    state.move_center(start)
    block=state.tensors[start]
    for tensor in state.tensors[start+1:start+width]:
        block=np.tensordot(block,tensor,axes=(-1,0))
    left,right=block.shape[0],block.shape[-1]
    block=gate.act(block.reshape(left,4**width,right))
    discarded=[]
    for offset in range(width-1):
        u,s,vh=np.linalg.svd(block.reshape(left*4,-1),full_matrices=False)
        rank=min(state.max_bond,max(1,int(np.sum(s>1e-13*s[0]))))
        discarded.append(float(np.sum(s[rank:]**2)))
        state.tensors[start+offset]=u[:,:rank].reshape(left,4,rank)
        block=s[:rank,None]*vh[:rank]
        left=rank
    state.tensors[start+width-1]=block.reshape(left,4,right)
    state.center=start+width-1
    normalize_center(state)
    return discarded


def doubled_marginal(state,start,length):
    state.move_center(start)
    block=state.tensors[start]
    for tensor in state.tensors[start+1:start+length]:
        block=np.tensordot(block,tensor,axes=(-1,0))
    matrix=block.reshape(block.shape[0],4**length,block.shape[-1]).transpose(1,0,2).reshape(4**length,-1)
    return matrix@matrix.conj().T


def cp_marginal_from_density(physical,start,length):
    total=int(np.log2(len(physical)))
    amplitude=base.to_interleaved(previous.small_positive_root(physical),total)
    matrix=amplitude.reshape(4**start,4**length,-1).transpose(1,0,2).reshape(4**length,-1)
    return matrix@matrix.conj().T


def onsite_update(physical,n,site,kraus):
    dimension=kraus[0].shape[0]
    support=int(np.log2(dimension))
    tensor=physical.reshape(2**site,dimension,2**(n-site-support),2**site,dimension,2**(n-site-support))
    result=np.zeros_like(tensor,dtype=np.result_type(physical,*kraus))
    for op in kraus:
        result+=np.einsum("ab,lbrmds,cd->larmcs",op,tensor,op.conj(),optimize=True)
    return result.reshape(physical.shape)


def channel_operators(channel,h):
    if channel=="X":
        p=-np.expm1(-h)/2
        return [np.sqrt(1-p)*np.eye(2),np.sqrt(p)*base.X]
    return previous.site_kraus(h,channel)


class LocalEvolution:
    def __init__(self,state,radius,buffer,physical_marginal_provider,noise_block=1):
        self.state=state
        self.radius,self.buffer=radius,buffer
        self.noise_block=noise_block
        self.period=int(np.ceil((noise_block+2*radius)/noise_block))
        self.regions={}
        self.geometry={}
        for block,site in enumerate(range(0,state.n,noise_block)):
            noisy_sites=list(range(site,min(state.n,site+noise_block)))
            start=max(0,site-radius)
            stop=min(state.n,site+noise_block+radius)
            region=(max(0,start-buffer),min(state.n,stop+buffer))
            self.geometry[block]=(start,stop,region,noisy_sites)
            if region not in self.regions:
                self.regions[region]=physical_marginal_provider(*region)
        self.regions_at_site=[[] for _ in range(state.n)]
        for region in self.regions:
            for site in range(*region):
                self.regions_at_site[site].append(region)
        self.steps=0

    def sweep(self,h,channel):
        kraus=channel_operators(channel,h)
        residuals,discarded=[],[]
        for color in range(self.period):
            blocks=list(range(color,len(self.geometry),self.period))
            active=[self.geometry[j][:2] for j in blocks]
            assert all(a[1]<=b[0] for a,b in zip(active,active[1:]))
            # Every gate in a layer is fitted to the same physical snapshot.
            gates=[]
            sites=[]
            for block in blocks:
                start,stop,region,noisy_sites=self.geometry[block]
                sites.extend(noisy_sites)
                gates.append((start,stop,fit_gate(self.regions[region],[j-region[0] for j in noisy_sites],
                             start-region[0],stop-start,kraus)))
            for start,stop,gate in gates:
                discarded.extend(apply_compact(self.state,start,stop-start,gate))
                residuals.append(gate.residual)
            for site in sites:
                for left,right in self.regions_at_site[site]:
                    region=(left,right)
                    self.regions[region]=onsite_update(self.regions[region],right-left,site-left,kraus)
        self.steps+=1
        return dict(max_local_fit=max(residuals),max_discarded=max(discarded,default=0.),
                    sum_sqrt_discarded=sum(np.sqrt(discarded)),bond=self.state.bond_dimension())


def thermal_cp(n,beta,field_z=.35,field_x=1.05,tau_step=.025,chi=24,order=2,coupling=1.):
    state=base.MPS(n,max_bond=chi)
    bell=np.array([1,0,0,1],float).reshape(1,4,1)/np.sqrt(2)
    state.tensors=[bell.copy() for _ in range(n)]
    steps=max(1,int(np.ceil(beta/2/tau_step)))
    step=beta/(2*steps)
    fields_x=np.broadcast_to(field_x,(n,))
    fields_z=np.broadcast_to(field_z,(n,))
    couplings=np.broadcast_to(coupling,(n-1,))
    if order==2:
        fractions=[1.]
    elif order==4:
        p=1/(2-2**(1/3))
        fractions=[p,1-2*p,p]
    else:
        raise ValueError("Supported imaginary-time orders are 2 and 4.")
    stages=[]
    for fraction in fractions:
        delta=step*fraction
        onsite=[np.kron(expm(delta*(gx*base.X+gz*base.Z)/2),base.I) for gx,gz in zip(fields_x,fields_z)]
        bonds=[np.diag([np.exp(delta*value*(1-2*(a//2))*(1-2*(b//2)))
                       for a in range(4) for b in range(4)]) for value in couplings]
        stages.append((onsite,bonds))
    for _ in range(steps):
        for onsite,bonds in stages:
            for j,gate in enumerate(onsite):
                state.apply_gate(j,1,gate)
                normalize_center(state)
            for parity in (0,1):
                for j in range(parity,n-1,2):
                    state.apply_gate(j,2,bonds[j])
                    normalize_center(state)
            for j,gate in enumerate(onsite):
                state.apply_gate(j,1,gate)
                normalize_center(state)
    return state


def classical_transition(coupling=.4,field=.15):
    spin=np.array([1.,-1.])
    transfer=np.exp(coupling*np.outer(spin,spin)+field*(spin[:,None]+spin[None,:])/2)
    eigen,vec=np.linalg.eigh(transfer)
    dominant=np.abs(vec[:,-1]);eigenvalue=eigen[-1]
    transition=transfer*dominant[None,:]/(eigenvalue*dominant[:,None])
    stationary=dominant**2/np.sum(dominant**2)
    assert np.max(abs(transition.sum(axis=1)-1))<1e-14
    return stationary,transition


def classical_cp(n,coupling=.4,field=.15,chi=24):
    stationary,transition=classical_transition(coupling,field)
    state=base.MPS(n,max_bond=chi)
    first=np.zeros((1,4,2))
    middle=np.zeros((2,4,2))
    last=np.zeros((2,4,1))
    for bit in range(2):
        first[0,3*bit,bit]=np.sqrt(stationary[bit])
        middle[:,3*bit,bit]=np.sqrt(transition[:,bit])
        last[:,3*bit,0]=np.sqrt(transition[:,bit])
    state.tensors=[first]+[middle.copy() for _ in range(n-2)]+[last]
    # Begin with an explicitly right-canonical representation.
    state.center=n-1
    state.move_center(0)
    normalize_center(state)
    return state


def classical_probabilities(n,coupling=.4,field=.15):
    stationary,transition=classical_transition(coupling,field)
    probabilities=stationary.copy()
    last=np.arange(2,dtype=np.int8)
    for _ in range(1,n):
        probabilities=(probabilities[:,None]*transition[last]).reshape(-1)
        last=np.tile(np.arange(2,dtype=np.int8),len(last))
    return probabilities


def bitflip_probabilities(probabilities,n,time_value):
    result=probabilities.copy()
    p=-np.expm1(-time_value)/2
    for site in range(n):
        view=result.reshape(2**site,2,-1)
        zero=view[:,0,:].copy()
        view[:,0,:]=(1-p)*zero+p*view[:,1,:]
        view[:,1,:]=p*zero+(1-p)*view[:,1,:]
    return result


def classical_cp_marginal(probabilities,n,start,length):
    matrix=np.sqrt(probabilities).reshape(2**start,2**length,-1).transpose(1,0,2).reshape(2**length,-1)
    reduced=matrix@matrix.T
    indices=np.array([sum(3*((s>>(length-1-j))&1)*4**(length-1-j) for j in range(length)) for s in range(2**length)])
    result=np.zeros((4**length,4**length))
    result[np.ix_(indices,indices)]=reduced
    return result


def csv_write(path,rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=keys)
        writer.writeheader();writer.writerows(rows)


def sanity(outdir):
    rng=np.random.default_rng(77)
    z=rng.normal(size=(16,16))+1j*rng.normal(size=(16,16))
    rho=z@z.conj().T;rho/=np.trace(rho).real
    ops=channel_operators("Y_damping",.1)
    assert np.linalg.norm(onsite_update(rho,4,2,ops)-previous.physical_update(rho,4,2,ops))<1e-13
    n=6
    state=classical_cp(n)
    target=np.sqrt(classical_probabilities(n))
    reference=base.to_interleaved(np.diag(target),n)
    assert np.linalg.norm(state.dense()-reference)<1e-12
    engine=LocalEvolution(state,1,1,lambda left,right:np.diag(classical_probabilities(right-left)))
    row=engine.sweep(.1,"X")
    exact=bitflip_probabilities(classical_probabilities(n),n,.1)
    row["dense_cp_error"]=np.linalg.norm(state.dense()-base.to_interleaved(np.diag(np.sqrt(exact)),n))
    for _,(_,_,region,_) in engine.geometry.items():
        assert np.linalg.norm(engine.regions[region]-np.diag(bitflip_probabilities(classical_probabilities(region[1]-region[0]),region[1]-region[0],.1)))<1e-12
    csv_write(outdir/"sanity.csv",[row])
    print("sanity",row,flush=True)


def classical_runs(outdir):
    coupling=.4;field=.15;length=2
    references={}
    for nref in (16,20,24):
        probability=classical_probabilities(nref,coupling,field)
        for t in (.2,.5):
            evolved=bitflip_probabilities(probability,nref,t)
            references[(nref,t)]=classical_cp_marginal(evolved,nref,nref//2,length)
    rows=[]
    for n,radius,buffer,chi in [(32,1,1,16),(64,1,1,16),(128,1,1,16),
                               (64,2,1,16),(64,2,2,16),(64,2,1,32)]:
        started=time.perf_counter()
        state=classical_cp(n,coupling,field,chi)
        provider=lambda left,right:np.diag(classical_probabilities(right-left,coupling,field))
        engine=LocalEvolution(state,radius,buffer,provider)
        for sweep in range(5):
            info=engine.sweep(.1,"X")
            if sweep+1 not in (2,5):continue
            t=round(.1*(sweep+1),8)
            probe=(n//2//engine.period)*engine.period
            gamma=doubled_marginal(state,probe,length)
            phases=[doubled_marginal(state,probe+i,length) for i in range(engine.period)]
            row=dict(n=n,model="classical hidden Markov",coupling=coupling,radius=radius,buffer=buffer,
                     chi=chi,sweep=sweep+1,time=t,probe=probe,cp_local_error=trace_distance(gamma,references[(24,t)]),
                     max_phase_error=max(trace_distance(g,references[(24,t)]) for g in phases),
                     phase_spread=max(trace_distance(g,phases[0]) for g in phases),
                     reference_16_20=trace_distance(references[(16,t)],references[(20,t)]),
                     reference_20_24=trace_distance(references[(20,t)],references[(24,t)]),
                     seconds=time.perf_counter()-started,**info)
            # End-point certificates use a large classical reference window. They
            # do not enlarge the gate or the window used to learn the gate.
            for b in (2,4,6,8,10):
                size=length+2*b
                p=bitflip_probabilities(classical_probabilities(size,coupling,field),size,t)
                local_ref=classical_cp_marginal(p,size,b,length)
                defect=trace_distance(gamma,local_ref)
                diameter=8*abs(coupling)*np.tanh(abs(coupling))**(b-1)
                tail=min(1.,np.expm1(diameter))
                row[f"certificate_b{b}"]=defect+tail
                row[f"tail_b{b}"]=tail
            rows.append(row);csv_write(outdir/"classical.csv",rows)
            print("classical",n,radius,buffer,chi,sweep+1,row["cp_local_error"],row["seconds"],flush=True)


def quantum_runs(outdir):
    rows=[]
    for n,radius,buffer,chi,tau in [(16,1,1,16,.025),(32,1,1,16,.025),
                                  (64,1,1,16,.025),(64,2,1,16,.025),
                                  (64,2,1,24,.025),(64,1,1,16,.0125)]:
        started=time.perf_counter()
        initial=thermal_cp(n,.8,chi=chi,tau_step=tau)
        state=copy.deepcopy(initial)
        exact_initial={}
        def provider(left,right):
            key=(left,right)
            if key not in exact_initial:exact_initial[key]=previous.local_density(initial,left,right)
            return exact_initial[key].copy()
        engine=LocalEvolution(state,radius,buffer,provider)
        probe=(n//2//engine.period)*engine.period;length=2
        reference_physical=previous.local_density(initial,probe,probe+length)
        references={b:previous.local_density(initial,probe-b,probe+length+b) for b in (1,2,3)}
        for sweep in range(5):
            info=engine.sweep(.1,"Z")
            ops=channel_operators("Z",.1)
            for k in range(length):reference_physical=onsite_update(reference_physical,length,k,ops)
            for b,rho in references.items():
                for k in range(length+2*b):rho=onsite_update(rho,length+2*b,k,ops)
                references[b]=rho
            if sweep+1 not in (2,5):continue
            gamma=doubled_marginal(state,probe,length)
            physical=previous.local_density(state,probe,probe+length)
            row=dict(n=n,radius=radius,buffer=buffer,chi=chi,tau=tau,sweep=sweep+1,
                     seconds=time.perf_counter()-started,physical_error=trace_distance(physical,reference_physical),**info)
            for b,rho in references.items():
                ref=cp_marginal_from_density(rho,b,length)
                row[f"local_cp_residual_b{b}"]=trace_distance(gamma,ref)
            rows.append(row);csv_write(outdir/"quantum.csv",rows)
            print("quantum",n,radius,chi,tau,sweep+1,row["physical_error"],row["local_cp_residual_b3"],row["seconds"],flush=True)


def tfim_covariance(n,beta,field=1.05):
    antisymmetric=np.zeros((2*n,2*n))
    for j in range(n):
        antisymmetric[2*j,2*j+1]=2*field
        if j<n-1:antisymmetric[2*j+1,2*j+2]=2
    antisymmetric-=antisymmetric.T
    eigen,vectors=np.linalg.eigh(1j*antisymmetric)
    covariance=1j*(vectors*np.tanh(beta*eigen/2))@vectors.conj().T
    return np.real_if_close(covariance)


def gaussian_spin_density(covariance,start,length):
    local=covariance[2*start:2*(start+length),2*start:2*(start+length)]
    values,vectors=np.linalg.eigh(-1j*local)
    kernel=(vectors*np.arctanh(np.clip(values,-1+1e-14,1-1e-14)))@vectors.conj().T
    majoranas=[]
    for site in range(length):
        for op in (base.X,base.Y):
            majoranas.append(base.kron_all([base.Z]*site+[op]+[base.I]*(length-site-1)))
    log_density=np.zeros((2**length,2**length),complex)
    for a in range(2*length):
        for b in range(a+1,2*length):
            log_density-=kernel[a,b]*(majoranas[a]@majoranas[b])
    rho=expm(log_density);rho/=np.trace(rho).real
    hadamard=base.kron_all([(base.X+base.Z)/np.sqrt(2)]*length)
    return np.real_if_close(hadamard@rho@hadamard)


def initialization_checks(outdir):
    rows=[]
    # Verify the independent Majorana construction against a dense four-site Gibbs state.
    n=4;beta=.8
    hamiltonian=np.zeros((2**n,2**n),complex)
    for j in range(n):
        hamiltonian-=1.05*base.kron_all([base.X if i==j else base.I for i in range(n)])
        if j<n-1:hamiltonian-=base.kron_all([base.Z if i in (j,j+1) else base.I for i in range(n)])
    exact=expm(-beta*hamiltonian);exact/=np.trace(exact).real
    gaussian=gaussian_spin_density(tfim_covariance(n,beta),1,2)
    assert trace_distance(gaussian,base.marginal(exact,n,[1,2]))<1e-12
    covariance=tfim_covariance(64,beta)
    rho_exact=gaussian_spin_density(covariance,30,2)
    infinite=gaussian_spin_density(tfim_covariance(256,beta),126,2)
    for field_z in (0.,.35):
        previous_density=None
        for order,tau in [(2,.025),(2,.0125),(2,.00625),(4,.025),(4,.0125)]:
            state=thermal_cp(64,beta,field_z=field_z,tau_step=tau,chi=24,order=order)
            density=previous.local_density(state,30,32)
            row=dict(n=64,beta=beta,field_z=field_z,tau=tau,order=order,
                     change_from_previous=trace_distance(density,previous_density) if previous_density is not None else None)
            if field_z==0:
                row["exact_gaussian_error"]=trace_distance(density,rho_exact)
                row["finite_size_64_256"]=trace_distance(rho_exact,infinite)
            rows.append(row);previous_density=density
            csv_write(outdir/"initialization.csv",rows)
            print("initialization",row,flush=True)


def precise_quantum(outdir):
    rows=[]
    for n,disorder in [(64,False),(128,False),(64,True)]:
        started=time.perf_counter()
        params=dict(n=n,beta=.8,tau_step=.025,order=4,chi=24)
        if disorder:
            rng=np.random.default_rng(20261103)
            params.update(field_x=1.05+.2*rng.uniform(-1,1,n),
                          field_z=.35+.2*rng.uniform(-1,1,n),
                          coupling=1+.15*rng.uniform(-1,1,n-1))
        initial=thermal_cp(**params)
        state=copy.deepcopy(initial)
        engine=LocalEvolution(state,2,1,lambda left,right:previous.local_density(initial,left,right))
        probe=(n//2//5)*5;length=2
        physical=previous.local_density(initial,probe,probe+length)
        contexts={b:previous.local_density(initial,probe-b,probe+length+b) for b in (2,3)}
        ops=channel_operators("Z",.1)
        for sweep in range(10):
            info=engine.sweep(.1,"Z")
            for j in range(length):physical=onsite_update(physical,length,j,ops)
            for b,rho in contexts.items():
                for j in range(length+2*b):rho=onsite_update(rho,length+2*b,j,ops)
                contexts[b]=rho
            if sweep+1 not in (2,5,10):continue
            gamma=doubled_marginal(state,probe,length)
            row=dict(n=n,disorder=disorder,sweep=sweep+1,chi=24,radius=2,buffer=1,
                     initialization_order=4,tau=.025,seconds=time.perf_counter()-started,
                     physical_error=trace_distance(previous.local_density(state,probe,probe+length),physical),**info)
            for b,rho in contexts.items():row[f"local_cp_residual_b{b}"]=trace_distance(gamma,cp_marginal_from_density(rho,b,length))
            rows.append(row);csv_write(outdir/"precise_quantum.csv",rows)
            print("precise",row,flush=True)


def scaling_checks(outdir):
    rows=[]
    p=bitflip_probabilities(classical_probabilities(24),24,.2)
    reference=classical_cp_marginal(p,24,12,2)
    cert_p=bitflip_probabilities(classical_probabilities(22),22,.2)
    cert_ref=classical_cp_marginal(cert_p,22,10,2)
    tail=np.expm1(8*.4*np.tanh(.4)**9)
    for radius in (1,2):
        for n in (32,64,128,256):
            started=time.perf_counter()
            state=classical_cp(n,chi=16)
            engine=LocalEvolution(state,radius,1,lambda left,right:np.diag(classical_probabilities(right-left)))
            setup=time.perf_counter()-started
            started=time.perf_counter()
            for _ in range(2):info=engine.sweep(.1,"X")
            elapsed=time.perf_counter()-started
            probe=(n//2//engine.period)*engine.period
            gamma=doubled_marginal(state,probe,2)
            row=dict(n=n,radius=radius,buffer=1,chi=16,sweeps=2,setup_seconds=setup,
                     update_seconds=elapsed,seconds_per_site=elapsed/n,
                     cp_local_error=trace_distance(gamma,reference),
                     certified_bound=trace_distance(gamma,cert_ref)+tail,**info)
            assert row["cp_local_error"]<=row["certified_bound"]+1e-12
            rows.append(row);csv_write(outdir/"scaling.csv",rows)
            print("scaling",row,flush=True)


class ProductCircuitOracle:
    """Exact physical marginals from the backward cone of a finite-range circuit."""
    def __init__(self,n,one_site):
        self.n=n
        self.one_site=one_site
        self.layers=[]
        self.cache={}
        self.largest_patch=0

    def append_layer(self,sites,operators):
        width=int(np.log2(operators[0].shape[0]))
        assert all(a+width<=b for a,b in zip(sites,sites[1:]))
        self.layers.append((width,{site:operators for site in sites}))

    def marginal(self,left,right):
        patch_left,patch_right=left,right
        selected=[]
        for width,events in reversed(self.layers):
            intersecting=[(site,events[site]) for site in range(max(0,patch_left-width+1),patch_right)
                          if site in events]
            selected.extend(intersecting)
            for site,_ in intersecting:
                patch_left=min(patch_left,site)
                patch_right=max(patch_right,site+width)
        selected.reverse()
        size=patch_right-patch_left
        self.largest_patch=max(self.largest_patch,size)
        # The product input and channel are homogeneous in this benchmark. The
        # translated cone signature therefore identifies exactly equal marginals.
        key=(size,left-patch_left,right-left,tuple((j-patch_left,k[0].tobytes(),k[1].tobytes()) for j,k in selected))
        if key not in self.cache:
            physical=base.kron_all([self.one_site]*size)
            for site,operators in selected:physical=onsite_update(physical,size,site-patch_left,operators)
            self.cache[key]=base.marginal(physical,size,range(left-patch_left,right-patch_left))
        return self.cache[key].copy()


def interacting_noise(outdir):
    rows=[]
    one_site=(base.I+.45*base.X+.2*base.Y+.25*base.Z)/2
    local_root=previous.small_positive_root(one_site)
    p=-np.expm1(-.2)
    no_jump=np.diag([1.,1.,1.,np.sqrt(1-p)])
    jump=np.zeros((4,4));jump[0,3]=np.sqrt(p)
    operators=[no_jump,jump]
    assert np.linalg.norm(sum(k.conj().T@k for k in operators)-np.eye(4))<1e-14
    for n in (8,32,64):
        started=time.perf_counter()
        state=base.MPS(n,max_bond=24)
        state.tensors=[base.to_interleaved(local_root,1).reshape(1,4,1).copy() for _ in range(n)]
        oracle=ProductCircuitOracle(n,one_site)
        fits=[];discarded=[]
        exact=base.kron_all([one_site]*n) if n==8 else None
        # Even then odd physical bonds, with two disjoint CP-gate colors each.
        for parity in (0,1):
            for color in (0,1):
                sites=list(range(parity+2*color,n-1,4))
                gates=[]
                for site in sites:
                    start=max(0,site-1);stop=min(n,site+3)
                    left=max(0,start-1);right=min(n,stop+1)
                    rho=oracle.marginal(left,right)
                    gate=fit_gate(rho,[site-left],start-left,stop-start,operators)
                    gates.append((start,stop,gate));fits.append(gate.residual)
                assert all(a[1]<=b[0] for a,b in zip(gates,gates[1:]))
                for start,stop,gate in gates:discarded.extend(apply_compact(state,start,stop-start,gate))
                for site in sites:
                    if exact is not None:exact=onsite_update(exact,n,site,operators)
                oracle.append_layer(sites,operators)
        probe=(n//2//4)*4
        if probe+2>n:probe-=4
        gamma=doubled_marginal(state,probe,2)
        local_physical=oracle.marginal(probe,probe+2)
        row=dict(n=n,noise="pair loss 11 to 00",cp_layers=4,chi=24,
                 physical_error=trace_distance(previous.local_density(state,probe,probe+2),local_physical),
                 max_local_fit=max(fits),max_discarded=max(discarded,default=0.),bond=state.bond_dimension())
        for buffer in (1,2):
            left=max(0,probe-buffer);right=min(n,probe+2+buffer)
            reference=cp_marginal_from_density(oracle.marginal(left,right),probe-left,2)
            row[f"local_cp_residual_b{buffer}"]=trace_distance(gamma,reference)
        if exact is not None:
            assert trace_distance(local_physical,base.marginal(exact,n,range(probe,probe+2)))<1e-12
            root=previous.small_positive_root(exact)
            full_ref=base.to_interleaved(root,n).reshape(4**probe,16,-1).transpose(1,0,2).reshape(16,-1)
            row["exact_cp_local_error"]=trace_distance(gamma,full_ref@full_ref.conj().T)
            row["global_cp_vector_error"]=np.linalg.norm(state.dense()-base.to_interleaved(root,n))
        row.update(seconds=time.perf_counter()-started,largest_oracle_patch=oracle.largest_patch)
        rows.append(row);csv_write(outdir/"interacting_noise.csv",rows)
        print("interacting",row,flush=True)


def finite_jump(outdir):
    rows=[]
    for model,n in [("classical",128),("quantum",64)]:
        if model=="classical":
            initial=classical_cp(n,chi=24)
            provider=lambda left,right:np.diag(classical_probabilities(right-left))
            channel="X"
        else:
            initial=thermal_cp(n,.8,order=4,tau_step=.025,chi=24)
            provider=lambda left,right:previous.local_density(initial,left,right)
            channel="Z"
        for elapsed in (.5,1.):
            started=time.perf_counter()
            state=copy.deepcopy(initial)
            engine=LocalEvolution(state,2,1,provider)
            info=engine.sweep(elapsed,channel)
            probe=(n//2//5)*5
            gamma=doubled_marginal(state,probe,2)
            row=dict(model=model,n=n,elapsed=elapsed,sweeps=1,chi=24,radius=2,buffer=1,
                     seconds=time.perf_counter()-started,**info)
            if model=="classical":
                p=bitflip_probabilities(classical_probabilities(24),24,elapsed)
                reference=classical_cp_marginal(p,24,12,2)
                row["local_cp_error"]=trace_distance(gamma,reference)
                cert_p=bitflip_probabilities(classical_probabilities(22),22,elapsed)
                row["certified_bound"]=trace_distance(gamma,classical_cp_marginal(cert_p,22,10,2))+np.expm1(3.2*np.tanh(.4)**9)
            else:
                ops=channel_operators(channel,elapsed)
                physical=provider(probe,probe+2)
                for j in range(2):physical=onsite_update(physical,2,j,ops)
                reference=provider(probe-3,probe+5)
                for j in range(8):reference=onsite_update(reference,8,j,ops)
                row["physical_error"]=trace_distance(previous.local_density(state,probe,probe+2),physical)
                row["local_cp_residual_b3"]=trace_distance(gamma,cp_marginal_from_density(reference,3,2))
            rows.append(row);csv_write(outdir/"finite_jump.csv",rows)
            print("jump",row,flush=True)


def block_runs(outdir):
    rows=[]
    for model,n,blocks,guard,buffer,chi in [
        ("classical",64,2,1,1,16),("classical",128,3,1,1,16),
        ("quantum",64,2,1,1,24),("quantum",64,3,1,1,24)]:
        started=time.perf_counter()
        if model=="classical":
            initial=classical_cp(n,chi=chi)
            provider=lambda left,right:np.diag(classical_probabilities(right-left))
            channel="X"
        else:
            initial=thermal_cp(n,.8,tau_step=.00625,chi=chi)
            provider=lambda left,right:previous.local_density(initial,left,right)
            channel="Z"
        state=copy.deepcopy(initial)
        engine=LocalEvolution(state,guard,buffer,provider,noise_block=blocks)
        period=engine.period*blocks
        probe=(n//2//period)*period;length=2
        if model=="quantum":
            physical=previous.local_density(initial,probe,probe+length)
            context=previous.local_density(initial,probe-3,probe+length+3)
        for step in range(5):
            info=engine.sweep(.1,channel)
            if model=="quantum":
                ops=channel_operators(channel,.1)
                for j in range(length):physical=onsite_update(physical,length,j,ops)
                for j in range(length+6):context=onsite_update(context,length+6,j,ops)
            if step+1 not in (2,5):continue
            gamma=doubled_marginal(state,probe,length)
            row=dict(model=model,n=n,noise_block=blocks,guard=guard,buffer=buffer,chi=chi,layers=engine.period,
                     sweep=step+1,seconds=time.perf_counter()-started,**info)
            if model=="classical":
                p=bitflip_probabilities(classical_probabilities(24),24,.1*(step+1))
                reference=classical_cp_marginal(p,24,12,length)
                phases=[doubled_marginal(state,probe+j,length) for j in range(period)]
                row["local_cp_error"]=trace_distance(gamma,reference)
                row["max_phase_error"]=max(trace_distance(g,reference) for g in phases)
                row["phase_spread"]=max(trace_distance(g,gamma) for g in phases)
            else:
                row["physical_error"]=trace_distance(previous.local_density(state,probe,probe+length),physical)
                row["local_cp_residual_b3"]=trace_distance(gamma,cp_marginal_from_density(context,3,length))
            rows.append(row);csv_write(outdir/"block_updates.csv",rows)
            print("block",row,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part",choices=["sanity","classical","quantum","initialization","blocks","precise","scaling","interacting","jump","all"],default="all")
    parser.add_argument("--outdir",type=Path,default=Path(__file__).parent/"large_results")
    args=parser.parse_args();args.outdir.mkdir(parents=True,exist_ok=True)
    if args.part in ("all","sanity"):sanity(args.outdir)
    if args.part in ("all","classical"):classical_runs(args.outdir)
    if args.part in ("all","quantum"):quantum_runs(args.outdir)
    if args.part in ("all","initialization"):initialization_checks(args.outdir)
    if args.part in ("all","blocks"):block_runs(args.outdir)
    if args.part in ("all","precise"):precise_quantum(args.outdir)
    if args.part in ("all","scaling"):scaling_checks(args.outdir)
    if args.part in ("all","interacting"):interacting_noise(args.outdir)
    if args.part in ("all","jump"):finite_jump(args.outdir)

if __name__=="__main__":main()
