"""Finite local Kraus step with buffered reference polar compression.

Maximizes Re Tr(Q_new) over a rectangular reference map supported in W.
It is a proposed rank-opening bridge, not a TDVP time integrator. Retained
norm alone does not certify the canonical branch.
"""
import tdvp_core as tn
import mpo_cp_force as mpo
import copy
import numpy as np
from scipy.linalg import expm


def kraus(kind,rate,time):
    if kind=='loss':
        probability=-np.expm1(-rate*time)
        return [np.diag([1.,np.sqrt(1-probability)]),np.sqrt(probability)*mpo.SM]
    if kind=='phase':
        probability=-np.expm1(-2*rate*time)/2
        return [np.sqrt(1-probability)*mpo.I,np.sqrt(probability)*mpo.Z]
    raise ValueError(kind)


def embed_local(op,site,width):
    result=np.ones((1,1),complex)
    for j in range(width):result=np.kron(result,op if j==site else mpo.I)
    return result


def block_to_operator(block,width):
    dl,dr=block.shape[0],block.shape[-1]
    order=[0]+[1+2*j for j in range(width)]+[2+2*j for j in range(width)]+[2*width+1]
    return block.reshape([dl]+[2,2]*width+[dr]).transpose(order).reshape(dl,2**width,2**width,dr)


def operator_to_block(block,width):
    dl,_,_,dr=block.shape
    order=[0]+[x for j in range(width) for x in (1+j,1+width+j)]+[2*width+1]
    return block.reshape([dl]+[2]*2*width+[dr]).transpose(order).reshape([dl]+[4]*width+[dr])


def trace_outside(state,start,width):
    left=np.ones(1,complex);right=np.ones(1,complex)
    for a in state.tensors[:start]:left=left@(a[:,0,:]+a[:,3,:])
    for a in state.tensors[start+width:][::-1]:right=(a[:,0,:]+a[:,3,:])@right
    return left,right


def replace_block(state,start,block,width,chi,normalize=True,inplace=False):
    out=state if inplace else copy.deepcopy(state);out.max_bond=chi;discard=[];dl=block.shape[0]
    for j in range(width-1):
        u,s,vh=np.linalg.svd(block.reshape(dl*4,-1),full_matrices=False)
        rank=max(1,min(chi,int(sum(s>1e-13*max(s[0],1e-300)))))
        discard.append(float(sum(s[rank:]**2)))
        out.tensors[start+j]=u[:,:rank].reshape(dl,4,rank)
        block=s[:rank,None]*vh[:rank];dl=rank
    out.tensors[start+width-1]=block.reshape(dl,4,-1);out.center=start+width-1
    length=float(np.linalg.norm(out.tensors[out.center]))
    if normalize:out.tensors[out.center]/=length
    return out,dict(rounding_bound=float(sum(np.sqrt(discard))),norm_after_rounding=length)


def local_step(state,site,operators,buffer=1,chi=16,hermitian=False,environments=None,inplace=False):
    start=max(0,site-buffer);stop=min(state.n,site+buffer+1);width=stop-start
    current=state if inplace else copy.deepcopy(state);current.move_center(start)
    block=current.tensors[start]
    for a in current.tensors[start+1:stop]:block=np.tensordot(block,a,axes=(-1,0))
    operator_block=block_to_operator(block,width)
    left,right=trace_outside(current,start,width) if environments is None else environments
    partial=np.einsum('a,aijb,b->ij',left,operator_block,right,optimize=True)
    ks=[embed_local(k,site-start,width) for k in operators]
    small=np.column_stack([k@partial for k in ks]);scale=np.linalg.norm(small)
    if scale<1e-300:raise RuntimeError('The trace environment vanished.')
    u,s,vh=np.linalg.svd(small/scale,full_matrices=False)
    keep=s>1e-13*max(s[0],1e-300)
    # Partial polar isometry: arbitrary null-space completions are omitted.
    reference=vh[keep].conj().T@u[:,keep].conj().T
    d=2**width;result=np.zeros_like(operator_block,dtype=complex)
    for mu,k in enumerate(ks):
        v=reference[mu*d:(mu+1)*d]
        result+=np.einsum('pi,aiqb,qj->apjb',k,operator_block,v,optimize=True)
    norm_before=float(np.linalg.norm(operator_block)**2);retained=float(np.linalg.norm(result)**2)
    loss=max(0.,norm_before-retained)
    correction=0.
    if hermitian:
        # For Hermitian input Q, the adjoint of K Q V is V^dag Q K^dag.
        # Both terms act on the same finite window; no global adjoint sweep.
        conjugate=np.zeros_like(result)
        for mu,k in enumerate(ks):
            v=reference[mu*d:(mu+1)*d]
            conjugate+=np.einsum('pi,aiqb,qj->apjb',v.conj().T,operator_block,k.conj().T,optimize=True)
        symmetric=(result+conjugate)/2
        correction=float(np.linalg.norm(symmetric/np.linalg.norm(symmetric)-result/np.linalg.norm(result)))
        result=symmetric
    out,info=replace_block(current,start,operator_to_block(result,width),width,chi,inplace=inplace)
    info['hermitian_correction']=correction
    info['density_step_bound']=loss/max(norm_before,1e-300)+correction+2*info['rounding_bound']/max(np.linalg.norm(result),1e-300)
    info.update(buffer=buffer,width=width,kraus_loss=loss,retained_norm_squared=retained,
                polar_rank=int(sum(keep)),trace_objective=float(sum(s)*scale))
    return out,info


def noise_sweep(state,operators,buffer,chi,hermitian=False):
    """Reuse trace environments in a left-to-right sweep at fixed buffer."""
    state.move_center(0);n=state.n;right=[None]*(n+1);right[n]=np.ones(1,complex)
    for j in range(n-1,-1,-1):right[j]=(state.tensors[j][:,0,:]+state.tensors[j][:,3,:])@right[j+1]
    left=np.ones(1,complex);previous_start=0;records=[]
    for j in range(n):
        start=max(0,j-buffer);stop=min(n,j+buffer+1)
        # Moving the center right can change the tensor about to enter the prefix.
        state.move_center(start)
        for k in range(previous_start,start):left=left@(state.tensors[k][:,0,:]+state.tensors[k][:,3,:])
        state,info=local_step(state,j,operators[j],buffer,chi,hermitian,(left,right[stop]),inplace=True)
        records.append(info);previous_start=start
    return state,records


def reverse_chain(state):
    out=copy.deepcopy(state);out.tensors=[a.transpose(2,1,0) for a in reversed(out.tensors)]
    out.center=state.n-1-state.center
    return out


def trace_sweep(state,buffer,chi):
    """Reference-only unitary polar updates; preserve density before rounding."""
    state.move_center(0);n=state.n;right=[None]*(n+1);right[n]=np.ones(1,complex)
    for j in range(n-1,-1,-1):right[j]=(state.tensors[j][:,0,:]+state.tensors[j][:,3,:])@right[j+1]
    left=np.ones(1,complex);previous_start=0;gates=0;rounding=0.;largest_gain=0.
    for j in range(n):
        start=max(0,j-buffer);stop=min(n,j+buffer+1);width=stop-start
        state.move_center(start)
        for k in range(previous_start,start):left=left@(state.tensors[k][:,0,:]+state.tensors[k][:,3,:])
        block=state.tensors[start]
        for a in state.tensors[start+1:stop]:block=np.tensordot(block,a,axes=(-1,0))
        block=block_to_operator(block,width)
        c=np.einsum('a,aijb,b->ij',left,block,right[stop],optimize=True);c/=max(np.linalg.norm(c),1e-300)
        u,s,vh=np.linalg.svd(c,full_matrices=False);gain=max(0.,float(sum(s)-np.trace(c).real));largest_gain=max(gain,largest_gain)
        if gain>1e-14:
            v=vh.conj().T@u.conj().T
            rotated=np.einsum('aiqb,qj->aijb',block,v,optimize=True)
            state,info=replace_block(state,start,operator_to_block(rotated,width),width,chi,inplace=True)
            rounding+=info['rounding_bound'];gates+=1
        previous_start=start
    return state,dict(gauge_gates=gates,gauge_rounding=rounding,max_trace_gain=largest_gain)


def trace_optimize(state,buffer,chi,sweeps=2):
    records=[]
    for sweep in range(sweeps):
        if sweep%2:state=reverse_chain(state)
        state,record=trace_sweep(state,buffer,chi);records.append(record)
        if sweep%2:state=reverse_chain(state)
    return state,records


def automatic_local_step(state,site,operators,cp_tolerance=1e-4,max_buffer=3,chi=16):
    previous=None;records=[]
    for buffer in range(max_buffer+1):
        out,info=local_step(state,site,operators,buffer,chi)
        change=tn.distance(out,previous) if previous is not None else float('inf')
        info['buffer_change']=change;records.append(info)
        # This is an operational acceptance rule, not a certified CP bound.
        if change<=cp_tolerance and info['kraus_loss']<=cp_tolerance**2/2:
            return out,dict(**info,accepted=True,attempts=len(records),canonical_certified=False)
        previous=out
    return out,dict(**info,accepted=False,attempts=len(records),canonical_certified=False)


def hamiltonian_step(state,time,chi,jzz=.7,hx=.43,hz=.19,normalize=True):
    out=copy.deepcopy(state);out.max_bond=chi;out.discarded_weights=[]
    u=expm(-1j*time/2*(hx*mpo.X+hz*mpo.Z));onsite=np.kron(u,u.conj())
    for j in range(state.n):out.apply_gate(j,1,onsite)
    u=expm(-1j*time*jzz*np.kron(mpo.Z,mpo.Z));gate=np.kron(u,u.conj())
    order=np.arange(16).reshape([2]*4).transpose(0,2,1,3).reshape(-1);gate=gate[np.ix_(order,order)]
    for j in range(state.n-1):out.apply_gate(j,2,gate)
    for j in range(state.n):out.apply_gate(j,1,onsite)
    if normalize:out.tensors[out.center]/=np.linalg.norm(out.tensors[out.center])
    return out,float(sum(np.sqrt(out.discarded_weights)))


def dense_polar_channel(root,site,operators,n):
    factors=np.column_stack([embed_local(k,site,n)@root for k in operators])
    u,s,_=np.linalg.svd(factors,full_matrices=False)
    return (u*s)@u.conj().T


def run(initial,elapsed,dt,buffer=1,chi=16,kind='loss',hamiltonian=True,dense_reference=False,hermitian=False,gauge_sweeps=0,gauge_buffer=None):
    state=copy.deepcopy(initial);n=state.n;steps=round(elapsed/dt);dt=elapsed/steps
    rates=.3+.12*np.cos(np.arange(n))**2 if kind=='loss' else np.full(n,.5)
    ref=tn.to_matrix(initial.dense(),n) if dense_reference else None
    reference_state=copy.deepcopy(initial) if dense_reference else None
    history=[];losses=[];rounding=[];gauge_gates=0;gauge_records=[];physical_bound=0.
    for step in range(steps):
        if hamiltonian:
            state,e=hamiltonian_step(state,dt/2,chi);rounding.append(e);physical_bound+=2*e
            if dense_reference:
                reference_state=tn.from_dense(tn.to_interleaved(ref,n),n)
                reference_state,_=hamiltonian_step(reference_state,dt/2,4**(n//2))
                ref=tn.to_matrix(reference_state.dense(),n)
        channel=[kraus(kind,rates[j],dt) for j in range(n)]
        state,records=noise_sweep(state,channel,buffer,chi,hermitian)
        for j,info in enumerate(records):
            losses.append(info['kraus_loss']);rounding.append(info['rounding_bound'])
            physical_bound+=info['density_step_bound']
            if dense_reference:ref=dense_polar_channel(ref,j,channel[j],n)
        if hamiltonian:
            state,e=hamiltonian_step(state,dt/2,chi);rounding.append(e);physical_bound+=2*e
            if dense_reference:
                reference_state=tn.from_dense(tn.to_interleaved(ref,n),n)
                reference_state,_=hamiltonian_step(reference_state,dt/2,4**(n//2))
                ref=tn.to_matrix(reference_state.dense(),n)
        if gauge_sweeps:
            state,records=trace_optimize(state,buffer if gauge_buffer is None else gauge_buffer,chi,gauge_sweeps)
            for record in records:rounding.append(record['gauge_rounding']);gauge_gates+=record['gauge_gates'];physical_bound+=2*record['gauge_rounding']
            gauge_records+=records
        row=dict(time=(step+1)*dt,max_loss=max(losses,default=0),sum_loss=sum(losses),sum_rounding=sum(rounding),
                 hermiticity=tn.distance(state,mpo.adjoint(state)),bond=state.bond_dimension(),gauge_gates=gauge_gates,
                 last_trace_gain=gauge_records[-1]['max_trace_gain'] if gauge_records else 0.,density_error_bound=min(1.,physical_bound))
        if dense_reference:
            a=tn.to_matrix(state.dense(),n);u,s,_=np.linalg.svd(a,full_matrices=False);own=(u*s)@u.conj().T
            row.update(cp_error=np.linalg.norm(a-ref),branch_error=np.linalg.norm(a-own),
                       density_error=np.linalg.norm(a@a.conj().T-ref@ref),min_hermitian_eigenvalue=min(np.linalg.eigvalsh((a+a.conj().T)/2)))
            delta=a@a.conj().T-ref@ref;row['density_trace_distance']=float(sum(abs(np.linalg.eigvalsh((delta+delta.conj().T)/2)))/2)
            assert row['density_trace_distance']<=physical_bound+1e-10
        history.append(row)
    return state,history,ref
