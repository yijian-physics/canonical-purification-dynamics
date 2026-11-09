"""MPS tangent projection and retracted explicit-midpoint TDVP.

Physical site dimension 4, pairing a spin with its canonical reference.
No physical gate is optimized. The force provider is a separate interface.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='1'
import copy
import numpy as np
from cluster_dephasing import MPS, to_interleaved, to_matrix


def scale(a,c):
    out=copy.deepcopy(a);out.tensors[0]=c*out.tensors[0];return out


def add(a,b,alpha=1.,beta=1.):
    assert a.n==b.n
    out=MPS(a.n,max_bond=a.max_bond+b.max_bond)
    if a.n==1:out.tensors=[alpha*a.tensors[0]+beta*b.tensors[0]];return out
    out.tensors=[np.concatenate([alpha*a.tensors[0],beta*b.tensors[0]],axis=2)]
    for x,y in zip(a.tensors[1:-1],b.tensors[1:-1]):
        z=np.zeros((x.shape[0]+y.shape[0],4,x.shape[2]+y.shape[2]),complex)
        z[:x.shape[0],:,:x.shape[2]]=x;z[x.shape[0]:,:,x.shape[2]:]=y
        out.tensors.append(z)
    out.tensors.append(np.concatenate([a.tensors[-1],b.tensors[-1]],axis=0));out.center=a.n-1
    return out


def norm(a):
    out=copy.deepcopy(a);out.center=out.n-1;out.move_center(0)
    return float(np.linalg.norm(out.tensors[0]))


def distance(a,b):return norm(add(a,b,beta=-1.))


def overlap(a,b):
    env=np.ones((1,1),complex)
    for x,y in zip(a.tensors,b.tensors):env=np.einsum('ab,asr,bsu->ru',env,x.conj(),y,optimize=True)
    return env.item()


def compress(a,chi,tolerance=1e-13,normalize=False):
    out=copy.deepcopy(a);out.max_bond=chi;out.center=out.n-1;out.move_center(0)
    discarded=[]
    for j in range(out.n-1):
        x=out.tensors[j];u,s,vh=np.linalg.svd(x.reshape(x.shape[0]*4,x.shape[2]),full_matrices=False)
        rank=max(1,min(chi,int(np.sum(s>tolerance*s[0]))))
        discarded.append(float(np.sum(s[rank:]**2)))
        out.tensors[j]=u[:,:rank].reshape(x.shape[0],4,rank)
        out.tensors[j+1]=np.tensordot(s[:rank,None]*vh[:rank],out.tensors[j+1],axes=(1,0))
    out.center=out.n-1
    length=float(np.linalg.norm(out.tensors[-1]))
    if normalize:out.tensors[-1]/=length
    return out,dict(discarded_bound=float(sum(np.sqrt(discarded))),norm_before=length)


def from_dense(vector,n,chi=None):
    chi=chi or 4**(n//2);out=MPS(n,max_bond=chi);out.tensors=[];block=vector.reshape(1,-1);dl=1
    for j in range(n-1):
        u,s,vh=np.linalg.svd(block.reshape(dl*4,-1),full_matrices=False)
        rank=max(1,min(chi,int(np.sum(s>1e-14*s[0]))))
        out.tensors.append(u[:,:rank].reshape(dl,4,rank));block=s[:rank,None]*vh[:rank];dl=rank
    out.tensors.append(block.reshape(dl,4,1));out.center=n-1
    return out


def vacuum(n):
    out=MPS(n,max_bond=1);v=np.array([1.,0,0,0],complex).reshape(1,4,1)
    out.tensors=[v.copy() for _ in range(n)];return out


def dyad(ket,bra,occupied=1.,vacuum_amplitude=0.):
    """Vectorize occupied |ket><bra| plus a vacuum projector, bond <=4."""
    n=len(ket);out=MPS(n,max_bond=4);out.tensors=[]
    for j,(a,b) in enumerate(zip(ket,bra)):
        x=np.zeros((4,4,4),complex);x[:,0,:]=np.eye(4)
        x[0,2,1]=a;x[2,2,3]=a;x[0,1,2]=b.conjugate();x[1,1,3]=b.conjugate()
        x[0,3,3]=a*b.conjugate()
        if j==0:x=x[0:1]
        if j==n-1:x=np.einsum('lsr,r->ls',x,np.array([vacuum_amplitude,0,0,occupied]))[:,:,None]
        out.tensors.append(x)
    out.center=n-1;out.move_center(0);return out


def tangent_project(state,force):
    """Full one-site MPS tangent projector, using an orthogonal gauge.

    The sum of all tensor variations is represented by a bond-2*chi MPS.
    This equals the usual sum of site projectors minus bond projectors.
    """
    right=copy.deepcopy(state);right.center=right.n-1;right.move_center(0)
    left=copy.deepcopy(right);left.move_center(left.n-1)
    n=state.n;renv=[None]*(n+1);renv[n]=np.ones((1,1),complex)
    for j in range(n-1,0,-1):
        renv[j]=np.einsum('ru,asr,bsu->ab',renv[j+1],right.tensors[j].conj(),force.tensors[j],optimize=True)
    lenv=np.ones((1,1),complex);variations=[]
    for j in range(n):
        f=np.einsum('al,lsr,br->asb',lenv,force.tensors[j],renv[j+1],optimize=True)
        if j<n-1:
            al=left.tensors[j].reshape(-1,left.tensors[j].shape[-1]);flat=f.reshape(al.shape[0],-1)
            f=(flat-al@(al.conj().T@flat)).reshape(f.shape)
        variations.append(f)
        if j<n-1:lenv=np.einsum('ab,asr,bsu->ru',lenv,left.tensors[j].conj(),force.tensors[j],optimize=True)
    out=MPS(n,max_bond=2*state.max_bond)
    if n==1:out.tensors=variations;return out
    out.tensors=[np.concatenate([left.tensors[0],variations[0]],axis=2)]
    for j in range(1,n-1):
        a,b,c=left.tensors[j],variations[j],right.tensors[j]
        x=np.zeros((a.shape[0]+c.shape[0],4,a.shape[2]+c.shape[2]),complex)
        x[:a.shape[0],:,:a.shape[2]]=a;x[:a.shape[0],:,a.shape[2]:]=b;x[a.shape[0]:,:,a.shape[2]:]=c
        out.tensors.append(x)
    out.tensors.append(np.concatenate([variations[-1],right.tensors[-1]],axis=0));out.center=n-1
    return out


def tangent_project_two(state,force):
    """Two-site tangent-space projector with room for bond growth.

    A three-state virtual automaton sums the gauged two-site variations in
    one MPS of bond at most 6*chi, without constructing a dense global vector.
    """
    n=state.n
    if n<2:return tangent_project(state,force)
    right=copy.deepcopy(state);right.center=n-1;right.move_center(0)
    left=copy.deepcopy(right);left.move_center(n-1)
    renv=[None]*(n+1);renv[n]=np.ones((1,1),complex)
    for j in range(n-1,1,-1):
        renv[j]=np.einsum('ru,asr,bsu->ab',renv[j+1],right.tensors[j].conj(),force.tensors[j],optimize=True)
    lenv=np.ones((1,1),complex);first=[];second=[]
    for j in range(n-1):
        block=np.tensordot(force.tensors[j],force.tensors[j+1],axes=(-1,0))
        f=np.einsum('al,lstr,br->astb',lenv,block,renv[j+2],optimize=True)
        flat=f.reshape(f.shape[0]*4,-1)
        if j<n-2:
            al=left.tensors[j].reshape(flat.shape[0],-1);flat-=al@(al.conj().T@flat)
        u,s,vh=np.linalg.svd(flat,full_matrices=False)
        rank=max(1,int(np.sum(s>1e-13*max(s[0],1e-300))))
        first.append(u[:,:rank].reshape(f.shape[0],4,rank))
        second.append((s[:rank,None]*vh[:rank]).reshape(rank,4,f.shape[-1]))
        if j<n-2:lenv=np.einsum('ab,asr,bsu->ru',lenv,left.tensors[j].conj(),force.tensors[j],optimize=True)
    out=MPS(n,max_bond=6*state.max_bond);out.tensors=[]
    out.tensors.append(np.concatenate([left.tensors[0],first[0],np.zeros_like(right.tensors[0])],axis=2))
    for j in range(1,n-1):
        al,ar=left.tensors[j],right.tensors[j];before_l, before_r=al.shape[0],al.shape[-1]
        mid_l,mid_r=second[j-1].shape[0],first[j].shape[-1]
        after_l,after_r=ar.shape[0],ar.shape[-1]
        x=np.zeros((before_l+mid_l+after_l,4,before_r+mid_r+after_r),complex)
        x[:before_l,:,:before_r]=al
        x[:before_l,:,before_r:before_r+mid_r]=first[j]
        x[before_l:before_l+mid_l,:,before_r+mid_r:]=second[j-1]
        x[before_l+mid_l:,:,before_r+mid_r:]=ar
        out.tensors.append(x)
    out.tensors.append(np.concatenate([np.zeros_like(left.tensors[-1]),second[-1],right.tensors[-1]],axis=0))
    out.center=n-1;return out


def midpoint_step(state,u,h,force_provider,chi,retract=None,mode='one'):
    """Integrate dPsi/du=P_T F(u,Psi), all tensors variable."""
    retract=retract or (lambda x:compress(x,chi,normalize=True))
    projector=tangent_project if mode=='one' else tangent_project_two
    force,info1=force_provider(u,state);k1=projector(state,force)
    k1=add(k1,state,beta=-overlap(state,force).real)
    mid,rm=retract(add(state,k1,beta=h/2))
    force,info2=force_provider(u+h/2,mid);k2=projector(mid,force)
    k2=add(k2,mid,beta=-overlap(mid,force).real)
    end,re=retract(add(state,k2,beta=h))
    return end,dict(force1=info1,force2=info2,mid_retraction=rm,end_retraction=re)
