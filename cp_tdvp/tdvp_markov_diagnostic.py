"""Separate local-unitary, unrestricted tensor, TDVP tangent, and CMI tests.

Exact CP targets/derivatives are supplied as oracles. This is not a closed,
integrated TDVP algorithm for generic Lindblad dynamics.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='1'
import argparse
import copy
from pathlib import Path
import numpy as np
import single_excitation_study as study
base=study.base


def difference_norm(a,b):
    """Contract the difference itself to avoid subtracting near-unit overlaps."""
    n=a.n;c=base.MPS(n,max_bond=a.max_bond+b.max_bond)
    if n==1:c.tensors=[a.tensors[0]-b.tensors[0]];return c.norm()
    tensors=[np.concatenate([a.tensors[0],-b.tensors[0]],axis=2)]
    for x,y in zip(a.tensors[1:-1],b.tensors[1:-1]):
        z=np.zeros((x.shape[0]+y.shape[0],4,x.shape[2]+y.shape[2]),complex)
        z[:x.shape[0],:,:x.shape[2]]=x;z[x.shape[0]:,:,x.shape[2]:]=y
        tensors.append(z)
    tensors.append(np.concatenate([a.tensors[-1],b.tensors[-1]],axis=0))
    c.tensors=tensors;c.center=n-1;c.move_center(0)
    return c.norm()


def tensor_fit(source,target,start,width,max_bond=4):
    """Exact least-squares center tensor, fixed exterior MPS tensors."""
    out=copy.deepcopy(source);out.move_center(start)
    left=np.ones((1,1),complex)
    for j in range(start):
        left=np.einsum('ab,asr,bsu->ru',left,out.tensors[j].conj(),target.tensors[j],optimize=True)
    right=np.ones((1,1),complex)
    for j in range(source.n-1,start+width-1,-1):
        right=np.einsum('ru,asr,bsu->ab',right,out.tensors[j].conj(),target.tensors[j],optimize=True)
    block=target.tensors[start]
    for j in range(start+1,start+width):block=np.tensordot(block,target.tensors[j],axes=(-1,0))
    target_shape=block.shape
    block=np.einsum('al,lpr,br->apb',left,block.reshape(target_shape[0],4**width,target_shape[-1]),right,optimize=True)
    discarded=0.;dl=block.shape[0]
    for offset in range(width-1):
        u,s,vh=np.linalg.svd(block.reshape(dl*4,-1),full_matrices=False)
        rank=min(max_bond,len(s));discarded+=float(np.sum(s[rank:]**2))
        out.tensors[start+offset]=u[:,:rank].reshape(dl,4,rank)
        block=s[:rank,None]*vh[:rank];dl=rank
    out.tensors[start+width-1]=block.reshape(dl,4,-1)
    out.center=start+width-1
    return out,discarded


def binary_entropy(p):
    p=float(np.clip(p,0.,1.))
    if p==0 or p==1:return 0.
    return -p*np.log(p)-(1-p)*np.log1p(-p)


def cmi(a,b,c):
    """a,b,c are unnormalized occupied masses on a complete tripartition."""
    return binary_entropy(a+b)+binary_entropy(b+c)-binary_entropy(b)-binary_entropy(a+b+c)


def unrestricted(outdir):
    rows=[]
    for n in (6,64,128):
        j=n//2
        for profile in ('uniform','packet'):
            v=(np.ones(n)/np.sqrt(n) if profile=='uniform' else study.wavepacket(n,j,2.,.4))[:,None]
            w=v.copy();w[j]*=np.sqrt(.7)
            source,target=study.cp_mps(v),study.cp_mps(w)
            for width in (1,2,3):
                start=j if width<=2 else j-1
                fitted,discard=tensor_fit(source,target,start,width)
                error=difference_norm(fitted,target)
                floor=study.unitary_floor(v[:,0],w[:,0],start,start+width)
                assert error<2e-11 and discard<1e-20
                if n==6:assert abs(error-np.linalg.norm(fitted.dense()-target.dense()))<1e-12
                inside=float(np.linalg.norm(v[start:start+width])**2)
                a=float(abs(v[j,0])**2);b=inside-a;c=1-inside
                before=cmi(a,b,c);after=cmi(.7*a,b,c)
                markov_bound=np.sqrt(max(0,before-after))
                assert floor<=markov_bound+2e-7
                rows.append(dict(n=n,profile=profile,active_width=width,cp_bond=fitted.bond_dimension(),
                                 unitary_floor=floor,tensor_error=error,discarded_weight=discard,
                                 cmi_before=before,cmi_after=after,channel_unitary_bound=markov_bound))
    study.save(outdir/'tensor_vs_unitary.csv',rows)


def markov(outdir):
    rows=[]
    for survival in (1.,float(np.exp(-.3))):
        for n in (32,64,128,256,512,1024,4096):
            # A is the leftmost site, B is the following N/2 sites, C is the rest.
            b=n//2
            value=cmi(survival/n,survival*b/n,survival*(n-b-1)/n)
            rows.append(dict(n=n,survival=survival,probe_size=1,buffer_size=b,cmi=value,n_times_cmi=n*value))
    study.save(outdir/'markov_scaling.csv',rows)


def tangent_project(psi,force,n):
    """Standard full MPS tangent projector from exact Schmidt-support frames."""
    left=[np.ones((1,1),complex)];right=[None]*(n+1);right[n]=np.ones((1,1),complex)
    ranks=[]
    for cut in range(1,n):
        u,s,vh=np.linalg.svd(psi.reshape(4**cut,-1),full_matrices=False)
        rank=int(np.sum(s>1e-12*s[0]));ranks.append(rank)
        left.append(u[:,:rank]);right[cut]=vh[:rank]
    projected=np.zeros_like(force)
    for j in range(n):
        l,r=left[j],right[j+1];x=force.reshape(4**j,4,4**(n-j-1))
        core=np.einsum('al,apb,rb->lpr',l.conj(),x,r.conj(),optimize=True)
        projected+=np.einsum('al,lpr,rb->apb',l,core,r,optimize=True).reshape(-1)
    for cut in range(1,n):
        l,r=left[cut],right[cut];x=force.reshape(4**cut,-1)
        projected-=(l@(l.conj().T@x@r.conj().T)@r).reshape(-1)
    return projected,ranks


def tangent(outdir):
    rows=[]
    for n in (4,5,6):
        for kind in ('uniform_loss','hopping_nonuniform_loss'):
            u=study.wavepacket(n,n//2-.5,1.7,.6)[:,None]
            if kind=='uniform_loss':ons=np.zeros(n);hops=np.zeros(n-1,complex);rates=np.ones(n)
            else:ons,hops,rates=study.model(n)
            for t in (.05,.3,1.):
                v=study.reference_integrate(u,ons,hops,rates,t)[:,0]
                dv=study.generator_action(v[:,None],ons,hops,rates)[:,0]
                s=float(np.vdot(v,v).real);ds=2*float(np.vdot(v,dv).real)
                occupied=(np.outer(dv,v.conj())+np.outer(v,dv.conj()))/np.sqrt(s)-ds*np.outer(v,v.conj())/(2*s**1.5)
                root_dot=np.zeros((2**n,2**n),complex);idx=[1<<(n-1-j) for j in range(n)]
                root_dot[np.ix_(idx,idx)]=occupied;root_dot[0,0]=-ds/(2*np.sqrt(1-s))
                psi=study.cp_mps(v[:,None]).dense();force=base.to_interleaved(root_dot,n)
                projected,ranks=tangent_project(psi,force,n)
                residual=np.linalg.norm(projected-force);relative=residual/np.linalg.norm(force)
                assert relative<2e-11
                assert abs(np.vdot(psi,force))<2e-12
                # Check the Sylvester derivative identity in full spin space.
                root=base.to_matrix(psi,n);rho_dot=np.zeros_like(root)
                rho_dot[np.ix_(idx,idx)]=np.outer(dv,v.conj())+np.outer(v,dv.conj());rho_dot[0,0]=-ds
                sylvester=np.linalg.norm(root@root_dot+root_dot@root-rho_dot)
                assert sylvester<2e-12
                rows.append(dict(n=n,model=kind,time=t,max_schmidt_rank=max(ranks),
                                 force_norm=np.linalg.norm(force),tangent_residual=residual,
                                 relative_residual=relative,sylvester_error=sylvester))
    study.save(outdir/'tangent_projection.csv',rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--outdir',type=Path,default=Path(__file__).parent/'tdvp_markov_results')
    args=parser.parse_args();args.outdir.mkdir(parents=True,exist_ok=True)
    unrestricted(args.outdir);markov(args.outdir);tangent(args.outdir)
    print('Unrestricted tensor, Markov-scaling, and instantaneous TDVP projection checks passed.')


if __name__=='__main__':main()
