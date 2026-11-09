"""Two-site tensor variational updates after one-site dephasing.

Small-system research benchmark. No variational gates or analytic CP generators.
Requires numpy, scipy, matplotlib, and cluster_dephasing.py beside this file.
The exact CP objective and PSD-distance penalty are dense diagnostic references.
The density, Hermiticity, and trace-anchor objectives require no target square root.
"""
from __future__ import annotations
import argparse
import copy
import csv
import json
import os
from pathlib import Path
import time
# Small matrices run faster and more reproducibly with one BLAS worker.
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize
import cluster_dephasing as base


def positive_root(rho):
    values, vectors = eigh((rho+rho.conj().T)/2)
    # Numerical rank threshold only; all results are small-system diagnostics.
    values = np.where(values > 5e-14, values, 0.)
    return (vectors*np.sqrt(values))@vectors.conj().T


def noise(rho, n, j, dt):
    signs = 1-2*((np.arange(2**n) >> (n-1-j)) & 1)
    p = -np.expm1(-dt)/2
    return (1-p)*rho+p*signs[:,None]*rho*signs[None,:]


def from_vector(vector, n, chi=4):
    out = base.MPS(n, max_bond=chi)
    block = vector.reshape(1,-1)
    left = 1
    for j in range(n-1):
        u,s,vh = np.linalg.svd(block.reshape(left*4,-1),full_matrices=False)
        rank = min(chi, max(1,int(np.sum(s>1e-13*s[0]))))
        out.tensors[j] = u[:,:rank].reshape(left,4,rank)
        block = s[:rank,None]*vh[:rank]
        left = rank
    out.tensors[-1] = block.reshape(left,4,1)
    out.center = n-1
    return out


class LocalProblem:
    """Linear embedding of the merged tensor into frozen orthonormal environments."""
    def __init__(self, state, start, width, target):
        self.state = copy.deepcopy(state)
        self.state.move_center(start)
        self.n, self.start, self.width = state.n, start, width
        self.target = target
        self.purity = float(np.vdot(target,target).real)
        if start:
            left = self.state.tensors[0]
            for a in self.state.tensors[1:start]:
                left = np.tensordot(left,a,axes=(-1,0))
            self.left = left.reshape(4**start,-1)
        else:
            self.left = np.ones((1,1),complex)
        stop = start+width
        if stop < state.n:
            right = self.state.tensors[stop]
            for a in self.state.tensors[stop+1:]:
                right = np.tensordot(right,a,axes=(-1,0))
            self.right = right.reshape(right.shape[0],-1)
        else:
            self.right = np.ones((1,1),complex)
        block = self.state.tensors[start]
        for a in self.state.tensors[start+1:stop]:
            block = np.tensordot(block,a,axes=(-1,0))
        self.shape = (self.left.shape[1],4**width,self.right.shape[0])
        self.initial = block.reshape(self.shape)
        assert np.linalg.norm(self.left.conj().T@self.left-np.eye(self.shape[0])) < 1e-10
        assert np.linalg.norm(self.right@self.right.conj().T-np.eye(self.shape[2])) < 1e-10

    def embed(self, block):
        v = np.einsum('la,asb,br->lsr',self.left,block,self.right,optimize=True).reshape(-1)
        return base.to_matrix(v,self.n)

    def adjoint(self, matrix):
        v = base.to_interleaved(matrix,self.n).reshape(self.left.shape[0],4**self.width,self.right.shape[1])
        return np.einsum('la,lsr,br->asb',self.left.conj(),v,self.right.conj(),optimize=True)

    @staticmethod
    def pack(block):
        z = block.reshape(-1)
        return np.concatenate([z.real,z.imag])

    def unpack(self, params):
        half = len(params)//2
        return (params[:half]+1j*params[half:]).reshape(self.shape)

    def loss_grad(self, params, method, weight=1.):
        raw = self.unpack(params)
        norm = np.linalg.norm(raw)
        block = raw/norm
        x = self.embed(block)
        residual = x@x.conj().T-self.target
        loss = float(np.vdot(residual,residual).real/self.purity)
        grad = 4*residual@x/self.purity
        if method in ('hermitian','psd','trace_anchor'):
            if method == 'psd':
                h = (x+x.conj().T)/2
                values,vectors = eigh(h)
                positive = (vectors*np.maximum(values,0))@vectors.conj().T
                defect = x-positive
                loss += weight*float(np.vdot(defect,defect).real)
                grad += 2*weight*defect
            else:
                defect = (x-x.conj().T)/2
                loss += weight*float(np.vdot(defect,defect).real)
                grad += 2*weight*defect
        if method == 'trace_anchor':
            strength = 1e-5
            loss -= strength*np.trace(x).real/np.sqrt(x.shape[0])
            grad -= strength*np.eye(x.shape[0])/np.sqrt(x.shape[0])
        gb = self.adjoint(grad)
        gb = (gb-block*np.vdot(block,gb).real)/norm
        return loss,self.pack(gb)

    def commit(self, block):
        state = copy.deepcopy(self.state)
        block = block/np.linalg.norm(block)
        left,_,right = self.shape
        discarded = 0.
        for offset in range(self.width-1):
            u,s,vh = np.linalg.svd(block.reshape(left*4,-1),full_matrices=False)
            rank = min(state.max_bond,max(1,int(np.sum(s>1e-13*s[0]))))
            discarded += float(np.sum(s[rank:]**2))
            state.tensors[self.start+offset] = u[:,:rank].reshape(left,4,rank)
            block = s[:rank,None]*vh[:rank]
            left = rank
        state.tensors[self.start+self.width-1] = block.reshape(left,4,right)
        state.center = self.start+self.width-1
        norm = state.norm()  # This canonicalization moves the center to site zero.
        state.tensors[state.center] /= norm
        return state,discarded

    def solve(self, method, seed=0, restarts=1, weight=1., maxiter=500):
        if method == 'oracle':
            # Exact least-squares solution in the merged tensor, then optimal SVD split.
            projected = self.adjoint(positive_root(self.target))
            state,discarded = self.commit(projected)
            return state,dict(iterations=0,evaluations=0,success=True,loss=0.,discarded=discarded,initial_gradient=0.)
        initial = self.pack(self.initial)
        initial_gradient = float(np.linalg.norm(self.loss_grad(initial,method,weight)[1]))
        # Optimize the two tensors themselves at fixed internal bond dimension.
        # This prevents an unconstrained merged block from using a larger bond.
        if self.width == 2:
            dl,_,dr=self.shape
            u,s,vh=np.linalg.svd(self.initial.reshape(dl*4,4*dr),full_matrices=False)
            rank=min(self.state.max_bond,len(s))
            a=(u[:,:rank]*np.sqrt(s[:rank])).reshape(dl,4,rank)
            b=(np.sqrt(s[:rank,None])*vh[:rank]).reshape(rank,4,dr)
            na=a.size
            initial=self.pack(np.concatenate([a.reshape(-1),b.reshape(-1)]))
            def merged(params):
                half=len(params)//2
                z=params[:half]+1j*params[half:]
                aa=z[:na].reshape(dl,4,rank)
                bb=z[na:].reshape(rank,4,dr)
                block=np.einsum('aik,kjb->aijb',aa,bb).reshape(self.shape)
                return aa,bb,block
            def objective(params):
                aa,bb,block=merged(params)
                loss,packed_grad=self.loss_grad(self.pack(block),method,weight)
                gb=self.unpack(packed_grad).reshape(dl,4,4,dr)
                ga=np.einsum('aijb,kjb->aik',gb,bb.conj())
                gbb=np.einsum('aik,aijb->kjb',aa.conj(),gb)
                return loss,self.pack(np.concatenate([ga.reshape(-1),gbb.reshape(-1)]))
            decode=lambda params:merged(params)[2]
            # A reproducible directional check of the factor-chain rule.
            check_rng=np.random.default_rng(123)
            probe=initial+.003*check_rng.normal(size=initial.shape)
            direction=check_rng.normal(size=initial.shape);direction/=np.linalg.norm(direction)
            value,gradient=objective(probe)
            eps=1e-6
            finite_difference=(objective(probe+eps*direction)[0]-objective(probe-eps*direction)[0])/(2*eps)
            assert abs(finite_difference-np.dot(gradient,direction)) < 2e-6
        else:
            objective=lambda params:self.loss_grad(params,method,weight)
            decode=self.unpack
        rng = np.random.default_rng(seed)
        starts = [initial]
        for _ in range(restarts):
            kick = rng.normal(size=initial.shape)
            starts.append(initial+.01*kick/np.linalg.norm(kick))
        results=[]
        for x0 in starts:
            result=minimize(objective,x0,jac=True,method='L-BFGS-B',
                            options=dict(maxiter=maxiter,ftol=1e-16,gtol=1e-10,maxls=40,maxcor=20))
            results.append(result)
        best=min(results,key=lambda r:r.fun)
        optimized=decode(best.x)
        pre_x=self.embed(optimized/np.linalg.norm(optimized))
        pre_residual=pre_x@pre_x.conj().T-self.target
        pre_trace=float(np.sum(np.abs(np.linalg.eigvalsh(pre_residual)))/2)
        state,discarded=self.commit(optimized)
        info=dict(iterations=int(best.nit),evaluations=sum(int(r.nfev) for r in results),
                  success=bool(best.success),loss=float(best.fun),discarded=discarded,
                  initial_gradient=initial_gradient,optimizer_message=str(best.message))
        info['pre_split_trace_distance']=pre_trace
        return state,info


def metrics(state,target,exact_global=None):
    x=base.to_matrix(state.dense(),state.n)
    rho=x@x.conj().T
    root=positive_root(target)
    overlap=np.vdot(root,x)
    cp_error=float(np.sqrt(max(0.,2-2*abs(overlap))))
    # Direct norm avoids cancellation near machine precision.
    aligned=x*np.exp(-1j*np.angle(overlap)) if abs(overlap)>0 else x
    cp_error=float(np.linalg.norm(aligned-root))
    canonical=positive_root(rho)
    h=(aligned+aligned.conj().T)/2
    values=np.linalg.eigvalsh(h)
    difference=rho-target
    out=dict(cp_error=cp_error,trace_distance=float(np.sum(np.abs(np.linalg.eigvalsh(difference)))/2),
             density_relative_hs=float(np.vdot(difference,difference).real/np.vdot(target,target).real),
             canonical_drift=float(np.linalg.norm(aligned-canonical)),
             hermiticity=float(np.linalg.norm(aligned-aligned.conj().T)/2),
             negative_part=float(np.linalg.norm(values[values<0])),bond_dimension=state.bond_dimension())
    if exact_global is not None:
        target_root=positive_root(exact_global)
        ov=np.vdot(target_root,x)
        out['global_cp_error']=float(np.linalg.norm(x*np.exp(-1j*np.angle(ov))-target_root))
        out['global_trace_distance']=float(np.sum(np.abs(np.linalg.eigvalsh(rho-exact_global)))/2)
    return out


def gradient_check(problem):
    rng=np.random.default_rng(20261102)
    p=problem.pack(problem.initial)+.03*rng.normal(size=2*problem.initial.size)
    direction=rng.normal(size=p.shape); direction/=np.linalg.norm(direction)
    errors={}
    for method in ['density','hermitian','psd','trace_anchor']:
        _,grad=problem.loss_grad(p,method)
        eps=1e-6
        fd=(problem.loss_grad(p+eps*direction,method)[0]-problem.loss_grad(p-eps*direction,method)[0])/(2*eps)
        error=abs(fd-np.dot(grad,direction))
        errors[method]=float(error)
        assert error < 1e-6, errors
    return errors


def write_csv(path,rows):
    keys=list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w',newline='',encoding='utf-8') as file:
        writer=csv.DictWriter(file,fieldnames=keys);writer.writeheader();writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--sweeps',type=int,default=3)
    parser.add_argument('--initial-time',type=float,default=0.,help='Initial uniform dephasing time for sequential runs.')
    parser.add_argument('--methods',nargs='+',choices=['oracle','density','hermitian','psd','trace_anchor'],
                        default=['oracle','density','hermitian','psd','trace_anchor'])
    parser.add_argument('--restarts',type=int,default=1,help='Perturbed starts in addition to the warm start in sequential runs.')
    parser.add_argument('--trajectory-only',action='store_true',help='Skip isolated updates and support diagnostics.')
    args=parser.parse_args();args.outdir.mkdir(parents=True,exist_ok=True)
    n,j,dt=6,2,.1
    support=[];isolated=[];trajectories=[]
    start_time=time.perf_counter()
    methods=args.methods
    gradients={}
    for initial_time in ([] if args.trajectory_only else [0.,.3]):
        initial=from_vector(base.to_interleaved(base.analytic_root(n,initial_time),n),n)
        x=base.to_matrix(initial.dense(),n)
        sigma=noise(x@x.conj().T,n,j,dt)
        blocks=[(j,1),(j-1,2),(j,2),(j-1,3)]
        for start,width in blocks:
            problem=LocalProblem(initial,start,width,sigma)
            updated,info=problem.solve('oracle')
            row=dict(initial_time=initial_time,start=start,width=width,**metrics(updated,sigma),**info)
            support.append(row)
        problem=LocalProblem(initial,j,2,sigma)
        if initial_time == 0:
            gradients=gradient_check(problem)
        for method in methods:
            # Warm-start-only tests diagnose stationary points before perturbations.
            for restarts in ([0,2] if method!='oracle' else [0]):
                updated,info=problem.solve(method,seed=42,restarts=restarts)
                row=dict(initial_time=initial_time,method=method,restarts=restarts,**metrics(updated,sigma),**info)
                isolated.append(row)
                print(json.dumps({'isolated':row}),flush=True)
    write_csv(args.outdir/'tensor_search_support.csv',support)
    write_csv(args.outdir/'tensor_search_isolated.csv',isolated)
    for method in methods:
        state=(base.prepare_cluster_cp(n) if args.initial_time == 0 else
               from_vector(base.to_interleaved(base.analytic_root(n,args.initial_time),n),n))
        exact=base.channel_reference(n,args.initial_time)
        for sweep in range(args.sweeps):
            for site in range(n):
                x=base.to_matrix(state.dense(),n)
                sigma=noise(x@x.conj().T,n,site,dt)
                sigma/=np.trace(sigma).real
                exact=noise(exact,n,site,dt)
                start=min(site,n-2)
                problem=LocalProblem(state,start,2,sigma)
                state,info=problem.solve(method,seed=1000+sweep*n+site,restarts=args.restarts,maxiter=600)
                row=dict(method=method,sweep=sweep+1,site=site,update=sweep*n+site+1,
                         **metrics(state,sigma,exact),**info)
                trajectories.append(row)
            print(json.dumps({'sweep':method,'number':sweep+1,'last':row}),flush=True)
    write_csv(args.outdir/'tensor_search_trajectories.csv',trajectories)
    summary=dict(n=n,dt=dt,sweeps=args.sweeps,initial_time=args.initial_time,restarts=args.restarts,
                 gradient_check_errors=gradients,
                 support=support,isolated=isolated,
                 final=[r for r in trajectories if r['update']==n*args.sweeps],
                 seconds=time.perf_counter()-start_time)
    (args.outdir/'tensor_search_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    # The exact-target fit must demonstrate representability independently of optimizer.
    if support:
        assert max(r['cp_error'] for r in support) < 1e-6, support
    if trajectories and 'oracle' in methods:
        assert max(r['global_cp_error'] for r in trajectories if r['method']=='oracle') < 1e-6
    if trajectories:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
        fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
        names=dict(oracle='Exact CP target (reference)',density='Density HS',hermitian='Density + Hermiticity',
                   psd='Density + PSD distance',trace_anchor='Density + Hermiticity + trace')
        colors=dict(oracle='#64748b',density='#2563eb',hermitian='#d97706',psd='#059669',trace_anchor='#9333ea')
        for method in methods:
            color=colors[method]
            rows=[r for r in trajectories if r['method']==method]
            for axis,key in zip(axes,['global_trace_distance','global_cp_error','canonical_drift']):
                axis.semilogy([r['update'] for r in rows],np.maximum([r[key] for r in rows],1e-15),
                              label=names[method],color=color)
        for axis,title in zip(axes,['Physical density error','Canonical-purification error','Purification gauge drift']):
            axis.set(title=title,xlabel='One-site noise updates',ylabel='Error')
            axis.grid(alpha=.15)
        axes[0].legend(fontsize=8)
        fig.suptitle(f'Two-site tensor optimization; N=6, bond cap=4, initial time={args.initial_time:g}')
        fig.savefig(args.outdir/'tensor_search_cost_comparison.png',dpi=180)
        plt.close(fig)
    print(json.dumps({'seconds':summary['seconds'],'final':summary['final']}),flush=True)


if __name__=='__main__':
    main()
