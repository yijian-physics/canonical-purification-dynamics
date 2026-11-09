import rank_opening_mps as opening
import mpo_cp_force as mpo
import tdvp_core as tn
import cluster_dephasing as cluster
import large_mpo_tdvp as tables
import test_automatic_rank_opening as dense
import numpy as np
from pathlib import Path
import argparse
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--n',type=int,nargs='+',default=[4,16,32]);p.add_argument('--cases',nargs='+',default=['product','cluster'])
    p.add_argument('--budget',type=float,default=1e-3);p.add_argument('--outdir',type=Path,default=Path(__file__).parent/'automatic_rank_results')
    args=p.parse_args();args.outdir.mkdir(exist_ok=True,parents=True);rows=[]
    for n in args.n:
        rates=.3+.12*np.cos(np.arange(n))**2
        for case in args.cases:
            initial=mpo.product_root(n,1.) if case=='product' else cluster.prepare_cluster_cp(n)
            start=time.perf_counter();state,info,factors=opening.automatic_startup(initial,rates,args.budget,check_handoff=False)
            row=dict(n=n,case=case,seconds=time.perf_counter()-start,**{k:v for k,v in info.items() if k!='site_leakages'})
            if n<=6:
                actual=tn.to_matrix(state.dense(),n);q0=tn.to_matrix(initial.dense(),n)
                exact=dense.exact_root(q0,dense.model(n),info['step'])
                f=np.column_stack([tn.to_matrix(v.dense(),n)[:,0] for v in factors]);u,s,_=np.linalg.svd(f,full_matrices=False)
                target=(u*(s/np.linalg.norm(s)))@u.conj().T
                row.update(cp_error=np.linalg.norm(actual-exact),kraus_root_error=np.linalg.norm(actual-target),min_root_eigenvalue=min(np.linalg.eigvalsh((actual+actual.conj().T)/2)))
                assert row['cp_error']<row['total_bound_excluding_roundoff']+1e-7
                assert row['kraus_root_error']<1e-8
            rows.append(row);print(row,flush=True)
            tag=f'n{n}_{case}_budget{args.budget:g}'
            tables.save(args.outdir/f'{tag}.csv',[row]);np.savez_compressed(args.outdir/f'{tag}_state.npz',**{f'a{j}':a for j,a in enumerate(state.tensors)})


if __name__=='__main__':main()
