import rank_opening_mps as opening
import mpo_cp_force as mpo
import cluster_dephasing as cluster
import large_mpo_tdvp as tables
import tdvp_core as tn
import automatic_rank_opening as dense
import test_automatic_rank_opening as models
import numpy as np
from pathlib import Path
import argparse
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--n',nargs='+',type=int,default=[4,16]);p.add_argument('--outdir',type=Path,default=Path(__file__).parent/'automatic_rank_results');args=p.parse_args();rows=[]
    for n in args.n:
        rates=.3+.12*np.cos(np.arange(n))**2
        for case in ('product','cluster'):
            root=mpo.product_root(n,1.) if case=='product' else cluster.prepare_cluster_cp(n)
            state,initial,factors=opening.automatic_startup(root,rates,1e-3,check_handoff=False)
            start=time.perf_counter();info=opening.detect_factor_support(factors,rates)
            row=dict(n=n,case=case,step=initial['step'],seconds=time.perf_counter()-start,**info)
            if n<=4:
                f=np.column_stack([tn.to_matrix(v.dense(),n)[:,0] for v in factors]);u,s,_=np.linalg.svd(f,full_matrices=False)
                exactroot=(u*(s/np.linalg.norm(s)))@u.conj().T
                exact=dense.detect(exactroot,models.model(n),initial['step'])
                row['dense_leakage_rate']=exact['leakage_rate']
                assert abs(info['leakage_rate']-exact['leakage_rate'])<1e-8
            rows.append(row);print(row,flush=True);tables.save(args.outdir/'handoff_audit.csv',rows)


if __name__=='__main__':main()
