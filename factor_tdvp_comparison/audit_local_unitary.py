"""Diagnose the repeated cluster adapter's first boundary-unitary failure.

The fit and application routines are reused unchanged. Direct density partial
trace and full dense unitary application independently check adapter wiring.
"""
import copy
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from factor_tdvp_comparison import lift_benchmarks as lift


def run_audit():
    n=4; rates=.7+.15*np.cos(np.arange(n))**2
    state=lift.original.reference(n,'cluster',0.,rates)
    amplitude=lift.tn.to_matrix(state.dense(),n)
    density=amplitude@amplitude.conj().T
    p=-np.expm1(-rates[0]*.05)/2
    channel=[np.sqrt(1-p)*lift.base.I,np.sqrt(p)*lift.base.Z]
    exposure=np.zeros(n);exposure[0]=rates[0]*.05
    target=lift.original.reference(n,'cluster',1.,exposure)
    oracle,_=lift.tilt.best_unitary(state,target,0,2)
    oracle_error=lift.tn.distance(oracle,target)
    rows=[]
    for fit_width in (2,3,4):
        factor=lift.hp.physical_factor(copy.deepcopy(state),0,fit_width)
        physical=factor@factor.conj().T
        direct=np.trace(density.reshape(2**fit_width,2**(n-fit_width),2**fit_width,2**(n-fit_width)),axis1=1,axis2=3)
        gate=lift.local_algo.fit_gate(physical,[0],0,2,channel)
        actual=copy.deepcopy(state);actual.max_bond=64
        discard=lift.local_algo.apply_compact(actual,0,2,gate)
        dense_unitary=np.eye(16)+gate.basis@(gate.rotation-np.eye(len(gate.rotation)))@gate.basis.conj().T
        direct_vector=(dense_unitary@state.dense().reshape(16,-1)).reshape(-1)
        source=lift.local_algo.previous.small_positive_root(physical)
        local_rows=lift.base.to_interleaved(source,fit_width).reshape(16,-1)
        u,s,_=np.linalg.svd(local_rows,full_matrices=False)
        local_support=u[:,s>1e-12*s[0]]
        global_rows=state.dense().reshape(16,-1)
        rows.append(dict(n=n,active_width=2,fit_width=fit_width,chi=64,dt=.05,site=0,
                         physical_cache_error=float(np.linalg.norm(physical-direct)),
                         local_fit_residual=float(gate.residual),
                         global_cp_error=float(lift.tn.distance(actual,target)),
                         dense_application_error=float(np.linalg.norm(actual.dense()-direct_vector)),
                         local_source_span_rank=local_support.shape[1],
                         global_source_outside_local_span=float(np.linalg.norm(global_rows-local_support@local_support.conj().T@global_rows)),
                         discarded_weight=float(sum(discard)),
                         exact_target_same_support_error=float(oracle_error)))
    return rows


if __name__=='__main__':
    rows=run_audit()
    path=ROOT/'factor_tdvp_comparison/results/local_unitary_audit.json'
    path.write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps(rows,indent=2))
