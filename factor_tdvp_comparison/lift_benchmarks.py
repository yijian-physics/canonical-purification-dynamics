"""Matched lift controls, reusing the original half-Petz and Procrustes code.

Repeated local-unitary tests are an explicit sequential adapter of the original
local-data unitary fit. Physical factor caches evolve under the channels, never
from a future canonical root. All scores are raw global Frobenius distances.
"""
from __future__ import annotations
import argparse
import copy
import csv
import importlib.util
import json
import resource
import sys
from pathlib import Path
from time import perf_counter
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT/'half_petz_evolution', ROOT/'tilted_half_petz_comparison'):
    if str(folder) not in sys.path: sys.path.insert(0,str(folder))
import half_petz_evolution as hp
import tdvp_core as tn
import cluster_dephasing as base


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


original=load('_comparison_hp_tests',ROOT/'half_petz_evolution/test_half_petz_evolution.py')
tilt=load('_comparison_tilt',ROOT/'tilted_half_petz_comparison/compare_tilted_half_petz.py')
sys.path.insert(0,str(ROOT/'local_cp_large'))
local_algo=load('_comparison_local',ROOT/'local_cp_large/local_cp_large.py')


def scalar(value):
    if isinstance(value,np.generic): return value.item()
    raise TypeError(type(value).__name__)


def write(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.with_suffix('.json').write_text(json.dumps(rows,indent=2,default=scalar)+'\n')
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.with_suffix('.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys,lineterminator='\n')
        writer.writeheader();writer.writerows(rows)


def metrics(amplitude,target):
    """Raw CP amplitude score; positivity is not assumed for the lift output."""
    result=dict(global_cp_error=float(tn.distance(amplitude,target)),
                normalization_defect=float(abs(tn.norm(amplitude)**2-1)),
                x_bond=amplitude.bond_dimension(),h_bond=None,
                average_x_bond=float(np.mean([a.shape[-1] for a in amplitude.tensors[:-1]])),
                density_error=None,canonical_branch_defect=None,minimum_hermitian_eigenvalue=None)
    if amplitude.n<=8:
        a=tn.to_matrix(amplitude.dense(),amplitude.n)
        s=tn.to_matrix(target.dense(),target.n)
        u,v,_=np.linalg.svd(a,full_matrices=False)
        own=(u*v)@u.conj().T
        result.update(density_error=float(np.linalg.norm(a@a.conj().T-s@s)),
                      canonical_branch_defect=float(np.linalg.norm(a-own)),
                      minimum_hermitian_eigenvalue=float(np.linalg.eigvalsh((a+a.conj().T)/2)[0]))
    return result


def repeated(n,family,method='half_petz',steps=20,total=1.,chi=4,window=3):
    start=perf_counter();rates=.7+.15*np.cos(np.arange(n))**2
    initial=original.reference(n,family,0.,rates)
    preparation=perf_counter()-start
    t=perf_counter()
    cache=hp.HalfPetzEvolution(initial,buffer=window//2 if method=='half_petz' else window//2+1,chi=chi)
    state=copy.deepcopy(initial)
    setup=perf_counter()-t+preparation
    evolution=scoring=0.;truncation=0.;inverse=0.;history=[];bonds=[]
    for step in range(steps):
        t=perf_counter()
        for site,rate in enumerate(rates):
            p=-np.expm1(-rate*total/steps)/2
            channel=[np.sqrt(1-p)*base.I,np.sqrt(p)*base.Z]
            if method=='half_petz':
                info=cache.update(site,channel);state=cache.state
                truncation+=info.truncation_bound;inverse=max(inverse,info.inverse_norm)
            else:
                active_start=max(0,site-window//2);active_stop=min(n,site+window//2+1)
                rs,rw=cache.site_window[site]
                if method=='local_unitary_regularized':
                    physical=cache.factors[(rs,rw)]@cache.factors[(rs,rw)].conj().T
                    gate=local_algo.fit_gate(physical,[site-rs],active_start-rs,active_stop-active_start,channel)
                    state.max_bond=chi
                    discarded=local_algo.apply_compact(state,active_start,active_stop-active_start,gate)
                    truncation+=float(sum(np.sqrt(discarded)))
                else:
                    state,info=tilt.local_data_unitary(state,site,channel,active_start,active_stop-active_start,
                                                      cache.factors[(rs,rw)],rs,rw)
                    state,rounding=tn.compress(state,chi,normalize=False)
                    truncation+=info['rounding']+rounding['discarded_bound']
                for key in cache.affected[site]:
                    left,width=key
                    local=[hp.embed(k,site-left,width) for k in channel]
                    factor=np.column_stack([k@cache.factors[key] for k in local])
                    u,s,discard=hp.factor_spectrum(factor,cache.cutoff)
                    cache.factors[key]=u*s
                    cache.cache_discarded_mass+=discard
            bonds.append(state.bond_dimension())
        evolution+=perf_counter()-t
        t=perf_counter()
        target=original.reference(n,family,(step+1)*total/steps,rates)
        score=metrics(state,target)
        scoring+=perf_counter()-t
        history.append(dict(benchmark='A' if family=='cluster' else 'B',method=method,n=n,
                            time=(step+1)*total/steps,dt=total/steps,steps=step+1,
                            evolution_seconds=evolution,scoring_seconds=scoring,**score))
    row=dict(benchmark='A' if family=='cluster' else 'B',family=family,method=method,n=n,status='ok',
             error_kind='measured_global',time=total,dt=total/steps,steps=steps,accepted_steps=steps,
             rejected_steps=0,elementary_updates=n*steps,window=window,fit_window=min(n,window+2) if method!='half_petz' else min(n,window),
             chi_x_limit=chi,setup_seconds=setup,preparation_seconds=preparation,
             evolution_seconds=evolution,scoring_seconds=scoring,total_seconds=setup+evolution+scoring,
             truncation_error=truncation,largest_local_inverse=inverse if method=='half_petz' else None,
             spectral_discarded_mass=cache.cache_discarded_mass,maximum_x_bond=max(bonds),
             average_x_bond_over_updates=float(np.mean(bonds)),
             peak_process_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
             timing_note='Scoring includes each endpoint; peak RSS is process high-water, not incremental.',
             adapter=('original_half_petz' if method=='half_petz' else 'original_local_cp_large_fit_gate_apply_compact_with_independent_channel_cache' if method=='local_unitary_regularized' else 'original_tilted_local_data_unitary_with_channel_factor_cache_and_explicit_compression'),
             **score)
    row['accuracy_status']='meets_1e-3' if row['global_cp_error']<=1e-3 else 'misses_1e-3'
    if n<=6:
        t=perf_counter();dense,physical=original.dense_reference(initial,family,total,rates)
        row['independent_dense_reference_error']=float(np.linalg.norm(tn.to_matrix(target.dense(),n)-dense))
        row['extra_validation_seconds']=perf_counter()-t
    return row,history


def tilted(n,theta,window=3,probability=.1):
    t=perf_counter();initial=tilt.prepare(n);preparation=perf_counter()-t
    site=n//2;pauli=np.cos(theta)*base.Z+np.sin(theta)*base.X
    channel=[np.sqrt(1-probability)*base.I,np.sqrt(probability)*pauli]
    t=perf_counter();target,_=tilt.exact_target(initial,site,probability,pauli)
    reference_time=perf_counter()-t
    rows=[]
    for method in ('half_petz','local_unitary','unitary_oracle'):
        t=perf_counter()
        if method=='half_petz':
            solver=hp.HalfPetzEvolution(initial,buffer=(window-1)//2,chi=64)
            left,width=solver.site_window[site]
            setup=perf_counter()-t+preparation
            t=perf_counter();info=solver.update(site,channel);output=solver.state
            evolution=perf_counter()-t;trunc=info.truncation_bound;local_residual=None
        else:
            left=max(0,site-(window-1)//2);right=min(n,site+(window-1)//2+1);width=right-left
            rs=max(0,left-1);rw=min(n,right+1)-rs
            if method=='local_unitary':
                factor=hp.physical_factor(copy.deepcopy(initial),rs,rw)
                u,s,_=hp.factor_spectrum(factor,1e-13);factor=u*s
            setup=perf_counter()-t+preparation
            t=perf_counter()
            if method=='local_unitary':
                output,info=tilt.local_data_unitary(initial,site,channel,left,width,factor,rs,rw)
                local_residual=float(info['residual'])
            else:
                output,info=tilt.best_unitary(initial,target,left,width)
                local_residual=None
            evolution=perf_counter()-t;trunc=info['rounding']
        t=perf_counter();score=metrics(output,target);scoring=perf_counter()-t+reference_time
        rows.append(dict(benchmark='C',method=method,n=n,status='ok',error_kind='oracle_lower_bound' if method=='unitary_oracle' else 'measured_global',
                         theta=theta,p=probability,time=1.,gamma=float(-np.log1p(-2*probability)),dt=1.,steps=1,
                         accepted_steps=1,rejected_steps=0,window=width,fit_window=rw if method=='local_unitary' else width,
                         chi_x_limit=64,setup_seconds=setup,preparation_seconds=preparation,
                         evolution_seconds=evolution,scoring_seconds=scoring,total_seconds=setup+evolution+scoring,
                         truncation_error=trunc,local_fit_residual=local_residual,
                         peak_process_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                         timing_note='Each method charged independent preparation; reference is scoring-only except oracle.',**score))
    return rows


def run_all(outdir=None,sizes_ab=(3,4,6,16,64,128,256),sizes_c=(4,6,8,32,64,256),
            unitary_sizes=(3,4,6,16),quick=False):
    out=Path(outdir or ROOT/'factor_tdvp_comparison/results');out.mkdir(parents=True,exist_ok=True)
    rows=[];histories=[]
    def retain(batch):
        rows.extend(batch);write(out/'lifts',rows)
        for row in batch: print(json.dumps(row,default=scalar),flush=True)
    for n in sizes_ab:
        for family in ('cluster','nonclifford'):
            for method in ('half_petz','local_unitary_regularized'):
                if method=='local_unitary_regularized' and n not in unitary_sizes: continue
                row,history=repeated(n,family,method)
                retain([row]);histories+=history;write(out/'lifts_history',histories)
    # Retain the null-completion stress control; this is not the established
    # regularized repeated local-unitary baseline above.
    for n in (4,6):
        row,history=repeated(n,'cluster','local_unitary_unregularized_adapter')
        row['resource_control']='unregularized_null_completion';retain([row])
    for n,chi in ((4,16),(6,64)):
        row,history=repeated(n,'cluster','local_unitary_regularized',chi=chi)
        row['resource_control']='larger_unitary_bond';retain([row])
    row,history=repeated(6,'cluster','local_unitary_regularized',chi=64,window=5)
    row['resource_control']='larger_unitary_window';retain([row])
    for n in sizes_c:
        for theta in (0.,.05,.1,.2):
            retain(tilted(n,theta,3))
        retain(tilted(n,.1,5))
    # Resource controls for common six-spin inputs.
    for family in ('cluster','nonclifford'):
        row,history=repeated(6,family,'half_petz',chi=2)
        row['resource_control']='undersized_bond';retain([row])
        row,history=repeated(6,family,'half_petz',steps=40)
        row['resource_control']='time_refinement';retain([row])
    return rows


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir',default=str(ROOT/'factor_tdvp_comparison/results'))
    parser.add_argument('--quick',action='store_true')
    args=parser.parse_args()
    if args.quick: run_all(args.outdir,(3,4,6),(4,6,8),(3,4,6))
    else: run_all(args.outdir)
