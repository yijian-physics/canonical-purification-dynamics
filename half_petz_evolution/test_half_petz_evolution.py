"""Complete repeated-evolution benchmarks, with independent global CP references."""
from pathlib import Path
import argparse
import csv
import time
import half_petz_evolution as algorithm
import tdvp_core as tn
import cluster_dephasing as base
import numpy as np


def save(path, rows):
    with Path(path).open('w',newline='') as stream:
        keys = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(stream,fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def phase_angles(n,family):
    if family == 'cluster':
        return np.full(n-1,np.pi)
    return .27+.03*np.arange(n-1)


def reference(n,family,t,rates):
    state = base.MPS(n,max_bond=4)
    state.tensors = []
    for rate in rates:
        q = np.exp(-rate*t)
        plus = np.sqrt((1+q)/2)
        minus = np.sqrt(-np.expm1(-rate*t)/2)
        root = ((plus+minus)*base.I+(plus-minus)*base.X)/2
        state.tensors.append(root.reshape(1,4,1))
    for j,angle in enumerate(phase_angles(n,family)):
        diagonal = [np.exp(1j*angle*((a//2)*(b//2)-(a%2)*(b%2)))
                    for a in range(4) for b in range(4)]
        state.apply_gate(j,2,np.diag(diagonal))
    return state


def dense_reference(initial,family,total,rates):
    n = initial.n
    density = tn.to_matrix(initial.dense(),n)
    density = density@density.conj().T
    for site,rate in enumerate(rates):
        p = -np.expm1(-rate*total)/2
        z = algorithm.embed(base.Z,site,n)
        density = (1-p)*density+p*z@density@z
    values,vectors = np.linalg.eigh((density+density.conj().T)/2)
    root = (vectors*np.sqrt(np.maximum(values,0)))@vectors.conj().T
    return root,density


def run(n,family,steps,total,buffer,chi,cutoff,output):
    rates = .7+.15*np.cos(np.arange(n))**2
    initial = reference(n,family,0.,rates)
    started = time.perf_counter()
    solver = algorithm.HalfPetzEvolution(initial,buffer,chi,cutoff)
    setup_seconds = time.perf_counter()-started
    history = []
    evolution_seconds = 0.
    max_norm_defect = 0.
    truncation_sum = 0.
    spectral_mass = 0.
    largest_inverse = 0.
    max_error = 0.
    for step in range(steps):
        started = time.perf_counter()
        for site,rate in enumerate(rates):
            p = -np.expm1(-rate*total/steps)/2
            record = solver.update(site,[np.sqrt(1-p)*base.I,np.sqrt(p)*base.Z])
            max_norm_defect = max(max_norm_defect,abs(record.output_norm-1),abs(record.raw_norm-1))
            truncation_sum += record.truncation_bound
            spectral_mass += record.discarded_density_mass
            largest_inverse = max(largest_inverse,record.inverse_norm)
        evolution_seconds += time.perf_counter()-started
        exact = reference(n,family,(step+1)*total/steps,rates)
        error = tn.distance(solver.state,exact)
        max_error = max(max_error,error)
        history.append(dict(time=(step+1)*total/steps,cp_error=error,
                            norm=solver.state.norm(),max_norm_defect=max_norm_defect,
                            truncation_sum=truncation_sum,bond=solver.state.bond_dimension()))
    row = dict(n=n,family=family,steps=steps,time=total,buffer=buffer,chi=chi,cutoff=cutoff,
               cp_error=error,max_cp_error=max_error,max_norm_defect=max_norm_defect,
               truncation_sum=truncation_sum,spectral_mass=spectral_mass,largest_inverse=largest_inverse,
               cache_discarded_mass=solver.cache_discarded_mass,
               setup_seconds=setup_seconds,evolution_seconds=evolution_seconds)
    if n <= 6:
        root,rho = dense_reference(initial,family,total,rates)
        amplitude = tn.to_matrix(solver.state.dense(),n)
        exact_matrix = tn.to_matrix(exact.dense(),n)
        row['dense_root_cp_error'] = np.linalg.norm(amplitude-root)
        row['analytic_reference_dense_check'] = np.linalg.norm(exact_matrix-root)
        row['physical_density_error'] = np.linalg.norm(amplitude@amplitude.conj().T-rho)
    tag = f'{family}_n{n}_T{total:g}_s{steps}_b{buffer}_c{chi}_eps{cutoff:g}'
    save(output/(tag+'.csv'),[row])
    save(output/(tag+'_history.csv'),history)
    np.savez_compressed(output/(tag+'_state.npz'),
                        **{f'a{j}':a for j,a in enumerate(solver.state.tensors)})
    print(row,flush=True)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n',type=int,nargs='+',default=[6,16,64])
    parser.add_argument('--families',nargs='+',choices=['cluster','nonclifford'],default=['cluster','nonclifford'])
    parser.add_argument('--steps',type=int,default=20)
    parser.add_argument('--time',type=float,default=1.)
    parser.add_argument('--buffer',type=int,default=1)
    parser.add_argument('--chi',type=int,default=4)
    parser.add_argument('--cutoff',type=float,default=1e-13,
                        help='Relative cutoff on physical-factor singular values.')
    parser.add_argument('--out',type=Path,default=Path(__file__).parent/'half_petz_factor_results')
    args = parser.parse_args()
    if args.steps < 1 or args.time <= 0:
        parser.error('Time and number of steps must be positive.')
    args.out.mkdir(parents=True,exist_ok=True)
    rows = [run(n,family,args.steps,args.time,args.buffer,args.chi,args.cutoff,args.out)
            for n in args.n for family in args.families]
    save(args.out/'summary.csv',rows)


if __name__=='__main__':
    main()
