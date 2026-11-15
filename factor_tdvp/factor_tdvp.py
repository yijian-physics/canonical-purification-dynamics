"""Rank-aware small-system MPO factor TDVP; run --case all for measurements.

The evolving state is H, with X=HH† and rho=X². Smooth velocities come
from a real least-squares solve in the one-site MPO tangent of H. Dense
support audits and finite CPTP bridges are explicit finite-size components.
Exact reference densities and roots are used only by score/run_experiments.
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy
from scipy.linalg import expm

try:
    from . import tangent_solver as ts
    from . import rank_bridge as rb
except ImportError:
    import tangent_solver as ts
    import rank_bridge as rb


I = np.eye(2, dtype=complex)
PAULI_X = np.array([[0, 1], [1, 0]], complex)
Z = np.diag([1., -1.]).astype(complex)
SM = np.array([[0, 1], [0, 0]], complex)


def embed(op, site, n):
    result = np.ones((1, 1), complex)
    for j in range(n):
        result = np.kron(result, op if j == site else I)
    return result


def positive_power(rho, power):
    """Dense initialization/validation oracle, never a smooth stage force."""
    values, vectors = np.linalg.eigh(rb.herm(rho))
    if values[0] < -1e-12 * max(1., values[-1]):
        raise ValueError("Nonpositive density supplied to validation oracle.")
    threshold = 32 * np.finfo(float).eps * len(rho) * max(values[-1], 1e-300)
    keep = values > threshold
    return (vectors * np.where(keep, np.maximum(values, 0.)**power, 0.)) @ vectors.conj().T


def numerical_rank(factor, rtol=1e-9):
    singular = np.linalg.svd(factor, compute_uv=False)
    return int(np.sum(singular > rtol * singular[0]))


def evolve(factor, hamiltonian, jumps, times, chi, *, clock="physical",
           factor_rtol=1e-9, rcond=1e-11, work_bond=None,
           velocity_solver=None, solver_options=None):
    """Evolve a fixed generator on an explicit grid, auditing every stage.

    ``times`` includes its initial physical time. For piecewise generators,
    call again with the previous H and a new generator; the initial audit
    is repeated automatically. A failed support audit raises RankOpeningError
    with diagnostics. No inverse singular density/root eigenvalues are used.
    velocity_solver optionally replaces the smooth force backend; the dense
    finite-size support audits and positive channel bridge remain unchanged.
    """
    times = np.asarray(times, float)
    if len(times) < 2 or np.any(np.diff(times) <= 0):
        raise ValueError("times must contain at least two increasing endpoints")
    if clock not in ("physical", "quartic"):
        raise ValueError("clock must be physical or quartic")
    n = factor.n
    ham_mpo = ts.matrix_to_mpo(hamiltonian, n)
    jump_mpos = [ts.matrix_to_mpo(jump, n) for jump in jumps]
    state = factor
    records = []
    clock_origin = times[0]

    for start, end in zip(times[:-1], times[1:]):
        started = perf_counter()
        dt = float(end-start)
        dense_before = ts.matrix(state)
        before = rb.audit_support(dense_before, hamiltonian, jumps, dt,
                                  factor_rtol=factor_rtol)
        if not before["safe_handoff"]:
            seed, bridge_info = rb.bridge_factor(dense_before, hamiltonian, jumps,
                                                  dt, factor_rtol=factor_rtol)
            raw_seed = ts.matrix_to_mpo(seed, n)
            candidate, retract_info = ts.retract_factor(raw_seed, chi)
            after = rb.audit_support(ts.matrix(candidate), hamiltonian, jumps,
                                     dt, factor_rtol=factor_rtol, probe=False)
            seed_rank = numerical_rank(seed, factor_rtol)
            if after["rank"] < seed_rank or not after["safe_handoff"]:
                raise rb.RankOpeningError("MPO seed truncation lost required physical support.",
                                           dict(bridge=bridge_info, after=after,
                                                retraction=retract_info))
            state = candidate
            record = dict(method="channel_bridge", event=before["mode"],
                          bridge=bridge_info, retraction=retract_info)
        else:
            def guard(stage):
                status = rb.audit_support(ts.matrix(stage), hamiltonian, jumps, dt,
                                          factor_rtol=factor_rtol, probe=False)
                if status["rank"] < before["rank"]:
                    raise rb.RankOpeningError("Smooth factor retraction caused numerical rank loss.", status)
                if not status["safe_handoff"]:
                    raise rb.RankOpeningError("Smooth stage requires another finite channel bridge.", status)
                return status

            if clock == "physical":
                coordinate_step, scales = dt, (1., 1.)
            else:
                u = (start-clock_origin)**.25
                v = (end-clock_origin)**.25
                coordinate_step = float(v-u)
                scales = (float(4*u**3), float(4*((u+v)/2)**3))
            candidate, info = ts.midpoint_step(state, ham_mpo, jump_mpos,
                                              coordinate_step, chi, rcond=rcond,
                                              work_bond=work_bond,
                                              stage_scales=scales, guard=guard,
                                              velocity_solver=velocity_solver,
                                              solver_options=solver_options)
            after = guard(candidate)
            state = candidate
            record = dict(method="factor_tangent_midpoint", event=before["mode"],
                          solver=info)
        record.update(time=float(end), step=dt, clock=clock,
                      rank_before=before["rank"], rank_after=after["rank"],
                      support_before=before, support_after=after,
                      h_bond=state.bond_dimension(),
                      x_product_bond=ts.root_mpo(state).bond_dimension(),
                      seconds=perf_counter()-started)
        records.append(record)
    return state, records


def reference_density(rho0, hamiltonian, jumps, time):
    """Independent dense Lindblad exponential used only for validation."""
    generator = rb.lindblad_superoperator(hamiltonian, jumps)
    return rb.herm((expm(time*generator) @ rho0.reshape(-1, order="F"))
                   .reshape(rho0.shape, order="F"))


def score(factor, reference):
    dense = ts.matrix(factor)
    root = dense @ dense.conj().T
    density = root @ root
    target = positive_power(reference, .5)
    error = density-reference
    eigenvalues = np.linalg.eigvalsh(rb.herm(reference))
    cutoff = 32*np.finfo(float).eps*len(reference)*max(eigenvalues[-1], 1e-300)
    reference_root_roundoff = float(np.sqrt(np.sum(np.maximum(eigenvalues[eigenvalues <= cutoff], 0))))
    gap_free = float(np.sqrt(np.sum(np.abs(np.linalg.eigvalsh(rb.herm(error))))))
    weighted = None
    if eigenvalues[0] > cutoff:
        # M=rho_ref^{-1}: valid small-system endpoint metric, no online use.
        weighted = float(np.sqrt(max(0., np.trace(error.conj().T @ error @ np.linalg.inv(reference)).real)))
    return dict(global_cp_root_error=float(np.linalg.norm(root-target)),
                density_frobenius_error=float(np.linalg.norm(error)),
                gap_free_root_bound=gap_free, inverse_weighted_root_bound=weighted,
                reference_root_roundoff_bound=reference_root_roundoff,
                positive_gauge_factor_error=float(np.linalg.norm(positive_power(density,.25)-positive_power(reference,.25))),
                factor_nonhermiticity=float(np.linalg.norm(dense-dense.conj().T)),
                minimum_root_eigenvalue=float(np.linalg.eigvalsh(rb.herm(root))[0]),
                normalization_error=float(abs(np.trace(density)-1)),
                factor_witness_distance=0.)


def _numbers(tree, key):
    if isinstance(tree, dict):
        for k, value in tree.items():
            if k == key and isinstance(value, (int, float)):
                yield value
            yield from _numbers(value, key)
    elif isinstance(tree, list):
        for value in tree:
            yield from _numbers(value, key)


def summarize(name, state, reference, history, *, n, steps, chi, time, seconds, **extra):
    root = ts.root_mpo(state)
    compact_root, root_rounding = ts.tn.compress(root,4**(n//2))
    row = dict(case=name, n=n, time=time, requested_steps=steps, chi_h_limit=chi,
               h_bond=state.bond_dimension(), x_product_bond=root.bond_dimension(),
               x_diagnostic_compressed_bond=compact_root.bond_dimension(),
               x_diagnostic_compression_bound=root_rounding["discarded_bound"],
               rank=numerical_rank(ts.matrix(state)),
               bridge_events=sum(r["method"] == "channel_bridge" for r in history),
               smooth_steps=sum(r["method"] == "factor_tangent_midpoint" for r in history),
               seconds=seconds, **score(state, reference), **extra)
    # Exact diagnostic names are supplied by the tangent backend.
    for key in ("force_residual", "relative_force_residual", "normalized_force_residual",
                "product_local_discard_sum", "factor_discarded_bound", "root_normalization_change",
                "root_truncation_error", "ls_nullity", "ls_retained_condition",
                "jacobian_column_error", "generator_mpo_error", "velocity_norm"):
        row["max_"+key] = max(_numbers(history,key),default=0.)
    return row


def run_experiments(case="all", outdir="factor_tdvp/results"):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    rows, histories = [], {}

    def run(name, rho0, ham, jumps, times, chi, clock="physical"):
        n = int(round(np.log2(len(rho0))))
        input_factor = positive_power(rho0, .25)
        state = ts.matrix_to_mpo(input_factor, n, chi=chi)
        state, _ = ts.normalize_factor(state)
        if numerical_rank(ts.matrix(state)) < numerical_rank(input_factor):
            raise rb.RankOpeningError("Initialization truncation lost physical support.",
                                      dict(rank_before=numerical_rank(input_factor),
                                           rank_after=numerical_rank(ts.matrix(state)),chi=chi))
        initial_error = score(state,rho0)["global_cp_root_error"]
        started = perf_counter()
        state, history = evolve(state, ham, jumps, times, chi, clock=clock)
        elapsed = perf_counter()-started
        reference = reference_density(rho0, ham, jumps, times[-1]-times[0])
        row = summarize(name,state,reference,history,n=n,steps=len(times)-1,chi=chi,
                        time=float(times[-1]),seconds=elapsed,clock=clock,
                        initial_global_cp_root_error=initial_error)
        rows.append(row)
        histories[name] = history
        print(json.dumps(row, sort_keys=True), flush=True)
        return state, history

    if case in ("all", "smooth"):
        rho1 = (I+.37*PAULI_X+.19*Z)/2
        for steps in (5, 10, 20):
            run(f"smooth_qubit_{steps}",rho1,.43*PAULI_X+.19*Z,
                [np.sqrt(.4)*SM],np.linspace(0,.2,steps+1),1)
        rho2 = np.kron(rho1,(I+.25*PAULI_X)/2)
        entangler = expm(-.31j*np.kron(Z,Z))
        rho2 = entangler@rho2@entangler.conj().T
        ham2 = .27*np.kron(Z,Z)+.31*(embed(PAULI_X,0,2)+embed(PAULI_X,1,2))
        jumps2 = [np.sqrt(.3)*embed(SM,j,2) for j in range(2)]
        for steps in (4, 8, 16):
            run(f"smooth_two_spin_{steps}",rho2,ham2,jumps2,np.linspace(0,.16,steps+1),4)
        run("restricted_two_spin_bond2",rho2,ham2,jumps2,np.linspace(0,.16,9),2)
        # Exactly representable initial H at bond two: isolate tangent error
        # from the deliberately truncated two-spin initial condition above.
        rho3 = (np.eye(8)+.3*np.kron(np.kron(Z,Z),Z)+.2*embed(PAULI_X,0,3))/8
        ham3 = .27*(embed(Z,0,3)@embed(Z,1,3)+embed(Z,1,3)@embed(Z,2,3))
        ham3 += .31*sum(embed(PAULI_X,j,3) for j in range(3))
        jumps3 = [np.sqrt(.3)*embed(SM,j,3) for j in range(3)]
        for steps in (4,8):
            run(f"restricted_three_spin_exact_initial_{steps}",rho3,ham3,jumps3,
                np.linspace(0,.08,steps+1),2)
        run("full_three_spin_same_initial",rho3,ham3,jumps3,np.linspace(0,.08,9),4)
        # Strict same-grid baseline: existing one-site X-TDVP with dense force.
        try:
            from .baseline import evolve_root
        except ImportError:
            from baseline import evolve_root
        for steps in (4, 8, 16):
            times = np.linspace(0,.16,steps+1)
            started = perf_counter()
            result, baseline_rows = evolve_root(rho2,ham2,jumps2,times,4)
            root = ts.matrix(result)
            target = reference_density(rho2,ham2,jumps2,.16)
            row = dict(case=f"baseline_x_tdvp_{steps}",n=2,time=.16,
                       requested_steps=steps,chi_x_limit=4,x_product_bond=result.bond_dimension(),
                       global_cp_root_error=float(np.linalg.norm(root-positive_power(target,.5))),
                       density_frobenius_error=float(np.linalg.norm(root@root-target)),
                       seconds=perf_counter()-started)
            rows.append(row)
            histories[row["case"]] = baseline_rows
            print(json.dumps(row,sort_keys=True),flush=True)

    if case in ("all", "opening"):
        plus = np.ones(2)/np.sqrt(2)
        pure = np.outer(plus,plus)
        for steps in (8, 16, 32):
            # Same finite seed time across refinements isolates smooth error.
            times = np.r_[0.,np.linspace(.02,.2,steps)]
            run(f"opening_qubit_{steps}",pure,np.zeros((2,2)),[Z/np.sqrt(2)],times,1,"quartic")
        for clock in ("physical", "quartic"):
            run(f"clock_{clock}",pure,np.zeros((2,2)),[Z/np.sqrt(2)],
                np.r_[0.,np.linspace(.005,.1,17)],1,clock)
        # A correlated three-spin cluster under all-site dephasing opens 8 sectors.
        n = 3
        indices = np.arange(2**n)
        bits = (indices[:,None] >> np.arange(n-1,-1,-1)) & 1
        psi = (-1.)**np.sum(bits[:,:-1]*bits[:,1:],axis=1)/np.sqrt(2**n)
        run("opening_cluster_three_spin",np.outer(psi,psi),np.zeros((8,8)),
            [embed(Z,j,n)/np.sqrt(2) for j in range(n)],np.r_[0.,np.linspace(.04,.08,5)],4,"quartic")
        scaling = []
        for dt in (1e-2,1e-3,1e-4,1e-5,1e-6,1e-8):
            seed, info = rb.bridge_factor(pure,np.zeros((2,2)),[Z/np.sqrt(2)],dt)
            singular = np.linalg.svd(seed,compute_uv=False)[-1]
            expected = (-np.expm1(-dt)/2)**.25
            scaling.append(dict(step=dt,newborn_h_singular=float(singular),
                                exact_quarter_root=expected,ratio_to_h_quarter=float(singular/dt**.25),
                                absolute_error=float(abs(singular-expected))))
        histories["quarter_root_scaling"] = scaling
        _write_csv(out/"quarter_root_scaling.csv",scaling)

    if case in ("all", "repeated"):
        psi = np.ones(4)/2
        rho0 = np.outer(psi,psi)
        state = ts.matrix_to_mpo(rho0,2,chi=1)
        ham = np.zeros((4,4))
        history = []
        reference = rho0
        started = perf_counter()
        for part, site in enumerate((0,1)):
            jumps = [embed(Z,site,2)/np.sqrt(2)]
            times = np.r_[part*.08,np.linspace(part*.08+.02,(part+1)*.08,7)]
            state, segment = evolve(state,ham,jumps,times,1,clock="quartic")
            history += segment
            reference = reference_density(reference,ham,jumps,.08)
        row = summarize("repeated_distinct_sites",state,reference,history,n=2,steps=len(history),
                        chi=1,time=.16,seconds=perf_counter()-started)
        rows.append(row)
        histories[row["case"]] = history
        print(json.dumps(row,sort_keys=True),flush=True)
        ground = np.diag([1.,0.])
        run("delayed_loss_dark_qubit",ground,.6*PAULI_X,[np.sqrt(.5)*SM],
            np.r_[0.,np.linspace(.04,.08,9)],1,"quartic")
        # Very small delayed bridge can lose Choi modes: it must stop explicitly.
        try:
            rb.bridge_factor(ground,.6*PAULI_X,[np.sqrt(.5)*SM],1e-7)
            histories["tiny_delayed_bridge"] = {"status":"resolved"}
        except rb.RankOpeningError as exc:
            histories["tiny_delayed_bridge"] = {"status":"guarded_failure","message":str(exc),
                                                  "diagnostics":exc.diagnostics}

    metadata = dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
                    scope="finite-size one-site MPO tangent; dense real Jacobian LS; dense rank bridge",
                    command=f"python factor_tdvp/factor_tdvp.py --case {case} --outdir {outdir}")
    (out/"measurements.json").write_text(json.dumps(dict(metadata=metadata,rows=rows),indent=2)+"\n")
    (out/"events.json").write_text(json.dumps(histories,indent=2)+"\n")
    _write_csv(out/"measurements.csv",rows)
    return rows, histories


def _write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=keys,lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case",choices=("all","smooth","opening","repeated"),default="all")
    parser.add_argument("--outdir",default="factor_tdvp/results")
    args = parser.parse_args()
    run_experiments(args.case,args.outdir)


if __name__ == "__main__":
    main()
