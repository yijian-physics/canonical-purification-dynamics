"""Single-state six-spin controls for force compression and CG convergence."""
import handoff_trial as trial
import numpy as np
from pathlib import Path

mpo = trial.mpo
tn = trial.tn


def full_space_cg(q, rhs, preconditioner, restart):
    def apply(x):
        return q @ x + x @ q
    def pre(x):
        return preconditioner @ x @ preconditioner
    solution = np.zeros_like(q)
    residual = rhs.copy()
    z = pre(residual)
    direction = z.copy()
    rz = np.vdot(residual, z).real
    rows = []
    for i in range(1, 301):
        operator_direction = apply(direction)
        alpha = rz / np.vdot(direction, operator_direction).real
        solution += alpha * direction
        residual -= alpha * operator_direction
        true_residual = np.linalg.norm(rhs - apply(solution)) / np.linalg.norm(rhs)
        if i in (60, 180, 300) or true_residual <= 1e-6:
            rows.append(dict(check='full_space_cg', restart=restart, iterations=i, residual=true_residual))
        if true_residual <= 1e-6:
            break
        if restart and i % restart == 0:
            residual = rhs - apply(solution)
            direction = pre(residual)
            rz = np.vdot(residual, direction).real
        else:
            z = pre(residual)
            next_rz = np.vdot(residual, z).real
            direction = z + (next_rz/rz)*direction
            rz = next_rz
    return rows


def main():
    state = mpo.product_root(6, 1.)
    for _ in range(5):
        state = trial.kraus_step(state, .01, 2, 16)
    state = mpo.hermitize(state, 16)
    q = trial.matrix(state)
    model = trial.models.model(6)
    rho = q @ q
    rhs = np.zeros_like(q)
    for jump in model.jumps:
        number = jump.conj().T @ jump
        rhs += jump @ rho @ jump.conj().T - (number @ rho + rho @ number)/2
    val, basis = np.linalg.eigh(q)
    assert val[0] > 0
    exact = basis @ ((basis.conj().T @ rhs @ basis)/(val[:,None]+val[None,:])) @ basis.conj().T
    preconditioner = trial.models.kron(mpo.product_preconditioner(state))
    rows = []
    for chi in (16,24,32,48,64):
        retained = trial.matrix(tn.from_dense(tn.to_interleaved(exact,6),6,chi))
        rows.append(dict(check='compress_exact_force',chi=chi,
                         residual=np.linalg.norm(q@retained+retained@q-rhs) / np.linalg.norm(rhs),
                         relative_force_error=np.linalg.norm(retained-exact)/np.linalg.norm(exact)))
    for restart in (12,0):
        rows.extend(full_space_cg(q, rhs, preconditioner, restart))
    for row in rows:
        row.update(time=.05,lambda_min=val[0],condition=val[-1]/val[0])
    out=Path(__file__).parent/'handoff_results'
    trial.observables.save(out/'force_controls.csv',rows)
    print(rows,flush=True)


if __name__=='__main__':
    main()
