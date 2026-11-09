"""Exact canonical-purification evolution of an open 1D cluster state.

Dependencies: numpy, scipy, matplotlib. Run: python cluster_dephasing.py
Outputs are written beside this script, or to --outdir.

Convention: L(rho) = gamma/2 * sum_j (Z_j rho Z_j - rho).
Each MPS site pairs (physical, ancilla), in the order 00,01,10,11.
This is an exact solvable benchmark, not a generic local-noise solver.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.linalg import eigh

I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.diag([1., -1.]).astype(complex)
ZA = np.kron(I, Z)
ZY = np.kron(Z, Y)


def kron_all(terms):
    out = np.array([[1.]], dtype=complex)
    for term in terms:
        out = np.kron(out, term)
    return out


def theta(t, gamma=1.):
    # atan2 avoids loss of accuracy in sqrt(1 - exp(-2 gamma t)).
    return .5 * np.arctan2(np.exp(-gamma*t), np.sqrt(-np.expm1(-2*gamma*t)))


def cluster_signs(n):
    bits = (np.arange(2**n)[:, None] >> np.arange(n-1, -1, -1)) & 1
    signs = 1 - 2*(np.sum(bits[:, :-1]*bits[:, 1:], axis=1) % 2)
    return bits, signs


def channel_reference(n, t, gamma=1.):
    bits, signs = cluster_signs(n)
    distances = np.sum(bits[:, None, :] != bits[None, :, :], axis=-1)
    return np.outer(signs, signs) * np.exp(-gamma*t*distances) / 2**n


def analytic_root(n, t, gamma=1.):
    q = np.exp(-gamma*t)
    plus, minus = np.sqrt((1+q)/2), np.sqrt((1-q)/2)
    local = ((plus+minus)*I + (plus-minus)*X)/2
    _, signs = cluster_signs(n)
    return signs[:, None]*kron_all([local]*n)*signs[None, :]


def to_interleaved(matrix, n):
    order = [a for j in range(n) for a in (j, n+j)]
    return matrix.reshape([2]*(2*n)).transpose(order).reshape(-1)


def to_matrix(vector, n):
    order = list(range(0, 2*n, 2)) + list(range(1, 2*n, 2))
    return vector.reshape([2]*(2*n)).transpose(order).reshape(2**n, 2**n)


class MPS:
    def __init__(self, n, max_bond=4):
        self.n, self.max_bond, self.center = n, max_bond, 0
        self.tensors = [np.full((1, 4, 1), .5, dtype=complex) for _ in range(n)]
        self.discarded_weights = []

    def move_center(self, target):
        while self.center < target:
            j = self.center
            a = self.tensors[j]
            left, d, right = a.shape
            q, r = np.linalg.qr(a.reshape(left*d, right), mode='reduced')
            self.tensors[j] = q.reshape(left, d, q.shape[1])
            self.tensors[j+1] = np.tensordot(r, self.tensors[j+1], axes=(1, 0))
            self.center += 1
        while self.center > target:
            j = self.center
            a = self.tensors[j]
            left, d, right = a.shape
            q, r = np.linalg.qr(a.reshape(left, d*right).T, mode='reduced')
            self.tensors[j] = q.T.reshape(q.shape[1], d, right)
            self.tensors[j-1] = np.tensordot(self.tensors[j-1], r.T, axes=(2, 0))
            self.center -= 1

    def apply_gate(self, start, width, gate):
        self.move_center(start)
        block = self.tensors[start]
        for j in range(start+1, start+width):
            block = np.tensordot(block, self.tensors[j], axes=(-1, 0))
        left, right = block.shape[0], block.shape[-1]
        block = np.einsum('uv,lvr->lur', gate, block.reshape(left, 4**width, right))
        for offset in range(width-1):
            block = block.reshape(left*4, -1)
            u, s, vh = np.linalg.svd(block, full_matrices=False)
            # Retain numerical support, capped by the requested spatial bond.
            rank = max(1, min(self.max_bond, int(np.sum(s > 1e-13*s[0]))))
            self.discarded_weights.append(float(np.sum(s[rank:]**2)))
            self.tensors[start+offset] = u[:, :rank].reshape(left, 4, rank)
            block = s[:rank, None]*vh[:rank]
            left = rank
        self.tensors[start+width-1] = block.reshape(left, 4, right)
        self.center = start+width-1

    def dense(self):
        block = self.tensors[0]
        for a in self.tensors[1:]:
            block = np.tensordot(block, a, axes=(-1, 0))
        return block.reshape(-1)

    def norm(self):
        self.move_center(0)
        return float(np.linalg.norm(self.tensors[0]))

    def bond_dimension(self):
        return max(a.shape[-1] for a in self.tensors)

    def product_expectation(self, operators):
        env = np.ones((1, 1), dtype=complex)
        for j, a in enumerate(self.tensors):
            op = operators.get(j, np.eye(4))
            env = np.einsum('ab,asr,st,btu->ru', env, a.conj(), op, a)
        return env.item()


def prepare_cluster_cp(n):
    state = MPS(n)
    phase = np.empty(16)
    for u in range(4):
        for v in range(4):
            phase[4*u+v] = (-1)**((u//2)*(v//2)+(u%2)*(v%2))
    for j in range(n-1):
        state.apply_gate(j, 2, np.diag(phase))
    return state


def local_generator(n, j):
    start, stop = max(0, j-1), min(n, j+2)
    return start, stop-start, kron_all([ZY if k == j else ZA for k in range(start, stop)])


def step(state, t_old, t_new, gamma=1.):
    delta = theta(t_new, gamma)-theta(t_old, gamma)
    for j in range(state.n):
        start, width, generator = local_generator(state.n, j)
        gate = np.cos(delta)*np.eye(4**width)-1j*np.sin(delta)*generator
        state.apply_gate(start, width, gate)


def entropy(matrix):
    values = np.linalg.eigvalsh((matrix+matrix.conj().T)/2)
    values = values[values > 1e-13]
    return float(-np.sum(values*np.log(values)))


def marginal(rho, n, keep):
    keep = list(keep)
    rest = [i for i in range(n) if i not in keep]
    perm = keep+rest+[n+i for i in keep]+[n+i for i in rest]
    tensor = rho.reshape([2]*(2*n)).transpose(perm)
    tensor = tensor.reshape(2**len(keep), 2**len(rest), 2**len(keep), 2**len(rest))
    return np.trace(tensor, axis1=1, axis2=3)


def cmi(rho, n, buffer):
    # A = first site, B = next buffer sites, C = all remaining sites.
    ab = range(buffer+1)
    bc = range(1, n)
    b = range(1, buffer+1)
    return entropy(marginal(rho, n, ab))+entropy(marginal(rho, n, bc))-entropy(marginal(rho, n, b))-entropy(rho)


def apply_dense_local(vector, n, start, width, gate):
    left, middle, right = 4**start, 4**width, 4**(n-start-width)
    return np.einsum('uv,lvr->lur', gate, vector.reshape(left, middle, right)).reshape(-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    n, gamma = 6, 1.
    times = np.linspace(0., 3., 31)
    state = prepare_cluster_cp(n)
    rows = []
    largest_canonical_defect = 0.
    largest_root_reference_error = 0.
    schmidt_weights = []
    for k, t in enumerate(times):
        if k:
            step(state, times[k-1], t, gamma)
        vector = state.dense()
        amplitude = to_matrix(vector, n)
        rho = channel_reference(n, t, gamma)
        root = analytic_root(n, t, gamma)
        eigenvalues, eigenvectors = eigh(rho)
        # Remove only roundoff at the rank-one initial endpoint.
        if t == 0:
            eigenvalues[eigenvalues < 1e-13] = 0
        dense_root = (eigenvectors*np.sqrt(np.maximum(eigenvalues, 0)))@eigenvectors.conj().T
        largest_root_reference_error = max(largest_root_reference_error, float(np.linalg.norm(dense_root-root)))
        cp_error = float(np.linalg.norm(vector-to_interleaved(dense_root, n)))
        formula_error = float(np.linalg.norm(vector-to_interleaved(root, n)))
        delta_rho = amplitude@amplitude.conj().T-rho
        trace_distance = float(np.sum(np.abs(np.linalg.eigvalsh((delta_rho+delta_rho.conj().T)/2)))/2)
        hermitian_defect = float(np.linalg.norm(amplitude-amplitude.conj().T))
        largest_canonical_defect = max(largest_canonical_defect, hermitian_defect)
        s = np.linalg.svd(vector.reshape(4**(n//2), -1), compute_uv=False)
        schmidt_weights.append(s[:4]**2)
        row = dict(time=float(t), q=float(np.exp(-t)), cp_error_dense_root=cp_error,
                   cp_error_analytic_root=formula_error, trace_distance=trace_distance,
                   bond_dimension=state.bond_dimension(), cmi_b1=cmi(rho,n,1),
                   cmi_b2=cmi(rho,n,2), cmi_b3=cmi(rho,n,3))
        rows.append(row)

    # Independent differential check away from t=0.
    differential_errors = []
    for t in [.2, .7, 2.]:
        eps = 1e-5
        vector = to_interleaved(analytic_root(n,t), n)
        derivative = to_interleaved((analytic_root(n,t+eps)-analytic_root(n,t-eps))/(2*eps), n)
        summed = np.zeros_like(vector)
        for j in range(n):
            start, width, generator = local_generator(n,j)
            summed += apply_dense_local(vector,n,start,width,generator)
        theta_dot = -np.exp(-t)/(2*np.sqrt(-np.expm1(-2*t)))
        differential_errors.append(float(np.linalg.norm(derivative+1j*theta_dot*summed)))

    # A finite jump from the rank-deficient initial state needs no time subdivision.
    direct = prepare_cluster_cp(n)
    step(direct, 0., 3.)
    direct_error = float(np.linalg.norm(direct.dense()-state.dense()))

    # A larger chain uses MPS operations only, never its dense density matrix.
    large_n = 64
    large = prepare_cluster_cp(large_n)
    step(large, 0., 1.)
    j = large_n//2
    stabilizer = {j-1:np.kron(Z,I), j:np.kron(X,I), j+1:np.kron(Z,I)}
    measured = large.product_expectation(stabilizer)
    large_error = float(abs(measured-np.exp(-1)))

    summary = dict(n_dense=n, n_mps_only=large_n, gamma=gamma, times=len(times),
        max_cp_error_dense_root=max(r['cp_error_dense_root'] for r in rows),
        max_cp_error_analytic_root=max(r['cp_error_analytic_root'] for r in rows),
        max_trace_distance=max(r['trace_distance'] for r in rows),
        max_abs_cmi_buffer_ge2=max(abs(r[key]) for r in rows for key in ['cmi_b2','cmi_b3']),
        max_bond_dimension=max(r['bond_dimension'] for r in rows),
        max_discarded_weight=max(state.discarded_weights),
        max_hermiticity_defect=largest_canonical_defect,
        max_dense_sqrt_vs_analytic_error=largest_root_reference_error,
        generator_finite_difference_errors=differential_errors,
        one_step_vs_many_steps_error=direct_error,
        large_chain_bond_dimension=large.bond_dimension(), large_chain_norm=large.norm(),
        large_chain_stabilizer_error=large_error)
    assert summary['max_cp_error_dense_root'] < 1e-8, summary
    assert summary['max_cp_error_analytic_root'] < 1e-10, summary
    assert summary['max_trace_distance'] < 1e-10, summary
    assert summary['max_abs_cmi_buffer_ge2'] < 1e-10, summary
    assert max(differential_errors) < 1e-7, summary
    assert direct_error < 1e-10 and large_error < 1e-10, summary
    assert abs(summary['large_chain_norm']-1) < 1e-10, summary
    assert summary['max_bond_dimension'] <= 4 and large.bond_dimension() <= 4
    with (args.outdir/'cluster_dephasing_results.csv').open('w',newline='',encoding='utf-8') as file:
        writer = csv.DictWriter(file,fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    (args.outdir/'cluster_dephasing_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':11, 'axes.spines.top':False, 'axes.spines.right':False})
    fig, ax = plt.subplots(1,3,figsize=(14,4),layout='constrained')
    fig.suptitle('Onsite dephasing of an open 1D cluster state: exact canonical-purification MPS',fontsize=14)
    ax[0].semilogy(times,np.maximum([r['cp_error_analytic_root'] for r in rows],1e-16),label='CP vector error',color='#2563eb')
    ax[0].semilogy(times,np.maximum([r['trace_distance'] for r in rows],1e-16),label='Density trace distance',color='#d97706')
    ax[0].set(title='Local gates + MPS compression',xlabel=r'$\gamma t$',ylabel='Error',ylim=(1e-16,1e-10))
    ax[0].legend(fontsize=9)
    for j, color in enumerate(['#2563eb','#d97706','#059669','#9333ea']):
        ax[1].plot(times,np.array(schmidt_weights)[:,j],label=f'Level {j+1}',color=color,ls=['-','--',':','-.'][j])
    ax[1].set(title='Central-cut Schmidt probabilities',xlabel=r'$\gamma t$',ylabel=r'$s_k^2$',ylim=(0,1))
    ax[1].text(.5,.9,r'$\chi\leq 4$ for all $t$',transform=ax[1].transAxes,ha='center')
    ax[1].legend(fontsize=9)
    ax[2].plot(times,[r['cmi_b1'] for r in rows],label='Buffer = 1',color='#2563eb')
    ax[2].plot(times,[r['cmi_b2'] for r in rows],label='Buffer = 2',color='#d97706')
    ax[2].plot(times,[r['cmi_b3'] for r in rows],label='Buffer = 3',color='#059669',ls='--')
    ax[2].set(title='Physical-state conditional MI',xlabel=r'$\gamma t$',ylabel='CMI (nats)',ylim=(-.025,.73))
    ax[2].legend(fontsize=9)
    for axis in ax:
        axis.grid(alpha=.16)
    fig.savefig(args.outdir/'cluster_dephasing_benchmark.png',dpi=180)
    plt.close(fig)
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
