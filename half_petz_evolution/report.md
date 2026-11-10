# Repeated half-Petz evolution of a canonical-purification MPS

The code evolves the full CP amplitude by sequential onsite Kraus updates. It reaches 256 spins on two exactly locally reversible dephasing families and measures the global amplitude error against an independently constructed analytic MPS. The default test starts from a pure state and performs 20 complete left-to-right noise sweeps to t=1.

## Algorithm

For the active window W around the noisy site, set r=ρ_W and s=N_W(r). Apply

$$
\mathcal F_W(Q)=s^{-1/2}\mathcal N_W(r^{1/2}Q)
 =\sum_\mu (s^{-1/2}K_\mu r^{1/2})QK_\mu^\dagger .
$$

This forward map is derived from the half-Petz lifting construction. The inverse acts on the retained support. Each term is contracted into one finite MPS block, followed by SVD splitting at the requested spatial bond cap. This implementation applies the map directly, without a unitary fit, trace maximization, Hermitian projection or subsequent amplitude normalization. [Huang, Sang, Wang and Zou, Eq. (6) and the channel extension](https://arxiv.org/html/2610.41926v1)

The physical windows are initialized once by contracting the given input MPS. Their subsequent evolution uses the specified physical channels independently of the evolving CP. For onsite noise, channels outside a window disappear under partial trace. Each noise operation updates only the cached windows containing its site. Thus this cache supplies the correct physical data even if the numerical CP starts to drift.

Each physical window is stored as a factor r=ZZ†. Its channel update stacks [K₀Z,K₁Z,…], followed by a thin SVD that caps the factor rank at the window Hilbert-space dimension. Roots and supported inverses use these singular values. The default support cutoff is 10⁻¹³ relative to the largest factor singular value, corresponding to a density population cutoff near 10⁻²⁶. The code reports discarded physical mass, inverse norm, spatial truncation and amplitude norm drift.

In the implementation, if r=U D² U† and s=V E² V†, the left factor is evaluated as V[E⁻¹(V†KμU)D]U†. This avoids explicitly multiplying two independently ill-conditioned dense matrix functions. Factor inverses can still amplify roundoff near rank opening.

At fixed buffer and spatial bond, the repeated evolution has O(N) work per sweep. Its constants grow with window Hilbert-space dimension and spatial bond. The physical caches do not enumerate whole-chain Kraus histories.

## Benchmarks and global CP error

The input is V|+⟩⊗N. For the cluster family, V is the nearest-neighbor controlled-Z circuit. For the non-Clifford family, each bond j uses diag(1,1,1,exp(iθⱼ)), with θⱼ=0.27+0.03j. Both families have an explicit three-site recovery for onsite Z noise by undressing the affected spin, restoring its previous one-site density and redressing it.

The channel is

$$
\mathcal N_j(\rho)=(1-p_j)\rho+p_j Z_j\rho Z_j,\qquad
p_j=\frac{1-e^{-\gamma_j h}}2,\quad
\gamma_j=0.7+0.15\cos^2j .
$$

The exact endpoint is V[⊗ⱼ√((I+exp(−γⱼt)X)/2)]V†. The reference is built separately as an MPS and used only for scoring. The reported error is the raw global Frobenius norm ‖Qnumerical−Qexact‖F, computed by MPS subtraction and canonicalization, with no phase alignment or normalization adjustment.

All rows below use a buffer of one site on each side, state bond four, h=0.05 and T=1. There are 20N elementary updates.

| Spins | Cluster CP error | Non-Clifford CP error | Evolution seconds, cluster / non-Clifford |
|---:|---:|---:|---:|
| 6 | 2.12×10⁻¹⁴ | 3.10×10⁻¹⁴ | 0.129 / 0.134 |
| 16 | 4.58×10⁻¹⁴ | 4.64×10⁻¹⁴ | 0.365 / 0.352 |
| 64 | 9.78×10⁻¹⁴ | 1.55×10⁻¹³ | 1.51 / 1.53 |
| 128 | 1.16×10⁻¹³ | 2.45×10⁻¹³ | 2.82 / 2.96 |
| 256 | 3.00×10⁻¹³ | 5.40×10⁻¹³ | 6.22 / 6.16 |

Setup and reference scoring are excluded from the evolution times. The timed runs were sequential, with one BLAS thread. At 256 spins the largest amplitude norm defect was 2.62×10⁻¹³ for the cluster family and 5.33×10⁻¹³ for the non-Clifford family.

At six spins we also evolved the full density independently and took its dense positive root. This agrees with the analytic MPS reference to 2.14×10⁻¹⁵ and 8.16×10⁻¹⁵ at t=1. A separate complex full-rank four-spin test checks loss, dephasing and unitary channels, local-cache consistency, and exactness with a whole-chain window. Its maximum dense-versus-MPS update discrepancy is below 6×10⁻¹⁵.

## Controls

- Using 100 sweeps at N=64 gives global errors 3.35×10⁻¹³ and 7.37×10⁻¹³. Both benchmark lifts are exact at finite step size in exact arithmetic, so time refinement here mostly adds roundoff.
- Changing the factor support cutoff from 10⁻¹¹ to 10⁻¹⁵ gives identical N=64 final errors at the reported precision.
- A two-site buffer on each side also passes at N=16, with errors 1.04×10⁻¹³ and 1.57×10⁻¹³.
- Removing the spatial buffer gives errors 0.806 and 0.184 at N=6. Reducing the state bond to two gives errors 0.947 and 0.153. The norm and truncation diagnostics expose these deliberately undersized controls.
- For a single sweep to t=10⁻⁶, the six-spin factor implementation gives errors 1.31×10⁻¹⁰ and 4.75×10⁻¹¹. At t=10⁻⁸ they are 5.09×10⁻⁹ and 4.76×10⁻⁹. The largest local inverse norms then approach 3.6×10⁸ and 3.1×10⁹. Analytic factorized references score these tests because taking the square root of a numerically formed full density introduces reference errors around 3–4×10⁻⁸.

The complete engine accepts arbitrary onsite qubit Kraus channels. Its tested large-chain accuracy comes from these two controlled families. A quantitative approximation error for a wider class of locally reversible states requires the corresponding recovery bound and a sufficiently wide window. The code records numerical diagnostics rather than claiming a general global certificate. Interacting Hamiltonian evolution and finite-range dissipative gates would require extensions to the physical-marginal provider and block-channel interface.

## Reproduce and reuse

Python 3.11, NumPy 2.1.2 and SciPy 1.16.2 were used. The archive includes all imports, final MPS tensors, per-sweep histories and a checksum manifest.

~~~sh
python -m pip install -r requirements.txt
python validate_half_petz_evolution.py
python test_half_petz_evolution.py --n 6 16 64 128 256
python test_half_petz_evolution.py --n 64 --steps 100
python test_half_petz_evolution.py --n 6 --buffer 0
python test_half_petz_evolution.py --n 6 --chi 2
python test_half_petz_evolution.py --n 6 --steps 1 --time 1e-6 --out tiny
~~~

The reusable interface is HalfPetzEvolution(initial, buffer=1, chi=4), followed by update(site, kraus_operators) for each physical operation. The current amplitude is in solver.state. The exact future CP is absent from that interface.
