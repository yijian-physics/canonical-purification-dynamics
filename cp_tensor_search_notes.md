# Two-tensor search for canonical-purification evolution

This is a small-system, exploratory cost-function comparison for an open six-qubit cluster state under onsite Z dephasing. The optimization variables are the complex entries of two adjacent MPS tensors, with internal bond dimension four. They are not parameters of a physical gate. The results establish representability in this example and expose differences between objectives; they do not establish a globally best cost or a scalable general algorithm.

## Setup and what support is actually needed

Write a normalized purification as |X>>, with physical density matrix XX† and canonical choice X = sqrt(rho). Each MPS site groups a physical qubit and its reference qubit, so its local dimension is four.

For U = product of nearest-neighbor CZ gates, the inhomogeneously dephased cluster state is

    rho({q_j}) = U [tensor product_j (I + q_j X_Pauli)/2] U†,
    q_j -> exp(-dt) q_j when site j is updated.

Here gamma = 1 and the Lindblad generator is (Z_j rho Z_j - rho)/2. Vectorizing the square root gives

    |sqrt(rho)>> = (U tensor conjugate(U)) tensor product_j v(q_j),
    v(q) = (a(q), b(q), b(q), a(q)),
    a(q) = [sqrt((1+q)/2) + sqrt((1-q)/2)]/2,
    b(q) = [sqrt((1+q)/2) - sqrt((1-q)/2)]/2.

The doubled CZ circuit is diagonal. Therefore the one-site, generally nonunitary filter

    F_j = diag(a(q'_j)/a(q_j), b(q'_j)/b(q_j),
               b(q'_j)/b(q_j), a(q'_j)/a(q_j))

commutes through that circuit and produces the exact CP update. The ratios are regular at every finite dephasing time; the q = 0 endpoint can be handled by continuity. Absorbing F_j into the physical index of just tensor j is sufficient. All other tensors may remain fixed. This is a representability proof, not a filter supplied to the numerical optimizer.

Consequently the earlier three-site unitary CP generator does not impose a three-tensor requirement. Both adjacent pairs containing j work, and a single tensor works too. The exact CP has bond dimension at most four for all inhomogeneous q_j, including the intermediate states during a sweep.

An independent projection of the exact target into the frozen MPS environments tested one tensor, either adjacent pair, and a three-tensor block at initial times 0 and 0.3. All CP errors were below 1e-14. The spectral square root used for these checks discards eigenvalues below 5e-14 as numerical zeros.

## Variational objectives

At each step the physical target is sigma = E_j(XX†), computed from the current approximate state. Let X' be the normalized matrix encoded by the two new tensors and all frozen environments. The tested objectives were:

1. Exact CP reference: C_CP = ||X' - sqrt(sigma)||_F^2. Orthogonal-environment projection and an optimal local rank-four SVD solve this reference problem. It requires the target square root and serves as a diagnostic.
2. Density HS: C_rho = ||X'X'† - sigma||_F^2 / Tr(sigma^2).
3. Density plus Hermiticity: C_H = C_rho + ||(X' - X'†)/2||_F^2.
4. Density plus PSD distance: C_+ = C_rho + dist_F(X', PSD)^2. The PSD projection is obtained by taking the positive part of (X'+X'†)/2. This needs a dense spectral decomposition in this implementation; it is a diagnostic for the positive branch, not yet an efficient tensor-network prescription.
5. Trace preference: C_T = C_H - 1e-5 Re Tr(X')/sqrt(64). This favors the positive square root but trades against density accuracy at finite weight.

Hermiticity and PSD penalties have coefficient one. These are specific tested choices, not optimized hyperparameters. Density matching alone leaves purification freedom: X' = sqrt(sigma) W has exactly the same density for any unitary W. Hermiticity removes much of this freedom but still permits Hermitian square roots with different eigenvalue signs. PSD plus exact density matching selects sqrt(sigma).

Non-reference objectives use L-BFGS-B on the real and imaginary parts of the two tensors. Their internal bond is fixed at four throughout the optimization; a larger intermediate bond followed by truncation is not used. All objective gradients and the factorization chain rule passed directional finite-difference checks. The merged-tensor gradient check discrepancies were below 7e-11.

This prototype contracts dense 64-by-64 amplitude matrices. The density and Hermiticity scalar objectives can be written as tensor-network contractions without a global square root, but that scalable implementation is not provided here. The exact generator/filter is never used in a variational update; exact roots are used for initialization, the reference objective, and diagnostics.

## Results

Errors are physical trace distance (one half the trace norm) and CP vector distance after removing a global phase. All runs used N = 6, dt = 0.1 and bond dimension four. Reported non-reference fits are the best found by the specified starts and iteration limits, not certified global optima.

For one noisy site starting from the pure cluster state:

| Objective and initialization | Density trace distance | CP error |
|---|---:|---:|
| Exact CP reference | 1.49e-15 | 5.88e-15 |
| Density HS, unchanged tensors only | 4.76e-2 | 2.19e-1 |
| Density HS, warm start plus two perturbations | 4.61e-7 | 3.67e-1 |
| Density + Hermiticity, warm plus perturbations | 2.36e-6 | 1.53e-3 |
| Density + PSD distance, warm plus perturbations | 1.89e-6 | 1.37e-3 |

The last two isolated fits reached their 500-iteration limit. The important counterexample is the density-only fit: excellent physical agreement does not certify proximity to the canonical purification.

The unchanged pure state is stationary for the density objective. Indeed, with orthogonal projectors P = |C><C| and Q = Z_j P Z_j, sigma = (1-p)P + pQ. At X' = P, the unconstrained gradient is radial, proportional to P; normalization removes it. Both additional squared penalties also have zero gradient there. The new CP component has size sqrt(p), explaining why a naive infinitesimal warm start misses it. Perturbed starts help but do not guarantee the correct branch.

For three complete site-by-site sweeps, starting pure, each update used the previous tensors and one reproducible random perturbation, with a 600-iteration limit. Noise was applied only once per site per sweep; the last site used the left pair and other sites the right pair. At the end every site had accumulated time 0.3:

| Objective | Global density trace distance | Global CP error |
|---|---:|---:|
| Exact CP reference | 4.40e-15 | 8.31e-15 |
| Density HS | 1.20e-2 | 1.12 |
| Density + Hermiticity | 3.10e-3 | 1.18 |
| Density + PSD distance | 1.86e-4 | 4.22e-4 |
| Density + Hermiticity + trace preference | 1.41e-3 | 3.48e-3 |

Some optimizations reached their iteration cap. The Hermitian run acquired negative square-root eigenvalues: at the final step its negative-part norm was 0.591 despite a local density trace distance of 1.11e-7. This is not a bond-truncation effect. The PSD penalty controlled this branch problem much better in the tested run, although errors from early fits accumulated.

To separate initialization from continuation, another set of 18 updates started from the exact full-rank CP at uniform time 0.3, ending at 0.6. It used only the previous tensors as the starting point, with no random perturbations:

| Objective | Final global density trace distance | Final global CP error |
|---|---:|---:|
| Exact CP reference | 3.48e-15 | 5.03e-15 |
| Density HS | 1.17e-8 | 1.33e-7 |
| Density + Hermiticity | 2.03e-8 | 3.95e-8 |
| Density + PSD distance | 1.34e-8 | 2.32e-8 |
| Density + Hermiticity + trace preference | 5.04e-5 | 1.33e-4 |

The maximum CP error along the Hermiticity-penalized full-rank trajectory was 4.86e-8. Good continuation in this commuting cluster example is encouraging, but is not a general guarantee that Hermiticity preserves the positive branch. The finite trace preference causes a measurable bias even when continuation was otherwise easy. Providing the exact CP at time 0.3 in this diagnostic does not solve initialization for a general algorithm.

## Consequences for a TDVP cost

The exact-CP distance is the reference objective when the aim is controlling CP error. A small physical-state error alone cannot replace that diagnostic, because purification freedom is invisible to the physical density matrix. Exact canonicality at every step is not required, but the chosen error controls must bound the drift that matters.

At full rank, set S = sqrt(rho) and G_j = L_j(rho). The canonical velocity Y_j is the unique Hermitian solution of

    S Y_j + Y_j S = G_j.

The CP Hilbert-space TDVP objective is then

    minimize over admissible tensor velocities: ||dot(X) - Y_j||_F^2.

This follows by expanding the finite-step exact-CP objective. Gauge fixing and the tangent-space construction must be specified when turning it into an integrator. The general MPS tangent-projector integration framework is described by Haegeman et al., https://arxiv.org/abs/1408.5056; the noisy CP specialization and experiments above are derived here.

Density-residual minimization gives a different metric. For Hermitian variations and eigenvalues s_a of S,

    ||S dot(X) + dot(X) S - G_j||_F^2
      = sum_ab (s_a + s_b)^2 |dot(X)_ab - (Y_j)_ab|^2.

Thus it underweights errors involving small eigenvalues compared with CP distance. Formally applying the inverse Sylvester operator to the residual recovers the CP metric, but efficiently approximating that inverse is an unresolved computational step. At rank changes, the differentiable full-rank formula itself need not apply; a finite-step initialization is required.

For the next algorithmic version, use exact CP distance to judge each candidate cost on this solvable family. Density plus Hermiticity is a promising inexpensive continuation objective in the regular regime. Positive-branch control and the first finite steps require separate treatment; the PSD-distance experiment provides a useful small-system benchmark for that treatment. This experiment does not yet prove that finite Markov length supplies an efficient local approximation of every quantity needed by the general method.

## Reproduction

Keep cp_tensor_search.py beside cluster_dephasing.py. Dependencies are NumPy, SciPy and Matplotlib. Run from the outputs directory:

```powershell
python cp_tensor_search.py --outdir cp_search_pure
python cp_tensor_search.py --outdir cp_search_full_rank --initial-time 0.3 --trajectory-only --restarts 0
```

Each result directory contains CSV data, a JSON summary and a three-panel plot. The pure-state search used fixed random seeds (42 for isolated tests, 1000 plus update index for sequential steps). Factor coordinates and floating-point/library differences can affect nonconvex search trajectories, so the exact failure numbers should not be treated as universal constants.
