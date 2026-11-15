# Evolving an MPO factor of the canonical root

**Reading options:** [Typeset PDF](factor_tdvp_note.pdf) · [Benchmark comparison](../factor_tdvp_comparison/report.md)

This directory implements a **small-system, rank-aware factor-MPO TDVP prototype**. Its evolving variable is a generally non-Hermitian MPO $`H`$, with

```math
X=HH^\dagger\geq0,\qquad \rho_H=X^2.
```

Smooth steps solve a real least-squares problem in the genuine one-site MPO tangent space of $`H`$. Rank events take a finite CPTP physical-channel step, reconstruct a new factor from Kraus amplitudes, and audit the support before returning to smooth evolution. The smooth solver never obtains its force by solving a dense Sylvester equation for $`X`$, and never refits $`H`$ to a separately evolved root.

The implementation uses MPO products to construct the generator and Jacobian, but **exports the Jacobian columns into a dense real least-squares system**. Support detection, channel bridges, and independent validation also use dense small-system arithmetic. This is a finite-size implementation of the proposed variational equation, not a generic scalable MPO solver. The recorded experiments cover one to three spins. Global MPO contractions are part of the intended algorithm; replacing them by local marginals is not a goal of this study.

The code reuses the tensor core and MPO algebra in [`scalable_cp_witness/code`](../scalable_cp_witness/code). The independent baseline imports the original [`cp_tdvp/cp_tdvp_evolution.py`](../cp_tdvp/cp_tdvp_evolution.py). The support detector and finite factor bridge adapt the mathematics in [`automatic_rank_opening/report.md`](../automatic_rank_opening/report.md). The distinction from [`scalable_cp_witness/report.md`](../scalable_cp_witness/report.md) is substantive: that algorithm evolves a root candidate and subsequently fits or corrects a positive witness; this one solves directly for the factor velocity and updates the factor itself.

## Exact differential and variational equation

Let $`K=\dot H`$, $`X=HH^\dagger`$, and $`G=\mathcal L_t(X^2)`$. Product differentiation gives

```math
\dot X=D_H(K):=KH^\dagger+HK^\dagger,
```

**Equation (1).**

```math
\boxed{\mathcal J_H(K)=X D_H(K)+D_H(K)X=G.}
```

This identity holds without Hermiticity or positivity of $`H`$. The derivative $`D_H(K)`$ and the image $`\mathcal J_H(K)`$ are Hermitian. The map is **real-linear**, not complex-linear: generally $`\mathcal J_H(iK)\ne i\mathcal J_H(K)`$.

The implemented unconstrained factor velocity is

**Equation (2).**

```math
\boxed{K_*\in\arg\min_{K\in T_H\mathcal M_\chi}
\|\mathcal J_H(K)-\mathcal L_t((HH^\dagger)^2)\|_F^2.}
```

Here $`T_H\mathcal M_\chi`$ is the full one-site fixed-bond MPS tangent space for the interleaved operator $`H`$. This is **density-residual McLachlan TDVP**. If $`F`$ denotes the exact root velocity satisfying $`XF+FX=G`$, then the objective is

```math
\|S_X(D_H(K)-F)\|_F^2,\qquad S_X(V)=XV+VX.
```

In an $`X`$ eigenbasis it weights entry $`ab`$ of the root-velocity discrepancy by $`(x_a+x_b)^2`$. Thus it is not ordinary unweighted canonical-root-velocity TDVP. A small instantaneous density residual can coexist with a larger canonical-root error near small populations, and it does not include integration error.

## Gauge, null space, and the positive-gauge oracle

There are two distinct redundancies. Bond gauges in the MPS parameterization are removed when constructing an orthonormal physical tangent basis. The factor right-unitary freedom $`H\mapsto HU`$ remains:

```math
K=H\Omega,\qquad \Omega^\dagger=-\Omega
\quad\Longrightarrow\quad \mathcal J_H(K)=0.
```

For square invertible $`H`$, these are all the Jacobian null directions. Orthogonality to them in the real Frobenius metric is equivalent to

**Equation (3).**

```math
H^\dagger K=K^\dagger H.
```

Consequently an unrestricted minimum-Frobenius-norm solution is horizontal. The code uses rank-revealing real SVD least squares on an orthonormal tangent basis. It therefore chooses the minimum-norm solution **within the represented tangent**, orthogonal to null directions available inside that tangent. A global horizontal projection can leave a restricted MPO tangent; equation (3) is not imposed as an extra constraint. The reported `horizontal_defect` is a diagnostic, not a required convergence test for restricted bonds.

The positive gauge $`H=\rho^{1/4}>0`$ gives a different useful oracle. Restricting its derivative to Hermitian $`K`$ and diagonalizing $`H`$ yields

**Equation (4).**

```math
\boxed{K^{\rm pos}_{ab}=
\frac{G_{ab}}{(h_a+h_b)(h_a^2+h_b^2)}.}
```

This follows by differentiating $`H^4`$. In contrast, the full-space horizontal minimum-norm velocity is

**Equation (5).**

```math
K^{\rm hor}_{ab}=\frac{h_bG_{ab}}{(h_a^2+h_b^2)^2}.
```

Both give $`D_H(K)_{ab}=G_{ab}/(h_a^2+h_b^2)`$ and therefore the same physical derivative. The positive-gauge velocity need not be horizontal unless it commutes with $`H`$. Tests of (4) compare its root/density derivative with the variational result rather than incorrectly requiring equality of raw factor velocities.

At rank deficiency the kernel is larger, and a null direction cannot produce missing physical support to first order. The solver separately reports image rank/nullity, retained singular values, least-squares stationarity, absolute physical residual, relative physical residual, and velocity norm. Small normal-equation residual means stationarity of the least-squares problem; it does not mean the physical residual is small. The `ls_rcond` is explicit, with default $`10^{-11}`$; a small nonzero physical direction removed by this threshold is regularization, not proof that it is a gauge direction.

## MPO tangent implementation and scope

The physical/reference index at each site is ordered $`00,01,10,11`$, matching the existing core. Matrix multiplication and adjoints use `mpo_cp_force.multiply` and `mpo_cp_force.adjoint` with those conventions.

The one-site basis construction in [`tangent_solver.py`](tangent_solver.py) uses a left-isometric prefix, a local variation orthogonal to the left-isometry columns, and a right-isometric suffix. The last tensor is unrestricted. This gives a complex orthonormal basis $`B_j`$ for the fixed-bond tangent; $`B_j`$ and $`iB_j`$ give an orthonormal real basis. Every column is constructed separately as $`\mathcal J_H(B_j)`$ or $`\mathcal J_H(iB_j)`$ using MPO products. Real and imaginary vector entries are concatenated, and `numpy.linalg.lstsq` solves the resulting real system. No complex-linear conjugate-gradient assumption is used.

The generator is built from

```math
\begin{aligned}
X&=HH^\dagger,\qquad \rho=X X,\\
G&=-i[H_{\rm phys},\rho]+\sum_\mu
\left(L_\mu\rho L_\mu^\dagger-\tfrac12\{L_\mu^\dagger L_\mu,\rho\}\right).
\end{aligned}
```

The resulting tangent velocity is reconstructed as an MPO sum. A retracted explicit midpoint update modifies $`H`$ itself. This is not unconstrained dense evolution disguised as a tensor method: the bond-two control uses the smaller actual MPO tangent, and the basis is independently testable against the repository's tangent projector. Conversely, full-capacity small-chain tests have a tangent equal to the full operator space; those controls alone would not establish restricted-bond performance.

Intermediate products are rounded with an explicit work bond. By default it is the full finite-chain capacity $`4^{\lfloor N/2\rfloor}`$, so the default experiments have no imposed intermediate bond-cap approximation. Ordinary floating-point SVD effects remain. The code records local compression allowances separately; their sum is **not** claimed to be a propagated global bound after subsequent products. Dense arithmetic independently checks generator error, the applied final Jacobian image, velocity reconstruction, and the true physical residual for these small systems. These checks reveal hidden approximation from a deliberately reduced `work_bond`.

The explicit design matrix has $`2\,4^N`$ rows and twice the complex tangent dimension in columns. Both the dense output-space representation and the SVD are finite-size limitations. A hard `max_real_columns` guard prevents treating this as an unlimited large-chain implementation. A scalable extension would need a real-linear iterative tangent solver, controlled MPO applications and preconditioning; that extension is not demonstrated here.

The stored root is always the **uncompressed product** $`HH^\dagger`$. Its witness distance is exactly zero as a representation identity. An independently compressed replacement for $`X`$ would require recording $`\|X_{\rm compressed}-HH^\dagger\|_F`$; this implementation does not silently introduce one. Exact product bonds obey $`\chi_X\le\chi_H^2`$ and $`\chi_{\rho_H}\le\chi_H^4`$, before redundant bonds are removed. The tables report the raw stored product bond. The result files additionally give the bond and discarded allowance of a separately compressed diagnostic copy; it is never fed back into evolution. For the full two- and three-spin examples, raw root bond sixteen reduces to diagnostic bond four. A low-bond root does not imply a comparably low-bond quarter-root factor.

## Physical normalization and retraction

The required condition is

```math
T(H)=\mathrm{Tr}[(HH^\dagger)^2]=1,
```

not $`\mathrm{Tr}(H^\dagger H)=1`$. The retraction first compresses the factor and then rescales it by $`T^{-1/4}`$. Its implied root stays positive. The root change caused by normalization alone is exactly $`|\sqrt T-1|`$. Factor truncation, its induced root change, and normalization change are recorded separately. Positivity by construction does not prevent truncation from erasing a small physical sector; support is audited after retraction.

Normalization also has an infinitesimal consequence when (2) has nonzero residual. Since

```math
\dot T=\mathrm{Tr}\mathcal J_H(K)=4\mathrm{Re}\mathrm{Tr}(K^\dagger XH),
```

normalizing an Euler increment changes its first-order velocity to

**Equation (6).**

```math
\bar K=K_*-\alpha H,\qquad
\alpha=\frac{\mathrm{Tr}\mathcal J_H(K_*)}{4T}.
```

Its density residual becomes $`\mathcal J_H(K_*)-G-4\alpha\rho_H`$. The code reports both raw and normalization-corrected residuals. Equation (2) is the actual stage solve, while the physically normalized integrator includes this radial correction. It is not claimed to solve the different constrained least-squares problem with $`\dot T=0`$. For full tangent controls the trace defect is at numerical precision, and second-order convergence is observed. No general second-order claim is made for restricted-tangent retraction with substantial trace residual.

## Conditioning and nonpositive gauges

At positive diagonal $`H`$, the nonzero singular values of the full-space Jacobian restricted to horizontal directions are

```math
\sqrt2\,(h_a^2+h_b^2)^{3/2};
```

the diagonal entries give $`4h_a^3`$. Thus its nonzero condition number is

```math
\kappa(\mathcal J_H|_{\rm horizontal})
=(h_{\max}/h_{\min})^3=(p_{\max}/p_{\min})^{3/4}.
```

A restricted tangent can be worse. This physical conditioning is distinct from exact gauge zeros. The denominator in (4) has the same extremal scales. Full-rank states with small populations are therefore tracked separately from true support-opening events.

Allowing a non-Hermitian $`H`$ lets the least-squares solver choose a convenient gauge. The smooth noncommuting tests indeed develop nonzero factor non-Hermiticity while $`X`$ remains positive. Right-unitary gauges preserve all singular values of $`H`$, however; they cannot remove the quarter-root small-population scale or the genuine physical singularity. No general conditioning advantage or low-bond representation theorem follows from choosing a nonpositive gauge.

## Rank opening, delayed events, and the finite channel bridge

Let $`Q`$ project onto the kernel of the input density. Then

**Equation (7).**

```math
B=Q\mathcal L(\rho)Q
=\sum_\mu(QL_\mu X)(QL_\mu X)^\dagger\ge0.
```

The Hamiltonian and anticommutator terms vanish in this block. If $`B\ne0`$, newborn populations scale as $`h`$, root eigenvalues as $`h^{1/2}`$, and factor singular values as $`h^{1/4}`$. The corresponding positive-gauge derivative can diverge as $`h^{-3/4}`$. No choice of right-unitary gauge changes those singular values. A finite physical-time tangent update cannot create the missing leading direction from an exact kernel.

A zero instantaneous value of (7) does not rule out later opening. The implemented dense audit uses factor singular values, jump innovations, a finite channel probe, and an interaction-picture jump-word span diagnostic. For constant generator, set

```math
A=-iH_{\rm phys}-\tfrac12\sum_\mu L_\mu^\dagger L_\mu.
```

The code builds the numerical operator span generated from jumps by repeated $`\mathrm{ad}_A`$, then closes the input support under those operators. This tracks jump directions in the no-jump frame. Pure unitary support rotation is not classified as mixed-state rank growth. The numerical span uses thresholds; it is a conservative unresolved-sector diagnostic rather than an exact symbolic rank theorem. When the finite probe cannot resolve a sector suggested by this diagnostic, the state is explicitly classified as `delayed_opening_unresolved`.

The default support threshold is $`10^{-9}`$ relative to the largest **factor** singular value; its fourth power corresponds to the density population scale. The discarded support mass and threshold are reported. A tiny but resolved positive singular value is not automatically declared a kernel: full-rank exposure $`h|\dot\rho_{aa}|/p_a`$ is recorded as a stiffness diagnostic. The present driver flags this case but does not implement a general adaptive stiffness controller.

At an event, the handler takes a finite physical channel with Kraus operators $`K_\mu`$, forms

```math
Z'=[K_0X_{\rm old},K_1X_{\rm old},\ldots]=U\Sigma V^\dagger,
```

and sets

**Equation (8).**

```math
\boxed{H_{\rm seed}=U\Sigma^{1/2}U^\dagger.}
```

Indeed $`Z'Z'^\dagger=U\Sigma^2U^\dagger`$, so $`H_{\rm seed}H_{\rm seed}^\dagger=U\Sigma U^\dagger=\sqrt{Z'Z'^\dagger}`$. The SVD acts on Kraus amplitudes, rather than first squaring them into density eigenvalues. Roundoff-scale amplitude singular values may still be removed; the removed factor singular values, root norm, and density mass are explicit diagnostics. There is no hidden division by zero or inverse of a newborn population.

For the commuting dephasing controls, the bridge uses analytic exact local channels, composed with the commuting unitary. The general dense backend exponentiates the finite Lindblad superoperator and reshuffles it into a Choi matrix. Eigensolver-scale Choi modes are removed and Kraus completeness is restored by right multiplication with $`(\sum K_\mu^\dagger K_\mu)^{-1/2}`$. This produces a CPTP repaired map in exact arithmetic. The superoperator difference from the computed exponential, negative/discarded Choi spectrum, and completeness defect are measured. There is no finite-time product-formula discretization error in these exact-exponential bridges, but there is floating-point and Choi-repair error. The driver never substitutes $`I+h\mathcal L`$ for a CPTP map.

After the channel bridge, and again after MPO conversion/truncation, the handler checks whether unresolved first-order or delayed sectors remain. Missing multijump sectors are not silently passed to the smooth solver. This prototype takes the supplied grid interval as its bridge; if support remains unresolved it raises `RankOpeningError` with diagnostics rather than automatically expanding a bridge. The user may provide a better-resolved bridge interval or increase accuracy/bond capacity. This guarded failure is an intended result for the very small delayed-opening stress case.

The driver audits each step and both smooth force stages. Supplying a new generator to the next `evolve` call triggers detection again. It also rejects numerical rank loss between stages/endpoints. This handles the recorded sequence that first dephases one site and later activates the second site. General smooth time-dependent generators would need an explicitly controlled time-dependent bridge/integrator; the current public interface handles constant generators and piecewise-constant changes.

The optional clock is $`u=(t-t_*)^{1/4}`$, with stage speed $`dt/du=4u^3`$. It is used **after a finite bridge**. For analytic finite-dimensional paths, a newborn population $`ch^m`$ gives factor scale $`c^{1/4}u^m`$, which motivates the clock. The driven loss-dark qubit has delayed onset rather than a first-order leakage term. A clock alone neither inserts absent tangent directions nor proves that all generator changes and numerical support losses are resolved.

## Error estimators and certified interpretations

All reported canonical-purification errors are global Frobenius distances

```math
\epsilon_{\rm CP}=\|HH^\dagger-\sqrt{\rho_{\rm exact}(t)}\|_F.
```

Independent dense Lindblad exponentials supply physical references only for scoring. Future exact roots never enter smooth factor updates. Dense channel actions at rank events are the explicitly stated bridge exception. The positive-gauge factor-error diagnostic compares $`\rho_H^{1/4}`$ with $`\rho_{\rm exact}^{1/4}`$; it does not compare a gauge-dependent raw $`H`$ with a positive target.

For $`E=X^2-\rho`$ and a valid $`M\succeq\rho^{-1}`$, the witness report specializes to

**Equation (9).**

```math
\|X-\sqrt\rho\|_F\le
\sqrt{\mathrm{Tr}(E^\dagger E M)},
```

because $`X=HH^\dagger`$ is exactly positive and the witness distance is zero. To see (9), write $`D=X-\sqrt\rho`$ and $`XD+D\sqrt\rho=E`$; in the left-$`X`$/right-$`\sqrt\rho`$ eigenbases, each denominator is at least the right root eigenvalue. If the independent density reference has trace-norm error $`\delta`$, add $`\sqrt{\|M\|_{\rm op}}\,\delta`$, requiring an inverse upper bound for the **true** density. A numerical inverse of an approximate reference does not by itself prove that assumption.

The finite-size scoring routine uses the inverse of the dense physical reference only when its spectrum is resolved as full rank. These are endpoint diagnostics under the numerical-reference assumption, not a scalable online certificate. For any positive target, Powers–Størmer also gives the gap-free bound

**Equation (10).**

```math
\|X-\sqrt\rho\|_F\le\sqrt{\|X^2-\rho\|_1}.
```

Both bounds are stored. The dense reference-root eigensolver threshold is reported through a discarded-root allowance, so tiny numerical kernel eigenvalues are not silently counted as exact physical sectors. No interval-arithmetic rounding guarantee is claimed.

For a continuously interpolated factor path, $`R=\mathcal J_H(\dot H)-\mathcal L_t(\rho_H)`$ would yield

```math
\|\rho_H(T)-\rho(T)\|_1
\le\|\rho_H(0)-\rho(0)\|_1+\int_0^T\|R(t)\|_1dt,
```

followed by (10). Retraction jumps and bridge errors must also be included. The code does not replace this integral by a few stage samples and call it a certified global bound. Local product-compression sums, stage residuals, and measured global errors are kept distinct.

## Recorded small experiments

The input definitions and grids are fixed in `run_experiments`. Full-rank qubit controls use $`\rho_0=(I+0.37\sigma_x+0.19\sigma_z)/2`$, $`H_{\rm phys}=0.43\sigma_x+0.19\sigma_z`$, and $`L=\sqrt{0.4}\,\sigma^-`$. The two-spin control entangles two mixed input spins with $`e^{-0.31i Z Z}`$, then evolves with an interacting noncommuting Hamiltonian and onsite loss. All figures below are measured outputs, not requested tolerances.

| Smooth control | Steps | Factor bond | Raw root-product bond | Global CP-root error |
|---|---:|---:|---:|---:|
| Qubit, $`t=0.2`$ | 5 | 1 | 1 | $`7.586\times10^{-6}`$ |
| Qubit, $`t=0.2`$ | 10 | 1 | 1 | $`1.889\times10^{-6}`$ |
| Qubit, $`t=0.2`$ | 20 | 1 | 1 | $`4.713\times10^{-7}`$ |
| Two spins, $`t=0.16`$ | 4 | 4 | 16 | $`1.495\times10^{-5}`$ |
| Two spins, $`t=0.16`$ | 8 | 4 | 16 | $`3.715\times10^{-6}`$ |
| Two spins, $`t=0.16`$ | 16 | 4 | 16 | $`9.260\times10^{-7}`$ |

Halving the step reduces these errors by approximately four. The qubit and two-spin factors become non-Hermitian; for the finest runs $`\|H-H^\dagger\|_F`$ is approximately $`0.00421`$ and $`0.00827`$. Their implied roots remain positive and normalized. This confirms the implementation does not require a positive gauge for its evolving variable.

The restricted two-spin run uses factor bond two and has final CP error $`6.668\times10^{-2}`$, against an initial CP representation error $`7.087\times10^{-2}`$. It includes initial factor TT-SVD truncation as well as restricted-tangent and integration errors; it is not an isolated tangent-error measurement. Increasing time resolution cannot by itself establish that bond two is adequate.

A separate three-spin control starts from an exactly represented factor of bond two (initial CP error $`4.32\times10^{-16}`$), then evolves with interacting Hamiltonian and loss. Its four/eight-step runs to $`t=0.08`$ give CP errors $`0.04505`$ and $`0.02448`$. These are **unsuccessful accurate-trajectory stress controls**. Maximum retained Jacobian condition numbers are $`9.42\times10^5`$ and $`2.97\times10^6`$, with factor-velocity norms about $`8.27`$ and $`21.72`$. Root changes from factor retraction reach $`0.04734`$ and $`0.02670`$, even though maximum instantaneous physical residuals are only $`0.00242`$ and $`0.00180`$. This directly exposes the need for adaptive bond selection, regularization, and retraction-error control. The decreasing endpoint errors alone do not establish a satisfactory restricted-bond method.

On the same three-spin initial state and eight-step grid, allowing factor bond four reduces the endpoint error to $`3.734\times10^{-6}`$, with maximum retained condition number $`24.80`$. The initial factor still has actual bond two, so its first one-site solve has a restricted tangent; the larger retraction cap permits subsequent growth to bond four. This is a prescribed-cap comparison, not an implemented adaptive two-site algorithm. It separates the poor restricted representation/conditioning from a failure of the factor identity or positivity mechanism.

| Rank-opening control | Bridge / smooth steps | Final factor rank | Global CP-root error |
|---|---:|---:|---:|
| Pure qubit dephasing, 8 intervals, $`t=0.2`$ | 1 / 7 | 2 | $`1.209\times10^{-5}`$ |
| Same, 16 intervals | 1 / 15 | 2 | $`2.942\times10^{-6}`$ |
| Same, 32 intervals | 1 / 31 | 2 | $`7.182\times10^{-7}`$ |
| Correlated three-spin cluster, $`t=0.08`$ | 1 / 4 | 8 | $`4.009\times10^{-5}`$ |
| Noise activated on successive sites, $`t=0.16`$ | 2 / 12 | 4 | $`2.705\times10^{-6}`$ |
| Driven loss-dark qubit, $`t=0.08`$ | 1 / 8 | 2 | $`3.337\times10^{-6}`$ |

The dephasing refinement uses the same first bridge time $`0.02`$, isolating the following smooth integration error. The correlated cluster bridge opens all eight physical sectors, with factor bond four and raw root-product bond sixteen. The repeated control records two separate events at the generator change. The loss-dark control starts at zero instantaneous leakage and is detected as delayed opening. A separate extremely short delayed bridge must return an explicit unresolved-support failure.

For qubit $`Z/\sqrt2`$ dephasing from $`|+\rangle`$, the exact new factor singular value is $`[(1-e^{-h})/2]^{1/4}`$. `quarter_root_scaling.csv` tests this from $`h=10^{-2}`$ down to $`10^{-8}`$ and records the ratio to $`h^{1/4}`$ and absolute reconstruction error. This checks the finite bridge itself, independently of later time integration.

On the same physical endpoint grid with a first bridge at $`0.005`$ and final time $`0.1`$, physical-time integration gives CP error $`1.271\times10^{-3}`$; quartic-clock integration gives $`1.212\times10^{-6}`$. This is evidence for this onset control, not a universal clock guarantee.

The baseline uses the existing one-site root TDVP with its dense Sylvester force and polar retraction, on the identical full-rank two-spin input and time grids:

| Steps | Factor TDVP CP error | Existing root TDVP CP error |
|---:|---:|---:|
| 4 | $`1.495\times10^{-5}`$ | $`1.648\times10^{-5}`$ |
| 8 | $`3.715\times10^{-6}`$ | $`4.095\times10^{-6}`$ |
| 16 | $`9.260\times10^{-7}`$ | $`1.021\times10^{-6}`$ |

Both use full small-chain expressivity: factor bond four reconstructs raw root bond sixteen, whereas the baseline stores root bond four. This is a consistency comparison, not evidence of a speed advantage. The explicit factor Jacobian is more expensive than the dense root-force control at these sizes; individual timings, including import overhead in the first baseline run, are retained in the tables.

The smallest smooth qubit endpoint has inverse-weighted bound $`1.019\times10^{-6}`$ and gap-free bound $`9.576\times10^{-4}`$, against actual CP error $`4.713\times10^{-7}`$. The delayed-loss endpoint has inverse-weighted bound about $`1.17\times10^{-2}`$ despite actual error $`3.34\times10^{-6}`$, illustrating poor conditioning rather than a failure of positivity. Stage residuals and truncation details are retained in `events.json`; small force residuals in the full tangent controls coexist with finite time-discretization error.

## Files and reproduction

Run from the repository root:

```sh
python -m pip install -r factor_tdvp/requirements.txt
OPENBLAS_NUM_THREADS=1 python -m unittest discover -s factor_tdvp -p 'test_*.py' -v
OPENBLAS_NUM_THREADS=1 python factor_tdvp/factor_tdvp.py --case all --outdir run_outputs/factor_tdvp
```

For individual groups, replace `--case all` by `--case smooth`, `--case opening`, or `--case repeated`, using a separate output directory for each. Each invocation overwrites the named result files in its output directory. The recorded environment was Python 3.12.14, NumPy 2.3.5, and SciPy 1.17.0. The local requirements file pins these measured numerical dependencies; the older root requirements file is unchanged. No validation under its older versions is claimed.

| File | Role |
|---|---|
| [`factor_tdvp.py`](factor_tdvp.py) | Rank-aware driver, named experiments, independent physical reference and scoring |
| [`tangent_solver.py`](tangent_solver.py) | One-site MPO tangent basis, real Jacobian solve, factor normalization and midpoint update |
| [`rank_bridge.py`](rank_bridge.py) | Dense support audit, CPTP finite-channel bridge, guarded handoff |
| [`baseline.py`](baseline.py) | Thin wrapper importing the original full-rank root TDVP |
| [`test_factor_tdvp.py`](test_factor_tdvp.py) | Unit and regression checks |
| [`test_rank_bridge.py`](test_rank_bridge.py) | Channel, delayed/repeated rank, stiffness and support-loss regressions |
| [`results/measurements.csv`](results/measurements.csv) | Endpoint errors, bonds, residual summaries, events and timings |
| [`results/measurements.json`](results/measurements.json) | Same measurements with environment metadata |
| [`results/events.json`](results/events.json) | Per-stage support, solver, bridge and truncation diagnostics |
| [`results/quarter_root_scaling.csv`](results/quarter_root_scaling.csv) | Newborn-factor quarter-power test |
| [`results/tests.txt`](results/tests.txt) and [`results/stdout.txt`](results/stdout.txt) | Actual test output and experiment stdout |

The completed regression run passes **21 tests**, covering the differential/Jacobian, gauge directions, tangent construction, physical normalization, positivity and reconstruction, finite CPTP bridge, delayed/repeated support events, and comparison with independent physical evolution. An initial unittest-discovery import failure was corrected before this successful run. The recorded runner output is the authority for executed test status. Numerical stress controls with inadequate accuracy are reported above rather than treated as passing accuracy benchmarks.

## What remains unresolved

This work establishes a working finite-size factor evolution and guarded repeated-rank mechanism. It does not establish a general efficient large-$`N`$ implementation. The main missing pieces are a real-linear iterative MPO tangent solver with controlled contractions, useful preconditioning for quarter-root populations, a general efficient mixed-MPO support representation/bridge, and a generic global root certificate with an affordable independent physical reference. One-site tangents do not by themselves provide adaptive spatial bond growth. Fixed-grid smooth evolution also needs an adaptive controller to manage stiffness, integration error, and large physically meaningful least-squares residuals.

Gauge choices can help tensor representation but cannot eliminate physical singular values. Time changes can regularize known onset powers but cannot replace support creation. Factor positivity solves the branch problem exactly for the represented root; it does not solve representability, conditioning, normalization error, or global dynamical accuracy.

## Repository delivery status

The implementation and measured results were committed locally on the new branch `research/factor-cp-tdvp`, based on `main` commit `4af4c9f9d9f5dbc03e896508592563abb2a74839`. Publishing was attempted but did not succeed: the Git CLI had no usable HTTPS credentials, and the connected GitHub service rejected branch creation with HTTP 403, `Resource not accessible by integration`. No remote branch or pull request was created by this run. An apply-ready Git patch accompanies the local files; it can be applied to the recorded base with `git am factor_tdvp.patch` on a new local branch. All numerical runs reported above were executed locally; the publication failure does not indicate an unexecuted test.
