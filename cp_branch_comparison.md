# Local unitary, Hamiltonian and constrained-trace searches

An open cluster chain with individual $Z_j$ dephasing steps is the reference. A doubled MPS site pairs the physical and reference qubits. The noise parameter is $p=(1-e^{-h})/2$, where $h=\Delta t$ and the rate is one. The tests distinguish three operations: a unitary on a block of the doubled chain, a rotation of the reference and Kraus label after applying the channel, and direct optimization of the two MPS tensors.

## A local SVD that selects the canonical amplitude

Let $S$ be the current square-root amplitude. Apply the physical channel as a purification with one extra label $e$:

$$
K=[K_0,K_1]=[\sqrt{1-p}\,S,\sqrt p\,Z_j S],\qquad KK^\dagger=\sigma .
$$

A right unitary on its enlarged reference leaves this density exactly unchanged while all labels are retained. Designate label zero as the column block to keep. Its amplitude after the rotation can be written $X=KW$, where $W^\dagger W=I$; these columns can be completed to a unitary. With no restriction on $W$, maximize $\operatorname{Re}\operatorname{Tr}KW$. For the thin SVD $K=L D R^\dagger$, $W=R L^\dagger$ achieves the upper bound $\operatorname{Tr}D$, gives $KW=L D L^\dagger=\sqrt{\sigma}$, and leaves no weight in the other label. This is the polar-decomposition form of optimizing a purification overlap with the vectorized identity. See [Uhlmann's original purification-overlap formulation](https://doi.org/10.1016/0034-4877(76)90060-4).

For a block $B$ containing $j$, restrict $W$ to stack local blocks $w_e$ acting on the reference qubits in $B$, with identity on the spectators. The trace objective then depends only on

$$
M_B=[\operatorname{Tr}_{\bar B}K_0,\operatorname{Tr}_{\bar B}K_1]
    =[k_0 S_B,k_1 S_B],\quad S_B=\operatorname{Tr}_{\bar B} S ,
$$

where $k_0=\sqrt{1-p}\,I_B$, $k_1=\sqrt p\,Z_j$. The partial trace is of the **amplitude**, not the physical density. An SVD $M_B=L_B D_B R_B^\dagger$ gives the exact local trace maximizer $w=R_B L_B^\dagger$. Split it into block rows $w_0,w_1$, then update $X=\sum_e k_e S w_e$, extending the local factors by identities. On a three-qubit reference block the search matrix is only $8\times16$. The MPS implementation contracts spectator trace environments and applies this map on the active block, followed by local bond truncation. It does not construct the global density or its square root.

Why it is exact here: write $S=U(\bigotimes_i s_i)U^\dagger$, where $U$ is the cluster CZ circuit and the physical Kraus operators commute with $U$. In the product basis, the one-qubit polar isometry for $[k_0s_j,k_1s_j]$ achieves the global trace maximum when tensored with spectator identities. Conjugation by $U$ spreads each of its two block rows only to $j-1,j,j+1$, clipped at the boundary. The local search thus contains a global optimum, including when spectator factors are rank deficient. This proof applies to sequential, inhomogeneous dephasing and is not provided to the optimizer.

For a general input, the local optimum need not be a global one. Before bond truncation, the probability outside the retained label is $\ell=1-\|X\|_F^2$; the omitted physical operator $\sigma-XX^\dagger$ is PSD and has trace $\ell$. This measures physical loss, not the sign or gauge of $X$. Even zero leakage does not by itself certify canonicality. Normalization after selection is recorded separately from bond truncation.

## Numerical checks

- At $N=6$, the 18 sequential updates from the pure state used $h=0.1$, bond cap four, and a three-site block in the bulk. The direct MPS algorithm's final vector error versus an independent cluster CP was $3.37\times10^{-15}$.
- At $N=64$, the same MPS algorithm ran 192 updates. The maximum discrepancy of the normalized CP overlap from one at sweep ends was $1.78\times10^{-15}$; the final discrepancy was $6.66\times10^{-16}$. The largest discarded squared Schmidt weight per update was $4.69\times10^{-30}$. The overlap is not a vector-error measurement at that precision. The reference MPS was built separately by preparing the analytic local factors and then applying doubled CZ gates.
- A separate dense diagnostic optimized **every doubled-space unitary on a given block** against the exact CP target by orthogonal Procrustes. In the first bulk step the globally optimal CP errors for one site, the left pair, the right pair and the triplet were respectively $0.21946,0.21946,0.21946,2.4\times10^{-15}$. Starting at uniform $t=0.3$, they were $0.04982,0.04982,0.04982,1.8\times10^{-14}$. This diagnostic uses the exact target. The earlier result that a single MPS tensor suffices is consistent: its entries can implement a nonunitary filter.

## Hamiltonian search

To extract a generator, use the local polar output as the target and fit $e^{-i\alpha P}|S\rangle\!\rangle$ over all 4,096 Pauli words on the three doubled sites and a continuous angle. For any fixed word the angle follows from the two overlaps with $|S\rangle\!\rangle$ and $-iP|S\rangle\!\rangle$. This fits a generator to a branch selected by the polar step; it is not an independent positivity argument. There is no need to supply the known analytic generator.

On the full-rank sample the search found $P=Z^{a}_{j-1} Z^{p}_j Y^{a}_j Z^{a}_{j+1}$ (paired-site word $IZ|ZY|IZ$). This same word was held fixed across $h=10^{-1},10^{-2},\ldots,10^{-5}$ at both starting times. At $t=0$, $|\alpha|/\sqrt h=0.707106$ at $h=10^{-5}$; at $t=0.3$, $|\alpha|/h=0.551440$. Log-log slopes over the smallest three steps were $0.499964$ and $0.999762$. The largest vector error versus the independent dense CP over the ten angle fits was $6.41\times10^{-14}$. A bounded $O(h)$ rotation misses the leading change when a sector first gains population. At a single fixed step size, separate $\sqrt h$ and $h$ coefficients multiplying the same operator would not be identifiable.

## Direct two-tensor trace maximization

For an isolated bulk step, SLSQP optimized all real and imaginary entries of two neighboring MPS tensors at internal bond four. It maximized $\operatorname{Re}\operatorname{Tr}X/\sqrt{2^N}$ at $\|X\|_F=1$, subject to $\|XX^\dagger-\sigma\|_F/\|\sigma\|_F\le\epsilon$. A deterministic perturbation in the trace-gradient direction starts the search; each smaller tolerance continues from the previous solution. The square root is used for scoring, never in the objective. All six runs reported success and the measured residuals were within $10^{-6}$ relative of their specified tolerances. The optimizer is local.

| Relative density tolerance | CP error from pure state | CP error from full rank, $t=0.3$ |
|---:|---:|---:|
| $10^{-3}$ | $3.51\times10^{-2}$ | $5.20\times10^{-3}$ |
| $10^{-5}$ | $3.51\times10^{-3}$ | $8.75\times10^{-4}$ |
| $10^{-7}$ | $3.53\times10^{-4}$ | $8.74\times10^{-5}$ |

The constrained search is useful when the two tensors must be free variables, but it uses its tolerance: improving the density residual by a factor 100 improved CP error only about tenfold on the pure-start test. Its best found solution is not a certificate of global optimality. The polar method removes this particular tradeoff by optimizing over exact purifications of the channel output before selecting a label.

## Reproduction

Run `python cp_branch_methods.py --outdir branch_results` beside `cluster_dephasing.py` and `cp_tensor_search.py` with NumPy and SciPy installed. The optional `--part` argument accepts `gates`, `mps`, or `tensors`. Raw support, step-size, trajectory and trace-fit data accompany the script in `branch_results/`.
