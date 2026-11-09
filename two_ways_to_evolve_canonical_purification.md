# Two ways to evolve a canonical purification under locally reversible noise

We are given the initial canonical purification and a **specified path of local physical channels**. The promise is that each step is locally reversible on the state encountered along that path, to the accuracy of interest. For a step on sites \(S\), this means that a recovery acting on \(S\) and a surrounding buffer can restore the pre-step state, including its correlations with the exterior. The recovery need only exist. The numerical task is to construct the evolution of \( |\Psi(t)\rangle=\operatorname{vec}\sqrt{\rho(t)} \), represented as an MPS with each physical spin paired with its reference.

## 1. Evolve the MPS tensors by TDVP

On a smooth, full-rank segment, write \(X=\sqrt{\rho}\). Differentiating \(X^2=\rho\) gives the force \(F=\dot X\) from the current state and Lindblad generator \(\mathcal L\),

$$
XF+FX=\mathcal L(X^2).
$$

Choose the variations of adjacent MPS tensors that minimize \(\|\dot\Psi-\operatorname{vec}F\|_2\) in the two-site variation space. The implementation projects the force, takes a midpoint step, splits the updated tensors by SVD, and normalizes. Two-site variations allow the spatial bond to grow. Step doubling checks time resolution, and the discarded Schmidt weights monitor spatial compression. [TDVP implementation and tests](cp_tdvp/report.md) · [MPS tangent projection](https://arxiv.org/abs/1408.5056)

**What changes at rank opening.** At time \(t_0\), let \(P_0\) project onto the kernel of \(\rho(t_0)\) and put \(B=P_0\mathcal L(\rho)P_0\). When \(B\ne0\), noise populates that kernel at first order, and

$$
X(t_0+h)=X(t_0)+\sqrt h\,\sqrt B+O(h).
$$

The algorithm must supply this new tensor direction. Our small-system implementation takes a finite channel step, stacks its Kraus factors as \(Z=[K_0X,K_1X,\ldots]\), and obtains the positive root \(U\Sigma U^\dagger\) from \(Z=U\Sigma V^\dagger\). It then integrates in \(u=\sqrt{t-t_0}\), where the initial derivative is finite. The clock resolves the time dependence. Computing the opening direction at affordable cost, and retaining it through truncation, are separate requirements. [Rank-opening construction](automatic_rank_opening/report.md)

**Test of time resolution.** Uniform loss of a single excitation has the exact root \(e^{-\gamma t/2}P_W+\sqrt{1-e^{-\gamma t}}P_{\rm vac}\) and requires at most bond four. At \(N=16\), \(\gamma=0.7\), \(T=1\), and 80 grid intervals, the global CP error fell from \(3.32\times10^{-4}\) on a uniform \(t\) grid to \(8.35\times10^{-7}\) on a uniform \(\sqrt t\) grid. Both runs used the known positive first step on their respective grids. This isolates the time singularity. A separate six-spin interacting-loss test exposed an additional cost after startup: the compressed MPO Sylvester solver failed its residual check on all nine attempted handoffs. That benchmark tested the solver without establishing a recovery radius for the trajectory. [Clock comparison](cp_tdvp/results/clock.csv) · [Interacting-loss test](kraus_tdvp_handoff/report.md)

## 2. Apply a finite unitary on doubled sites

For one physical step \(\rho'=\Phi_S(\rho)\), seek a unitary \(U\) acting on the **physical and reference spins** of \(W=S\cup B\), where \(B\) is the active buffer. Let \(E\) be the exterior. The channel–unitary correspondence bounds the best attainable global vector error by

$$
\min_{U_{W\bar W}}\big\|\,|\sqrt{\rho'}\rangle-U_{W\bar W}|\sqrt\rho\rangle\,\big\|_2
\ \le\ \sqrt{I_\rho(E:S\mid B)-I_{\rho'}(E:S\mid B)}
$$

with conditional mutual information in nats. Exact recovery on \(W\) makes the difference zero by data processing in both directions. For approximate recovery, the quantitative question is how wide \(B\) must be at the desired error. The bound concerns information **lost in this step**, so it can be small even when the initial conditional mutual information is large. [Li, Firanko and Hsieh, Corollary 13](https://arxiv.org/html/2605.00088v2#S5.SS2)

**How to find the gate.** Take a larger fitting window \(R\supset W\). Obtain its physical marginal \(\rho_R\), apply the known channel to get \(\rho'_R\), and form their canonical roots. Reshape the two root vectors into matrices \(L,T\), with the doubled sites of \(W\) as rows. Solve \(\min_{U^\dagger U=I}\|UL-T\|_F\) by taking a unitary polar factor of \(TL^\dagger\). Apply that gate to the current CP MPS and split by SVD. The width of \(W\) determines which gates can be represented. The extra sites in \(R\) determine how much exterior information the fit sees. A small fitting residual alone leaves the global error undetermined. [Finite-unitary algorithm and marginal construction](local_cp_large/report.md)

**Test of spatial support.** In a 64-site classical Ising chain under bit flips at \(t=0.5\), increasing the active gate from three to five doubled sites reduced the two-site CP marginal discrepancy from \(2.95\times10^{-3}\) to \(1.11\times10^{-4}\). Doubling the bond cap at the latter width gave essentially the same error. In a dephased quantum Ising Gibbs state on 128 sites, a five-site gate at bond 24 gave a doubled two-site discrepancy of \(8.69\times10^{-6}\) at \(t=1\), measured against an eight-site root reference. The classical study also bounded the full exterior contribution at \(t=0.2\), giving a local CP error bound of \(5.39\times10^{-4}\) through 256 sites. The quantum figure is a finite-window diagnostic. [Ising tests and certification](local_cp_large/report.md)

## A common test under the promise

Start from the open 1D cluster state and dephase one spin \(j\) at a time. If \(C\) is the cluster controlled-Z circuit, the physical state throughout is \(C(\bigotimes_i\rho_i)C^\dagger\), with \(\rho_i=(I+q_i X_{\rm Pauli})/2\). An onsite phase channel changes only \(q_j\). There is an explicit local recovery: undo the controlled-Z gates incident on \(j\), reset the exposed spin to its previous \(\rho_j\), and reapply those gates. The global canonical root is known by replacing each factor by \(\sqrt{\rho_i}\), and its MPS bond is at most four throughout. [Cluster derivation](cluster_dephasing_derivation.md)

The first step from the pure cluster state answers a different question for each method. For **TDVP**, the target increment is \(O(\sqrt h)\). An unrestricted change to even one MPS tensor can represent the exact endpoint here. A six-spin two-tensor fit supplied with that endpoint attained vector error \(5.9\times10^{-15}\), establishing representability. The single-excitation clock experiment above tested the actual time integrator. For the **unitary method**, an optimal gate on either adjacent pair had global vector error \(0.21946\) at \(h=0.1\). Allowing the triplet \(j-1,j,j+1\) reduced it to \(2.4\times10^{-15}\). This support test also used the exact endpoint. The gate angle scales as \(\sqrt h\) at onset and as \(h\) at positive time, with the same three-site support. [Tensor fit](cp_tensor_search_notes.md) · [Unitary support and angle fits](cp_branch_comparison.md)

For a direct comparison, run both numerical updates along this same site-by-site trajectory at bond four, reserving the analytic root for scoring. Record the global vector error after the first noise step and after each sweep. Refine the TDVP time grid and the unitary active and fitting widths independently, then compare work at matched final error. The existing tests establish the two bottlenecks separately. This common-trajectory cost comparison remains open. The earlier Kraus–TDVP handoff used a restricted reference compression as its fallback, and GHZ dephasing violated the recovery promise. Neither experiment made this comparison.
