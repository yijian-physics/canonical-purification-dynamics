# Canonical-purification evolution under onsite dephasing

An exact benchmark on the open 1D cluster state. Derived and numerically checked on November 2, 2026.

## Model and conventions

Let

\[
|C_N\rangle=U|+\rangle^{\otimes N},\qquad
U=\prod_{j=1}^{N-1}\mathrm{CZ}_{j,j+1}.
\]

The physical dynamics is

\[
\dot\rho=\frac\gamma2\sum_j(Z_j\rho Z_j-\rho),\qquad
\rho(0)=|C_N\rangle\langle C_N|.
\]

A single-qubit off-diagonal element decays as \(q(t)=e^{-\gamma t}\). A time interval \(\Delta t\) corresponds to the phase-flip probability \(p=(1-e^{-\gamma\Delta t})/2\).

Physical and reference degrees of freedom are paired into each MPS site, in local basis \((00,01,10,11)\). Vectorization means

\[
|M\rangle\!\rangle=\sum_{x,y}M_{xy}|x\rangle_P|y\rangle_{P'}.
\]

The normalized canonical purification is \(|\Psi(t)\rangle=|\sqrt{\rho(t)}\rangle\!\rangle\).

## Exact state and its positive square root

Because every physical \(Z_j\) commutes with \(U\),

\[
\rho(t)=U\tau(q)^{\otimes N}U^\dagger,\qquad
\tau(q)=\frac{I+qX}{2}.
\]

Consequently,

\[
\sqrt{\rho(t)}=U\sqrt{\tau(q)}^{\otimes N}U^\dagger,
\qquad
|\Psi(t)\rangle=(U\otimes\overline U)
|\sqrt{\tau(q)}\rangle\!\rangle^{\otimes N}.
\]

The connection between local dephasing and thermal cluster states is established background; see [Aguilar et al., PRL 112, 160501 (2014)](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.112.160501). The doubled-chain formulas below follow directly from the displayed factorization. No claim of novelty is made for this solvable example.

Define the normalized Bell states \(|\Phi^+\rangle=(|00\rangle+|11\rangle)/\sqrt2\), \(|\Psi^+\rangle=(|01\rangle+|10\rangle)/\sqrt2\), and

\[
\theta(t)=\tfrac12\arcsin q(t)\in[0,\pi/4].
\]

Directly taking the square root of \(\tau\) gives

\[
|\sqrt\tau\rangle\!\rangle
=\cos\theta\,|\Phi^+\rangle+\sin\theta\,|\Psi^+\rangle
=e^{-i\theta Z_PY_{P'}}|\Phi^+\rangle.
\]

Here \(Z_PY_{P'}|\Phi^+\rangle=i|\Psi^+\rangle\) and \(Z_PY_{P'}|\Psi^+\rangle=-i|\Phi^+\rangle\), fixing the sign convention.

## An exact local generator on the doubled chain

Write \(V=U\otimes\overline U\). Then

\[
Q_j=V(Z_jY_{j'})V^\dagger
=Z_j Z_{(j-1)'}Y_{j'}Z_{(j+1)'},
\]

omitting nonexistent neighbors at an open boundary. These are Hermitian Pauli strings with \(Q_j^2=I\), supported on at most three adjacent doubled sites. They commute with one another, since the undressed \(Z_jY_{j'}\) commute.

Using \(V|\Phi^+\rangle^{\otimes N}=|\Phi^+\rangle^{\otimes N}\),

\[
|\Psi(t)\rangle=e^{-i\theta(t)\sum_jQ_j}|\Phi^+\rangle^{\otimes N}.
\]

Thus an exact effective Hamiltonian along this particular state trajectory is

\[
H_{\mathrm{CP}}(t)=\dot\theta(t)\sum_jQ_j,
\qquad
\dot\theta(t)=-\frac{\gamma e^{-\gamma t}}{2\sqrt{1-e^{-2\gamma t}}}.
\]

This is a state-trajectory lift, not a universal Hamiltonian implementing dephasing on arbitrary purifications.

The coefficient diverges as \(-\sqrt{\gamma/(8t)}\) near the rank-deficient initial endpoint. Its integral is finite. For any two times,

\[
|\Psi(t_2)\rangle=\prod_jG_j(t_2,t_1)|\Psi(t_1)\rangle,
\]

\[
G_j=\exp[-i(\theta(t_2)-\theta(t_1))Q_j]
=\cos(\Delta\theta)I-i\sin(\Delta\theta)Q_j.
\]

These gates are well-defined even for \(t_1=0\), and the product has no Trotter error. In the parameter \(s=\pi/4-\theta(t)\), the generator is the bounded, constant operator \(-\sum_jQ_j\).

## MPS representability throughout the evolution

Before applying \(V\), the state is a product over doubled sites. Across any spatial cut, \(V\) has only one physical CZ and one reference CZ crossing the cut. Each has operator Schmidt rank two, so

\[
\boxed{\chi_{\mathrm{CP}}(t)\le4\quad\text{for all }N,t.}
\]

The initial cluster-state purification has rank four across an interior cut; at complete dephasing the CP becomes a product of onsite Bell pairs, with rank one.

The same bound holds after each individual update gate: such a gate changes only the angle \(\theta_j\) in

\[
V\bigotimes_j e^{-i\theta_jZ_jY_{j'}}|\Phi^+\rangle.
\]

For angles in \([0,\pi/4]\), this is still the exact canonical purification of an inhomogeneously dephased cluster state. Therefore capping the MPS bond dimension at four after every gate introduces no exact-arithmetic truncation error.

Along the smooth part of this curve, the exact derivative lies in the bond-four MPS tangent space. A correctly formulated TDVP therefore has no tangent-projection error on this trajectory. The supplied implementation uses the exact commuting gates; it does not implement a generic TDVP integrator.

## Exact finite Markov length

The physical density matrix is

\[
\rho(t)=2^{-N}\prod_j(I+qK_j),\qquad
K_j=Z_{j-1}X_jZ_{j+1},
\]

with open-boundary stabilizers \(K_1=X_1Z_2\), \(K_N=Z_{N-1}X_N\).

Let nonempty contiguous \(A,B,C\) cover the open chain, with sizes \(a,b,c\), and let \(h=h_2((1+q)/2)\), using natural logarithms. Independent stabilizer counting gives, for \(b\ge2\),

\[
S(ABC)=Nh,\quad S(AB)=\log2+(a+b-1)h,
\]
\[
S(BC)=\log2+(b+c-1)h,\quad S(B)=2\log2+(b-2)h.
\]

Hence

\[
\boxed{I(A:C\mid B)=0\quad\text{for }b\ge2.}
\]

For a single-site buffer, \(I(A:C\mid B)=\log2-h\). These formulas extend continuously to the pure initial state and the maximally mixed endpoint. The geometry is specifically a contiguous full-chain tripartition of an open chain.

## Implemented benchmark and validation

`cluster_dephasing.py` uses NumPy/SciPy MPS tensors, QR center shifts, and local SVD compression. It prepares the initial CP from \(|++\rangle^{\otimes N}\) using physical and reference CZ gates. Evolution applies the local \(G_j\) gates with maximum bond dimension four. Dense matrices are used only for the six-site reference checks.

The independent density reference applies dephasing directly in the computational basis:

\[
\rho_{xy}(t)=\rho_{xy}(0)e^{-\gamma t\,d_H(x,y)}.
\]

The positive square root is computed by dense diagonalization and compared with the evolved MPS, without optimizing a purification gauge.

For \(N=6\), \(\gamma=1\), and 31 times from zero to three:

| Diagnostic | Maximum observed value |
|---|---:|
| CP vector error against dense positive square root | \(1.96\times10^{-13}\) |
| CP vector error against analytic square root | \(1.64\times10^{-14}\) |
| Physical density trace distance | \(1.35\times10^{-14}\) |
| Absolute CMI for buffers two and three | \(2.67\times10^{-15}\) |
| Discarded squared singular-value weight per split | \(3.47\times10^{-32}\) |
| Maximum spatial MPS bond dimension | 4 |

Additional checks:

- A single update from \(t=0\) to \(t=3\) agrees with 30 updates to within \(1.72\times10^{-14}\) in vector norm.
- Finite differences verify the effective-Hamiltonian equation at three positive times, with residual below \(9\times10^{-10}\).
- A 64-site MPS-only update from zero to one retains bond dimension four. Its norm differs from one by less than \(5\times10^{-16}\), and a bulk stabilizer expectation agrees with \(\langle K_j\rangle=e^{-1}\) to within \(2\times10^{-16}\). This large-chain check does not compare a full density matrix.

Run `python cluster_dephasing.py`, or add `--outdir PATH` to select the output directory. The script writes a plot, CSV data, and a JSON validation summary.

## Interpretation for the proposed algorithm

All three concerns have explicit resolutions in this example: the singular time coefficient integrates to finite gates; constructing each gate is elementary; and the required spatial bond dimension is bounded by four at every step. At fixed bond dimension and gate width, a sweep costs linearly in chain length.

It provides an exact reference trajectory against which to test a more general method that learns or constructs local updates from the current MPS. Its special commuting structure does not establish generic efficiency or novelty of the broader algorithm.

If a normalized approximate purification is within \(\varepsilon\) of the target canonical purification in vector norm, its physical density matrix is within \(\varepsilon\) in trace distance. This benchmark monitors the CP vector itself, so it also directly checks the closeness relevant to canonical-purification tracking.
