# Testing the Kraus-to-TDVP handoff

November 9, 2026

The buffered Kraus method gave the lowest final global canonical-purification error within the measured runtime budgets on four and six spins. TDVP could improve on a narrow Kraus buffer when supplied with a cheap dense force. On six spins the current MPO solver struggled to compute that force accurately enough to accept a step.

## Experiment

The input was the pure state |+⟩ on every site of an open chain. The Hamiltonian had ZZ coupling 0.7, X field 0.43 and Z field 0.19. Each site j had a loss operator |0⟩⟨1| with rate 0.3 + 0.12 cos²(j), starting at j=0. All runs ended at t=0.1. State bond dimension was capped at 16. Force bond dimension was 16 on four spins and 24 on six.

The hybrid used ten physical steps. The initial step used buffered Kraus evolution. Subsequent trials used the existing two-site MPS tangent projection with a retracted explicit midpoint in u=√t. The two midpoint forces solve the canonical-root Sylvester equation at their respective stage states. Each retraction compresses, Hermitizes and normalizes the amplitude.

At handoff the controller compared the TDVP trial with a buffered Kraus step from the same input. The discrepancy was the maximum trace distance over all adjacent doubled-site marginals. For the Kraus check it compared buffer one with two, then two with three if needed, and selected the wider estimate upon agreement. Each candidate included two reference trace sweeps. The controller checked the first TDVP step and every fourth step thereafter, and retried at the next step after rejection. A failed force or marginal check triggered the accepted Kraus step. The primary marginal tolerance was 10⁻⁴. A separate run used 10⁻⁵. The force solver targeted a relative residual of 10⁻⁶ with an acceptance ceiling of 10⁻⁵.

The score was ‖Q−√ρ(t)‖F. We computed ρ(t) by exponentiating the sparse Lindblad generator, then diagonalized the result. Only scoring used this trajectory reference. Timings include initialization, both midpoint stages, failed trials, buffer comparisons, reference rotations and tensor compression. They exclude reference generation and file I/O. All timed evolutions ran sequentially with one BLAS thread.

## Results

| Spins | Method | Physical steps | Accepted TDVP intervals | Seconds | Global CP error |
|---:|---|---:|---:|---:|---:|
| 4 | MPO hybrid | 10 | 9 of 9 | 3.99 | 1.40×10⁻⁴ |
| 4 | MPO hybrid, tolerance 10⁻⁵ | 10 | 6 of 9 | 4.29 | 5.73×10⁻⁵ |
| 4 | MPO hybrid, four TDVP substeps per interval | 10 | 9 of 9 | 13.19 | 8.53×10⁻⁶ |
| 4 | Kraus, buffer two | 10 | — | 0.080 | 2.70×10⁻⁵ |
| 4 | Kraus, buffer two | 461 | — | 3.48 | 1.80×10⁻⁶ |
| 4 | Kraus, buffer three | 470 | — | 3.69 | 1.18×10⁻⁸ |
| 6 | MPO hybrid | 10 | 0 of 9 | 62.07 | 8.19×10⁻⁵ |
| 6 | Kraus, buffer two | 10 | — | 0.275 | 8.19×10⁻⁵ |
| 6 | Kraus, buffer two | 1857 | — | 48.24 | 7.17×10⁻⁵ |
| 6 | Kraus, buffer three | 1515 | — | 51.28 | 2.91×10⁻⁶ |

For the budget comparison, a 20-step timing pilot determined the baseline step count for each chain and buffer. The target was 94% of the measured primary hybrid runtime. The resulting runs used 78–92% of that budget. The four-spin primary hybrid repeated in 4.11 seconds with the same error. Timing ratios are specific to this implementation and machine.

On four spins the primary hybrid ended within 1.5×10⁻¹⁵ of the positive square root of its **own** density. Its physical trace-distance error was 8.99×10⁻⁵. Refining only the TDVP substeps reduced this to 5.70×10⁻⁶. This places much of the four-spin error in integrating the physical trajectory after handoff. At tolerance 10⁻⁵, Kraus alone using the same buffer checks reached 2.61×10⁻⁵ in 0.202 seconds.

## What limited the six-spin force

All nine compressed MPO trials failed the force safeguard. Their true relative residuals ranged from 0.0128 to 0.147, so the choice of a 10⁻⁵ acceptance ceiling was far from marginal. After each rejection the controller committed its Kraus fallback. Retrying at each following interval accounts for most of the 62-second cost.

At t=0.05, the Hermitian amplitude used for the solve had smallest eigenvalue 1.28×10⁻⁷ and condition number 7.74×10⁶. Solving directly and compressing just the dissipative force to bond 24 gave relative residual 4.37×10⁻⁶. An acceptable force therefore fits at that bond for this state. The complete matrix calculation using the same product preconditioner reached 10⁻⁶ in 122 CG iterations without restarts. A separate compressed MPO retry using 180 uninterrupted iterations gave residuals 4.60 and 0.335 at force bonds 24 and 32. Removing restarts alone was insufficient for the compressed iteration.

## When the handoff helped

For a small-system control, we evaluated the Sylvester force by direct eigendecomposition at each TDVP stage and converted it to an MPS with the same force bond cap. The true trajectory remained available only for scoring. At six spins this control accepted all nine TDVP intervals. Increasing midpoint substeps from one to two to four gave global CP errors 2.22×10⁻⁴, 5.48×10⁻⁵ and 1.40×10⁻⁵.

| Six spins, comparable wall time | Seconds | Global CP error | Distance to positive root of own density |
|---|---:|---:|---:|
| Direct-force hybrid, four TDVP substeps | 1.27 | 1.40×10⁻⁵ | 2.2×10⁻¹⁵ |
| Kraus, buffer two, 40 steps | 1.13 | 7.24×10⁻⁵ | 7.21×10⁻⁵ |
| Kraus, buffer three, 32 steps | 1.15 | 4.61×10⁻⁶ | 1.9×10⁻¹⁵ |

TDVP reduced the branch error that limited the narrow buffer. Increasing the Kraus buffer by one site reduced it as well and gave a smaller overall error. At these sizes the wider buffer covers the entire chain for some updates, and on four spins buffer three always does. The comparison establishes a finite-chain result. The relative cost on a long chain remains open. An efficient and reliable MPO force is the immediate requirement for testing the same handoff there.

## Reproduction

The archive includes the controller, solver, validation controls, raw timings, per-step decisions and SHA256 manifest. Run the commands below from the extracted folder. Numerical outputs are written to handoff_results.

~~~powershell
python handoff_trial.py --mode mpo --n 4
python handoff_trial.py --mode mpo --n 6 --force-chi 24
python handoff_trial.py --mode mpo --n 4 --tol .00001
python handoff_trial.py --mode mpo --n 4 --tdvp-substeps 4
python handoff_trial.py --mode dense --n 6 --force-chi 24 --tdvp-substeps 4
python handoff_trial.py --mode fixed --n 6 --force-chi 24 --buffer 3 --steps 32
python handoff_costs.py
python handoff_force_check.py
python handoff_mpo_retry.py
~~~
