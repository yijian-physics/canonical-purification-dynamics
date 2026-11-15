# Increasing-size canonical-purification TDVP comparison

Executed benchmarks, **not a proof of an asymptotic ordering**. All use correlated full-rank mixed-chain family D, final time 0.1 and uniform grids. Each case runs in a separate one-BLAS-thread process. The wall times include subprocess startup and scoring. Single observations are not confidence intervals.

Root+witness and witness-only corrections have an independently propagated density and inverse-metric bound unavailable to plain root/factor trajectories: this is an unequal-information comparison. The witness global CP quantity is a measured error through N=8 and otherwise an upper bound (conditional on correct reference/metric). Never replace the latter by a measured error.

Direct factor uses the accelerated MPO-LSQR force but its driver retains dense support audits and finite-size scoring. N=4 factor cap is 16; N=5,6 cap is 4 and hence restricted, not a full-capacity comparison. The factor-implied raw root bond can be as large as chi_H squared.

## Trajectory measurements

| N | Method | Steps | Status | Measured CP error | CP upper bound | Wall s |
|--:|:--|--:|:--|--:|--:|--:|
| 3 | root | 5 | completed | 9.28e-07 | — | 0.615 |
| 3 | matrix_free | 5 | completed | 9.32e-07 | — | 0.966 |
| 3 | root | 10 | completed | 2.28e-07 | — | 0.315 |
| 3 | matrix_free | 10 | completed | 2.3e-07 | — | 1.77 |
| 3 | root | 20 | completed | 5.67e-08 | — | 0.365 |
| 3 | matrix_free | 20 | completed | 5.7e-08 | — | 3.22 |
| 4 | root | 5 | completed | 1.13e-06 | — | 0.315 |
| 4 | matrix_free | 5 | completed | 1.14e-06 | — | 2.97 |
| 4 | root | 10 | completed | 2.79e-07 | — | 0.365 |
| 4 | matrix_free | 10 | completed | 2.8e-07 | — | 5.58 |
| 5 | root | 5 | completed | 1.31e-06 | — | 0.465 |
| 5 | matrix_free | 5 | completed | 1.32e-06 | — | 15.5 |
| 6 | root | 5 | completed | 1.47e-06 | — | 2.27 |
| 6 | matrix_free | 5 | failed | — | — | 0.615 |
| 3 | witness | 5 | completed | 1.15e-06 | 2.34e-06 | 0.315 |
| 3 | witness | 5 | completed | 3.09e-09 | 6.29e-09 | 0.365 |
| 3 | correction_only | 5 | completed | 1.2e-07 | 2.44e-07 | 0.264 |
| 3 | correction_only | 5 | completed | 2.06e-09 | 4.24e-09 | 0.265 |
| 4 | witness | 5 | completed | 1.46e-06 | 3.54e-06 | 0.415 |
| 4 | witness | 5 | completed | 3.78e-08 | 8.22e-08 | 0.615 |
| 4 | correction_only | 5 | completed | 3.17e-07 | 6.63e-07 | 0.315 |
| 4 | correction_only | 5 | completed | 3.78e-08 | 8.22e-08 | 0.465 |
| 5 | witness | 5 | completed | 1.74e-06 | 4.37e-06 | 0.615 |
| 6 | witness | 5 | completed | 2.02e-06 | 5.14e-06 | 0.665 |
| 8 | witness | 5 | completed | 2.56e-06 | 6.69e-06 | 0.966 |
| 8 | correction_only | 5 | completed | 6.98e-07 | 1.46e-06 | 0.566 |
| 16 | witness | 5 | completed | — | 1.88e-06 | 1.82 |
| 32 | witness | 5 | completed | — | 2.84e-06 | 3.77 |
| 64 | witness | 5 | completed | — | 4.22e-06 | 8.03 |
| 128 | witness | 5 | completed | — | 7.33e-06 | 16.9 |
| 256 | witness | 5 | completed | — | 8.94e-06 | 38.8 |

## Cheapest observed qualifying run at target accuracy

A completed row qualifies if its measured global error or separately reported global upper bound is below target. A dash means not attained among the executed rows, not that the method can never succeed.

### Target 1e-05

| N | Plain root | Direct factor | Root + witness | Correction only |
|--:|--:|--:|--:|--:|
| 3 | 0.315s (measured) | 0.966s (measured) | 0.315s (measured) | 0.264s (measured) |
| 4 | 0.315s (measured) | 2.97s (measured) | 0.415s (measured) | 0.315s (measured) |
| 5 | 0.465s (measured) | 15.5s (measured) | 0.615s (measured) | — |
| 6 | 2.27s (measured) | — | 0.665s (measured) | — |
| 8 | — | — | 0.966s (measured) | 0.566s (measured) |
| 16 | — | — | 1.82s (bound) | — |
| 32 | — | — | 3.77s (bound) | — |
| 64 | — | — | 8.03s (bound) | — |
| 128 | — | — | 16.9s (bound) | — |
| 256 | — | — | 38.8s (bound) | — |

### Target 1e-07

| N | Plain root | Direct factor | Root + witness | Correction only |
|--:|--:|--:|--:|--:|
| 3 | 0.365s (measured) | 3.22s (measured) | 0.365s (measured) | 0.265s (measured) |
| 4 | — | — | 0.615s (measured) | 0.465s (measured) |
| 5 | — | — | — | — |
| 6 | — | — | — | — |
| 8 | — | — | — | — |
| 16 | — | — | — | — |
| 32 | — | — | — | — |
| 64 | — | — | — | — |
| 128 | — | — | — | — |
| 256 | — | — | — | — |

## Scientific interpretation and limitations

An advantage of root+witness at matched global accuracy can arise from fewer time steps thanks to cheap positive-factor endpoint correction. It does not show that each corrected TDVP step is faster. The factor method spends time on real-linear Jacobian/adjoint Krylov solves and can be ill-conditioned at small physical populations. Unsupported, timed-out and inaccurate factor trajectories cannot be converted into invented runtime or an asymptotic lower bound.

The strongest defensible conclusion is implementation- and benchmark-specific; a universal or asymptotic speed ordering would require matched-information, matched-accuracy families and genuine polynomial-cost support handling for the factor method.

Reproduction: python -m factor_tdvp_accelerated.size_scaling --suite full --out <new-directory>
