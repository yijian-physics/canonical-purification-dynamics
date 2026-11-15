# Eight-spin parity failure review

This read-only review concerns the three runs under
[`local_20261115/parity_eight/`](../scaling_runs/local_20261115/parity_eight/),
`E008_matrix_free_s50_0b5a078e_r{1,2,3}`. They use 50 physical-time steps to
`t=1`, factor bond 2, and work bond 64. Each reports
`LinAlgError: SVD did not converge`, with seven validated steps and last
validated time `0.14`. All three runs remain failed final-endpoint observations.

## Failure-stage localization

The retained `stdout.txt` files are empty. The legacy benchmark catches the
exception and stores its type and message without a traceback. Therefore the
location below is inferred from recorded counters and source control flow,
not from a captured traceback.

Each run's `summary.json` records:

| Counter | Calls |
|---|---:|
| Factor tangent force | 16 |
| Factor retraction | 16 |
| Dense support audit | 39 |
| Independent reference scoring | 9 |

Seven completed steps account for 14 returned forces, 14 retractions, 35
support audits, and eight reference evaluations including initialization.
The additional two returned forces, two retractions, four support audits,
and one reference evaluation place the exception after the eighth midpoint
step, its endpoint support guard, and evaluation of the reference at `t=0.16`.
It precedes the final audit used to construct that endpoint's recorded row.

In [`factor_benchmarks.py`](../../factor_tdvp_comparison/factor_benchmarks.py),
`run_case()` calls `metrics(amplitude, reference(right))` at line 174.
`metrics()` performs `np.linalg.svd(amplitude, full_matrices=False)` at line
105. This is the SVD in that interval of the control flow. The parity reference
uses `parity_power()` and dense export, neither of which performs an SVD.
The inferred failure is consequently the **dense endpoint-scoring SVD**, not
a failed MPO-LSQR iteration or factor retraction. The endpoint's force solves
and final support guard had returned before scoring began.

## Accepted-step evidence only

The first 14 entries of each `solves.json` correspond to the seven validated
steps. Their numerical values agree across the three repetitions:

| Quantity | Value |
|---|---:|
| Last validated time | 0.14 |
| Last validated global canonical-root error | 4.318873208028957e-6 |
| Last validated density Frobenius error | 4.325269799894033e-7 |
| Maximum accepted-force relative residual | 4.207715246544298e-14 |
| Maximum accepted-force relative stationarity residual | 5.3723246127723327e-14 |
| Maximum Krylov iterations per accepted force | 3 |
| Maximum accepted-force LSQR condition estimate | 6.5713648673440614 |

The last two returned forces belong to the attempted eighth step. They must
not be presented as evidence of a validated eighth endpoint. Residuals are
those of the rounded MPO operator, and the LSQR condition estimate is not the
full Jacobian condition number.

## Limits of the diagnosis

No offending root matrix or traceback was saved. The records therefore do
not establish why the dense SVD failed, whether its input was nonfinite, or
whether a particular LAPACK/backend behavior was responsible. They do not
establish force-solver divergence, integration instability, inadequate factor
bond, or an intrinsic limitation of factor TDVP. No rerun was used for this
review. Successful diagnostics through `t=0.14` do not establish accuracy or
successful runtime at the requested final endpoint `t=1`.

The legacy `evolution_seconds` field is also unsuitable for decomposing this
failure's cost: `run_case()` increments evolution time at line 171, then its
exception handler increments elapsed step time again at line 189 when scoring
raises. Use the new exclusive timing ledger and worker `total_seconds` for
failed-run accounting, retaining the original legacy values as evidence.
