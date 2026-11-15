# TDVP implementation handoff

Updated 2026-11-15. This file is the starting point for a fresh Codex chat.

## Current checkpoint: factor versus witness scaling

- Working branch: `research/factor-vs-witness-scaling`, tracking the same branch
  on `xiaoyanghuangphys-create/canonical-purification-dynamics`.
- [PR #3](https://github.com/xiaoyanghuangphys-create/canonical-purification-dynamics/pull/3)
  targets `research/factor-tdvp-accelerated`. Do not merge automatically or
  overwrite historical measurements.
- Start with [the scaling report](factor_tdvp_accelerated/scaling_report.md)
  and [reproduction guide](factor_tdvp_accelerated/README.md). The original
  force acceleration and its PDF remain a separate historical cohort below.

### Recovery and reliable publication

All 165 original files from run `43745844867`, artifact `14174167664`, are
preserved byte-for-byte in
`factor_tdvp_accelerated/scaling_runs/actions_43745844867/`. All hashes and all
31 cases were audited. Thirty cases completed; D6 factor with work bond 32
failed compression before the first accepted step. The GitHub job's final
failure was an ordinary push rejected after another commit advanced the branch,
not a failure to run or upload the numerical sweep.

The publication helper now uses immutable per-run output paths, serialized
workflow runs, artifact upload before publication, and fetch/rebase/retry with
ordinary pushes. Four real-Git integration tests cover concurrent branch
advances, collisions and preservation of failed-publication results. The fixed
[workflow run 43747960010](https://github.com/xiaoyanghuangphys-create/canonical-purification-dynamics/actions/runs/43747960010)
completed and published commit `02dd78b` successfully. Its separate 31-case
dataset is under `factor_tdvp_accelerated/size_scaling_results/actions-43747960010-attempt-1/`.
Do not pool its timing observations with the local cohort.

### New benchmark implementation and evidence

- `factor_tdvp_accelerated/repeated_scaling.py` instruments the existing four
  methods without changing their numerical objectives. It records nested
  inclusive/exclusive timing, actual global CP errors or conditional bounds,
  iterations, bonds, memory, partial outputs and explicit failure statuses.
- `factor_tdvp_accelerated/analyze_scaling.py` audits the recovered artifact and
  local data, then generates tables and plots in `scaling_figures/`. A repeated
  frontier requires every requested repetition to complete and meet its target.
- `factor_tdvp_accelerated/scaling_runs/local_20261115/` contains the repeated
  D/E and interacting-loss controls. Apple M2, 8 GiB RAM, NumPy 2.0.2 with
  Accelerate, SciPy 1.13.1; one BLAS thread requested. Source hashes and late
  hardware/backend verification are retained. Initial parent RSS polling was
  unavailable; those early cases have measured child peak RSS but no enforced
  RSS cap. Later workers use a separately tested child watchdog. Preserve that
  distinction; a zero parent RSS field means unavailable monitoring.

At D3/D4 and measured global CP error at most 1e-7, repeated median factor
times are 2.120/7.439 seconds, versus 0.09345/0.25399 for root plus witness:
22.69/29.29 times slower. All three selected repeats pass. Factor uses 20
steps and witness five endpoints. Auxiliary density, inverse metric, fitting,
correction and certification are included. The factor tangent solve takes
about 95%/89% of total. D6 factor succeeds with work bond 64, whereas the
earlier work-bond-32 configuration fails; a bounded D8 attempt also fails
compression. These failures are not successful runtimes.

The E3 parity refinement reaches error at most 1e-6 in all three repeats:
factor 6.713 seconds at 150 steps, witness 0.1469 seconds at five endpoints
(45.71 times slower for factor). E8 factor with work bond 64 completes a coarse
five-step trajectory at error 1.0159e-3; its 50-step refinement fails SVD in all
three repeats after seven accepted steps. Neither is a successful matched-error
E8 factor result. The raw partial trajectories are retained.
Call counts and control flow locate the failing SVD in endpoint scoring after
the eighth step's force/retraction calls, but no traceback or offending matrix
was saved. The underlying cause remains unresolved; this does not establish
force-solver divergence. See the retained validation review for details.

Large E witness/correction-only runs reach N=512 with measured global errors
3.0895e-6/3.9636e-6. These are exploratory single observations. D256 witness
succeeds with a 120-second budget (44.74 worker seconds, bound 8.944e-6), after
the original 45-second process limit timed out. D512 witness still times out
at 120 seconds. Never substitute a partial endpoint or an older cohort's
success for these incomplete runs.

Validation: 70 numerical/accounting/frontier regressions and four publication
integration tests pass; logs are retained in
`factor_tdvp_accelerated/scaling_validation/`.
The full study and raw results were pushed in commit `164fb0d`. The published
GitHub preview was inspected: all three display-equation blocks render as
typeset mathematics. PR #3 remains open and unmerged.

The observed disadvantage is implementation-specific and at matched global
error. It does not establish universal or asymptotic inferiority. Conventional
root is faster than root plus witness in some cases, and correction-only can
be fastest when the independently propagated physical reference is cheap.
Loose-tolerance D3–8 witness trajectories can pass without any endpoint
correction. Witness fitting alone does not repair physical-density error.
At large D use conditional global bounds; parity E has a global analytic MPO
reference at every size. Different nominal root/factor bonds are not equal
representation budgets. Dense rank/support handling and scoring remain limits
of the factor trajectory adapter, separate from its matrix-free force.

### Next implementation priorities

1. Reduce tangent-force cost with a restricted-tangent preconditioner and
   reusable contraction environments; evaluate success at global CP error,
   not only LSQR stationarity, especially for small populations.
2. Control integration/retraction and compression error with adaptive steps
   and bonds. Project reused velocities into the new tangent frame.
3. Replace dense production support/rank handling with a validated tensor
   procedure; retain dense routines as small-system oracles.
4. Compare methods under the same reference-information budget. A factor
   endpoint-corrected variant is a new algorithm, not a free improvement of
   the existing factor-TDVP objective.

## Prior force-acceleration checkpoint (2026-11-14)

The following sections record the predecessor work and branch state at that
checkpoint; the current working branch and latest evidence are given above.

## Repository and task

- Fork: `xiaoyanghuangphys-create/canonical-purification-dynamics`.
- Base: `research/factor-tdvp-benchmarks`, commit `facc0b7c85f2fbf816607f6365d87233a05e4054`.
- Working branch: `research/factor-tdvp-accelerated` (already existed at the base; continued without resetting it).
- The benchmark branch is not an ancestor of `origin/main` as checked at clone time.
- Do not modify the upstream `yijian-physics` repository or merge a pull request automatically.

## History recovered

All available text in Implement TDVP Method (5 turns), Summarize TDVP Results
(5 turns), and two Continue TDVP Implementation chats (1 turn each) was read.
All four reads ended with no further pages or message truncation. The two
continuation chats contained only initiating user requests. Hidden reasoning,
complete original tool logs and old sandbox files are not supplied by that API.
The current uploaded acceleration prompt is byte-identical to the historical
unanswered acceleration attachment (SHA256
`1b387087176fbaf2a1ae4cd8794602802c5ca6a897aec065ba1cb18728bb66b8`).
Raw chat archives are not published in this repository.

The completed predecessor work was the finite-size factor solver, benchmark
comparison, and math/PDF rendering repair at the base commit. There was no
accelerated implementation to recover. Prior claims are historical evidence;
new measurements live in `factor_tdvp_accelerated/results/`.

## Current implementation

- `factor_tdvp_accelerated/matrix_free_solver.py`: real-coordinate one-site
  tangent maps, MPO Jacobian/adjoint, MPO-output LSQR with residual checks,
  compression/nonconvergence guards, optional damping, diagonal coordinate
  scaling, and projected warm-start support. No dense output-space vectors or
  explicit Jacobian columns in this production force.
- `factor_tdvp/tangent_solver.py`: optional `velocity_solver` / `solver_options`
  injection into midpoint integration. The explicit backend stays the default.
- `factor_tdvp/factor_tdvp.py`: forwards injection while retaining the dense
  first/repeated/delayed support audits, positive bridge, and rank-loss guards.
- `factor_tdvp_accelerated/test_matrix_free_solver.py`: independent regression
  tests; `run_benchmarks.py`: isolated-worker baseline comparisons;
  `run_ablations.py`: explicitly dense small-system conditioning/metric oracles;
  `product_scaling.py`: analytically checked MPO-only product-path force sweep.
- See the accelerated README for commands and report for measured conclusions.

## Interpretation and remaining research

This is a force acceleration, not a generic scalable end-to-end factor method.
The preserved rank audits and general bridge remain dense. LS stationarity can
miss physically important small-singular-value components; interacting-loss
accuracy must be checked separately. A product-state force check through 64
sites is not a correlated 64-site trajectory. No local error replaces a global
canonical-root error in the results.

Next priorities: physical-error-aware stopping and adaptive stepping; a useful
local preconditioner in the restricted tangent; scalable rank/support handling;
compressed-Krylov error control; two-site/adaptive bond growth; and independent
reference/certification cost at matched global error. ALS/local sweeps and
alternative root-metric evolution are not production implementations here.
Never overwrite old measurements; choose fresh output directories.

## Verified checkpoint

- Combined predecessor + accelerated tests: **48 passed**; see
  `factor_tdvp_accelerated/results/tests_full.log`.
- Final D6 profiled force: explicit 14.827 s, MPO-LSQR 0.547 s (27.1x),
  relative velocity difference 3.83e-9.
- D3 minimum observed cost at global CP error <=1e-7: matrix-free 2.092 s,
  conventional root 0.131 s, witness 0.097 s, correction-only 0.047 s.
  Witness comparisons use the same available uniform endpoint grids but extra
  independent density/metric correction information.
- F_loss3 remains an accuracy failure for the default matrix-free tolerance:
  CP error 0.00427; tolerance 1e-12 improves it to 1.58e-4, versus explicit
  5.94e-5. Do not hide this by reporting stationarity or density error alone.
- MPO-only product-force checks completed through 64 sites with global relative
  velocity error below 5e-14. These are not integrated correlated trajectories.
- Authoritative dataset selection is in
  `factor_tdvp_accelerated/results/BENCHMARK_MANIFEST.md`. Preliminary data are
  retained, including the documented correction to residual aggregation.

The typeset report is `factor_tdvp_accelerated/report.pdf`; its pages were rendered
and visually checked. Build source is retained alongside the Markdown.

## Publication

Implementation and measured results were pushed in commit
`001ccc1b426827c407c8fd16dcca72a62477e51c`.
[PR #2](https://github.com/xiaoyanghuangphys-create/canonical-purification-dynamics/pull/2)
is open against `research/factor-tdvp-benchmarks`; it has not been merged.
The local branch tracks `origin/research/factor-tdvp-accelerated`.
