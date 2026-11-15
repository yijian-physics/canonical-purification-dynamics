# Repeated local scaling cohort, 2026-11-15

All measurements in this directory were executed on the same Apple M2 Mac (8 GiB RAM). `hardware_verified.json` records a separately verified hardware query because the original sandbox denied `sysctl`. NumPy uses Apple Accelerate; `blas_verified.json` records the build configuration. Thread-count environment variables, including `VECLIB_MAXIMUM_THREADS=1`, were set before numerical imports. The active native thread count was not independently measured.

The main suite uses three independent subprocess repetitions for the primary small-system configurations and the witness controls through N=16. Larger witness runs and N=8 factor attempts are exploratory single observations. Matched-accuracy selection requires a complete repeat group with finite successful results meeting the target in every repeat; it uses the median total time, with minimum and maximum separately available. Single observations, unsupported sizes, failures and timeouts do not become repeated timing evidence.

## Reproduction

From the repository root, write to fresh output directories:

```sh
PYTHONPYCACHEPREFIX=/private/tmp/tdvp-pycache python3 -m factor_tdvp_accelerated.repeated_scaling --suite all --out <new-main-directory> --timeout 45 --memory-mib 2048 --budget 840
PYTHONPYCACHEPREFIX=/private/tmp/tdvp-pycache python3 -m factor_tdvp_accelerated.repeated_scaling --suite supplemental --out <new-supplemental-directory> --timeout 60 --memory-mib 2048 --budget 420
PYTHONPYCACHEPREFIX=/private/tmp/tdvp-pycache python3 -m factor_tdvp_accelerated.repeated_scaling --suite parity_eight --out <new-parity-eight-directory> --timeout 45 --memory-mib 2048 --budget 140
```

The supplemental suite defines three repeats of E3 with 50 and 150 steps for root/factor methods, D256 and D512 witness retries with a 120-second cap, and an E8 factor-cap-two/work-cap-64 control with a 45-second cap. After that E8 control completed cheaply, `parity_eight` adds three repeats at the same factor/work caps with 50 steps to test matched accuracy. These per-case overrides are recorded in the configurations. No strict compression or support guard is relaxed. All four methods retain the existing physical models and arithmetic.

`measurements_main.json` and `.csv` preserve the main suite exactly as emitted. Final `measurements.json` and `.csv` append the supplemental rows with `supplemental/` or `parity_eight/` prefixed to their directory paths; original main values are unchanged. Each run directory retains its original `measurement.json`, `summary.json` when produced, stdout, endpoint/trial CSVs, checkpoint, and factor-solve diagnostics where applicable. Each additional cohort has its own environment and original tables. Do not load both the combined table and these constituent tables into one analysis.

The deterministic merge and provenance sidecars are reproduced with `python3 -m factor_tdvp_accelerated.merge_local_cohorts factor_tdvp_accelerated/scaling_runs/local_20261115`. `collection_provenance.json` records constituent-table checksums. The final collection has 328 rows: 289 completed, 5 failed, 4 timeouts and 30 unsupported. The unsuccessful E8 50-step refinement is preserved: all three repetitions stopped after seven accepted steps at t=0.14 with `LinAlgError: SVD did not converge`. It supplies no final global-error or successful matched-accuracy timing. Its separate five-step control completed, but missed the 1e-5 target.

## Resource limits and provenance

The main suite initially attempted parent-process RSS monitoring with `ps`. The app sandbox denied `ps`; early rows therefore have `peak_observed_rss_mib=0`, which **does not indicate zero memory usage or an enforced memory cap**. Normal-exit `peak_process_memory_mib` is a separate valid `resource.getrusage` high-water measurement. Time limits were enforced by the parent throughout.

During the run, a child-side `getrusage` watchdog was added. It polls high-water RSS every 250 ms and writes `memory_limit_event.json` before exiting with code 86 if the cap is exceeded. It is a sampled safety guard, not a hard address-space quota; transient overshoot is possible. Subsequent summaries identify this guard explicitly. `guard_validation/validation.json` and the automated memory-guard regression verify that an intentionally low 1 MiB cap triggers the guarded exit. Resource-monitor availability is audited separately in `resource_guard_audit.json`; no absent monitor is relabeled as an enforced cap. Timed-out workers without a completion summary cannot provide a final child high-water observation.

The numerical solver was unchanged during this cohort. The benchmark runner gained the memory guard, source recording, and later defensive accuracy-selector checks. The initial `environment.json` keeps its original source hash; its exact initial runner snapshot was not retained. Later worker summaries record their runner hash. Matching source snapshots, where actually retained, are in `instrumentation_sources/`. The supplemental environment records its complete source version. These provenance differences concern instrumentation and evidence handling, not a change to the TDVP equation or tolerances.

## Timing interpretation

`total_seconds` includes worker model initialization, evolution, independent references, correction/certification, scoring and native checkpoint I/O; it excludes interpreter startup. `wall_seconds` additionally includes subprocess startup and orchestration. Both are measured rather than inferred. The nested `timing_exclusive_seconds` categories partition each worker's total without double counting. Their `other_worker_seconds` remainder includes uninstrumented setup, exact scoring, harness imports, some correction application and I/O. In particular, conventional-root dense Liouvillian construction can dominate its setup remainder; `legacy_summary.json` also reports setup/evolution/scoring subdivisions.

`timing_inclusive_seconds` gives contextual enclosing costs and must not be summed. Inclusive witness fitting contains its nested retractions, whereas exclusive witness fitting excludes separately attributed retractions. The witness `correction_direction` timer measures the physical-density-based update construction; it is distinct from fitting a witness to a predicted root. Root predictor force, retraction, independent density/inverse-metric propagation, metric initialization and certification are separately charged. None of the witness's auxiliary computations are removed from total time.

For D, exact global-root errors are evaluated through N=8; larger native witness results use a global upper bound conditional on the independent reference and inverse metric, with ordinary floating-point evaluation. For E, the analytic reference is itself an MPO, so global-root error is evaluated by full-chain contraction at every tested size. This is not a local-marginal proxy.

The root/factor finite driver still has dense components. Root N>=8 is unsupported here because its dense Liouvillian alone exceeds the memory budget. Factor N=8 is actually attempted; factor N>=12 is excluded by the finite-driver dense-audit policy, separately from the matrix-free force implementation. Such policy exclusions are not measured runtimes or proofs of asymptotic complexity. The N=4 D factor trajectory uses cap16, while N=3,5,6 use cap4; equal nominal root/factor caps should not be interpreted as equal expressivity.
