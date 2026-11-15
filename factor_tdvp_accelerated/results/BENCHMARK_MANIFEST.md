# Benchmark datasets

All datasets below were actually executed on the current host. Each worker uses one BLAS thread. Wall time can vary with shared-host load; no uncertainty intervals or universal speedup claims are inferred from single measurements. Environment files record Python/library versions. Historical directories elsewhere in the repository were not overwritten.

Primary datasets:

- `forces_corrected_timing/`: final explicit-versus-LSQR D initial-force profiles at N=3,4,6, factor cap 4, exact finite-chain work caps 4,16,64. **Use `force_total_seconds`** for complete force-call timing; the LSQR `solve_seconds` field excludes its internal setup. Profiled timings include cProfile overhead for both methods. Cumulative profile categories overlap. Velocity comparisons are independent dense checks after the timed solves.
- `trajectories_final/`: final A/B/C/E/F small-system controls plus D N=3,4 refinement. The N=4 D trajectory uses factor cap 16 (full tangent capacity); it must not be described as the same restricted cap-four resource test as the N=4 initial-force profile. Benchmarks use the historical dense support detector and finite positive channel bridges. Those components and dense scoring remain exponential; the matrix-free force alone removes dense Jacobian output storage.
- `matched_witness/`: D3 native root+witness and correction-only controls using **the same uniform endpoint grids** as the factor/root refinement: 5,10,20 intervals to t=0.1, each at certificate tolerances 1e-3,1e-5,1e-7. Predictor, correction and independent reference/certification costs remain part of their end-to-end totals. Timing subcategories may overlap. Their access to independent density and inverse-metric information differs from uncorrected evolution.
- `stress_ablation/`: F_loss N=3, ten intervals to t=.03, factor cap/work cap 4. Explicit and matrix-free tolerances 1e-8/1e-10/1e-12 are run on identical settings. Improved least-squares stationarity does not ensure small canonical-root error in this ill-conditioned rank-opening experiment.
- `witness/`: supplementary native-grid D/E witness/correction-only controls. These use the historical nonuniform grids. They are intentionally excluded from the primary D matched-grid figures/frontier.

`../figures/combined_measurements.json` combines final force, final trajectory and matched-grid D witness rows. Its frontier selects the cheapest observed successful row in the measured grid for a fixed family, size and endpoint, without interpolation. E witness comparisons are supplementary because only D has the new matched-grid witness sweep. Worker `total_seconds` excludes interpreter/import startup but includes model setup, evolution, audits and scoring; `subprocess_seconds` additionally includes interpreter startup and orchestration. For the witness controls, native time includes initialization, prediction, correction, certificate evaluation, internal exact scoring and native checkpoint I/O.

Preliminary evidence retained:

- `forces/` used the earlier per-iteration checking implementation and had an ambiguous `solve_seconds` aggregation (solver internal timing overwrote the wrapper's timing). It is superseded by `forces_corrected_timing/`; do not use its stored speedup in final conclusions.
- `trajectories/` is the preliminary per-iteration checking run. Its recursive historical diagnostic aggregation included intermediate Krylov residuals and is superseded by `trajectories_final/`.
- `explicit_n6_profile/` is an independently reproduced explicit D6 force profile, useful for the original bottleneck audit.

For the final trajectory/stress directories, `postprocessing.txt` records that final-stage Jacobian residuals and LSQR condition estimates were recomputed from saved `solves.json`, excluding intermediate Krylov iterates. No measured times or numerical state results changed. The production benchmark wrapper now prevents that recursive diagnostic contamination directly. LSQR's bidiagonal condition estimate is not the retained full Jacobian singular-value condition number reported by the explicit solver.
