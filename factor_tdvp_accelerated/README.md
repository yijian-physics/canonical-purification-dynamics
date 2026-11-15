# Matrix-free factor-TDVP

Read [the repeated scaling study](scaling_report.md), [original force report](report.md)
and [repository handoff](../HANDOFF.md).
The solver accelerates the original density-residual factor objective. It
uses real tangent coordinates and MPO-output LSQR, retaining factor positivity.
The existing finite-size rank-aware driver accepts it as an injected backend.

## Reproduce

From the repository root, using Python 3.9+ with NumPy, SciPy and Matplotlib:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m unittest factor_tdvp.test_factor_tdvp factor_tdvp.test_rank_bridge factor_tdvp_accelerated.test_matrix_free_solver -v
python3 -m factor_tdvp_accelerated.run_benchmarks --suite all --out run_outputs/accelerated-new --timeout 120
python3 -m factor_tdvp_accelerated.run_ablations --out run_outputs/accelerated-ablations
python3 -m factor_tdvp_accelerated.product_scaling --n 64 --out run_outputs/product64.json
```

Create `run_outputs/` before the last command. The benchmark suite requires a
fresh output directory and writes a result after each isolated worker; failures
and timeouts remain in its tables. `--suite force`, `trajectories`,
`matched_witness`, or `witness` selects a subset. `all` includes the same-grid
D3 witness controls; `witness` separately reproduces historical native grids.
Measured environments are stored alongside results; this run used the available
NumPy 2.0.2 / SciPy 1.13.1 / Matplotlib 3.9.4 installation. Existing requirements
files describe older experiments and have not been silently rewritten.

```python
from factor_tdvp import factor_tdvp as fd
from factor_tdvp_accelerated.matrix_free_solver import factor_velocity
state, events = fd.evolve(
    initial_factor, hamiltonian, jumps, times, chi=4, work_bond=16,
    velocity_solver=factor_velocity,
    solver_options={"tolerance": 1e-10, "maxiter": 400},
)
```

The Hamiltonian/jumps above use the existing dense finite-size driver API.
For an MPO-only force use `factor_velocity(H, hamiltonian_mpo, jump_mpos,
work_bond=...)`. Large chains must declare a work bond. The force uses no dense
exports, but fixed bonds do not guarantee convergence or sufficient accuracy.
The compatibility `rcond` argument is not an LSQR singular-value cutoff.

Damping changes the objective. Coordinate scaling can change the selected
null-space representative. A warm start must be projected into the current
tangent frame using `TangentCoordinates(H).encode(previous_velocity)`; previous
coordinate vectors cannot be reused across changed canonical frames blindly.
Strict compression and stationarity guards are on by default. Disabling strict
mode is a diagnostic experiment, not a safe evolution setting.

`run_ablations.py` uses dense two-spin oracles and is explicitly separate from
the production solver. `product_scaling.py` checks only a force on a product
path, not an integrated trajectory or a correlated scalability claim.

## Report and figures

`plot_results.py` builds the figures and primary matched-accuracy tables from
saved authoritative datasets. `build_report_pdf.py` converts the Markdown note
to LaTeX and runs an installed Tectonic with cached packages. The PDF build
requires Tectonic and its TeX bundle; it does not rerun numerical experiments.


## Increasing-chain-size comparison with root TDVP and the positive witness

The reproducible [size-scaling benchmark runner](size_scaling.py) executes
correlated mixed-chain D trajectories in isolated, single-BLAS-thread workers.
Small-system direct factor and conventional root trajectories use the existing
finite-reference adapter. The native root+witness and correction-only methods
also run at larger chain lengths without importing dense reference scoring.
Failed, time-limited, and uncertified runs remain in the output.

Run from the repository root:

```sh
python -m factor_tdvp_accelerated.size_scaling \
  --suite smoke --timeout 90 --out run_outputs/size-scaling-smoke
python -m factor_tdvp_accelerated.size_scaling \
  --suite full --timeout 90 --out run_outputs/size-scaling-full
```

The [automated benchmark](../.github/workflows/factor-vs-witness-scaling.yml)
runs the full D suite and publishes each run into a unique
`size_scaling_results/actions-RUN_ID-attempt-ATTEMPT/` directory. Publication
fetches, rebases and retries an ordinary push after concurrent branch advances;
it never force-pushes or replaces an existing measurement directory. Raw results
are uploaded before publication. The [verified successful run](size_scaling_results/actions-43747960010-attempt-1/report.md)
also records CPU, RAM, BLAS configuration and the inherited 6 GiB address-space cap.
The [recovered earlier artifact](scaling_runs/actions_43745844867/RECOVERY.md)
retains every original file and the failed publication log.

### Repeated D/E study with detailed cost accounting

The [new report](scaling_report.md) and [complete tables](scaling_figures/tables.md)
compare conventional root, root plus witness, matrix-free factor and
correction-only evolution. The runner saves endpoint states, solver diagnostics,
exclusive/inclusive timing categories, peak memory, errors, bounds and failures.
Use a fresh output path; historical results are never overwritten.

```sh
python3 -m factor_tdvp_accelerated.repeated_scaling --help
python3 -m factor_tdvp_accelerated.repeated_scaling \
  --out run_outputs/repeated-scaling-new --timeout 45 --memory-mib 2048 --budget 840
python3 -m factor_tdvp_accelerated.repeated_scaling --suite supplemental \
  --out run_outputs/repeated-scaling-extra --timeout 60 --memory-mib 2048 --budget 420
python3 -m factor_tdvp_accelerated.repeated_scaling --suite parity_eight \
  --out run_outputs/parity-eight-new --timeout 45 --memory-mib 2048 --budget 180
python3 -m factor_tdvp_accelerated.analyze_scaling --help
```

The supplemental suite refines parity E3 to 50/150 steps, retries the larger D
witness cases with a 120-second limit, and tests a larger E8 factor work bond.
Its case-specific time limits are retained with each configuration.
The `parity_eight` suite retains the unsuccessful 50-step refinement and its
SVD failure for diagnosis. Run the new
accounting and evidence-selection regressions with:

```sh
python3 -m unittest factor_tdvp_accelerated.test_repeated_scaling \
  factor_tdvp_accelerated.test_frontier_evidence \
  factor_tdvp_accelerated.test_analyze_scaling -v
python3 -m unittest discover -s .github/scripts -p 'test_publish*.py' -v
```

See [the recorded local run](scaling_runs/local_20261115/) for exact configurations,
additional parity refinements, source hashes and hardware. The early local runs
had unavailable parent RSS monitoring; their child-reported peak memory remains
valid, but their configured cap was not enforced. Later workers add a tested
child RSS watchdog. The report distinguishes these cohorts explicitly.

Matched-error frontiers require three successful repetitions of a configuration
that each meet the target. Single exploratory runs, failed or unsupported cases,
and cases omitted by the time budget are not used as repeated performance
evidence. Missing rows must **not** be represented as measured infinite time.

**Important scope:** The corrected-root methods access separately evolved
physical density and an inverse-bound metric, whereas direct factor and
ordinary root methods do not. The factor evolution driver retains dense
rank/support checks even though its tangent force is matrix-free. For large D chains,
the witness supplies a global certified bound under its reference assumptions;
the parity E family has an analytic global MPO-root reference at every size.
Therefore size scaling can establish
an empirical *current-implementation* cost-to-accuracy ranking on successful
overlaps, not universal or asymptotic TDVP superiority.
