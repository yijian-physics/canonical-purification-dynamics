# Canonical-purification algorithm comparison

Run these commands from the repository root. The recovered direct factor-MPO implementation is in `factor_tdvp/`; the comparison adapters reuse the historical algorithms without overwriting their results.

```sh
python -m pip install -r factor_tdvp_comparison/requirements.txt
OPENBLAS_NUM_THREADS=1 python -m unittest discover -s factor_tdvp -p 'test_*.py' -v
OPENBLAS_NUM_THREADS=1 python -m unittest discover -s factor_tdvp_comparison -p 'test_*.py' -v
OPENBLAS_NUM_THREADS=1 python factor_tdvp_comparison/run_comparison.py --component all
```

The unified driver supports `--component factor`, `--component lifts`, `--component witness`, and `--component summarize` for component reruns and regenerating combined tables/figures. Read `--help` before changing output directories or numerical budgets. The named components can be costly; the recorded factor coverage is intentionally limited to sizes that the current explicit-Jacobian backend can actually execute. Requested larger cases are retained as unsupported or budget-limited rows.

The following additional commands reproduce the budgeted scaling attempts, force profile, and repeated-unitary audit. The extended run deliberately includes a larger attempt that may be killed by the environment; it is not required to claim a successful trajectory.

```sh
OPENBLAS_NUM_THREADS=1 python factor_tdvp_comparison/factor_benchmarks.py --extended
OPENBLAS_NUM_THREADS=1 python factor_tdvp_comparison/factor_benchmarks.py --force-profile --n 6 --work-bond 16
OPENBLAS_NUM_THREADS=1 python factor_tdvp_comparison/audit_local_unitary.py
OPENBLAS_NUM_THREADS=1 python factor_tdvp_comparison/run_comparison.py --component summarize
```

| File or directory | Purpose |
|---|---|
| `report.md` | Methods, measured conclusions, fairness restrictions, unresolved work |
| `run_comparison.py` | Unified runner, aggregation, comparison plots |
| `factor_benchmarks.py` | Direct factor and conventional root small-system trajectories/profiling |
| `lift_benchmarks.py` | Repeated half-Petz/local-unitary and tilted single-channel comparisons |
| `witness_benchmarks.py` | Original positive-witness and corrected-critical controls |
| `comparison_metrics.py`, `test_comparison.py` | Independent reference/metric checks |
| `audit_local_unitary.py` | Independent cache, gate-application and occupied-span audit of a repeated cluster fitting failure |
| `results/` | New endpoint tables, histories, profile data, failure records, and executed test output |
| `figures/` | Comparison figures generated from the new tables |

Global measured CP errors are raw Frobenius distances to the positive canonical root. No purification-gauge optimization is used for scoring. Analytic upper bounds and local diagnostics have separate labels; an absent error is never interpreted as zero. `total_seconds` and evolution-only times answer different cost questions. See the report before comparing method costs or bond caps.

`results/matched_accuracy.csv` reports the best observed setup-plus-evolution cost on the measured grids, and `results/matched_accuracy_end_to_end.csv` uses the common total time including scoring. A witness result can attain an accuracy target while its separate setup/evolution breakdown is unavailable; this is not relabeled as an accuracy failure. Neither frontier interpolates unmeasured runs or claims an optimal complexity law.

The fork used for this work is `xiaoyanghuangphys-create/canonical-purification-dynamics`, branch `research/factor-tdvp-benchmarks`. No publication to the original upstream repository is part of this task.
