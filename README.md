# Canonical purification dynamics

Research implementations for evolving the canonical purification of a mixed spin state under local noise.

The current scope assumes a specified path of physical channels whose steps are locally reversible on the states encountered. The two approaches are tensor TDVP with explicit rank-opening treatment and finite unitary updates on buffered physical and reference sites.

Read [the algorithm note](two_ways_to_evolve_canonical_purification.md) for the assumptions, algorithms and numerical evidence. The note proposes a common-trajectory comparison at matched global CP error. That comparison remains open.

## Implementations

| Directory or entry point | Content |
|---|---|
| [cp_tdvp](cp_tdvp/report.md) | MPS tensor TDVP, square-root-time integration, dense and single-excitation force backends |
| [local_cp_large](local_cp_large/report.md) | Finite doubled-site unitary fitting from physical marginals, MPS application and local error checks |
| [automatic_rank_opening](automatic_rank_opening/report.md) | Support leakage detection, finite positive startup and guarded continuation |
| [cluster_dephasing.py](cluster_dephasing.py) | Exact cluster-state benchmark with bounded CP bond dimension |
| [cp_branch_methods.py](cp_branch_methods.py) | Cluster gate-support, angle-scaling and tensor-search controls |
| [kraus_tdvp_handoff](kraus_tdvp_handoff/report.md) | Historical four- and six-spin handoff experiment and force-solver controls |
| [cat_buffer_counterexample](cat_buffer_counterexample/report.md) | Regression for the restricted reference-compression update on GHZ dephasing |

Each implementation folder carries its Python dependencies from this project so its scripts can be run directly. Recorded CSV results and selected state tensors accompany the reports. Python 3.11 was used for the recorded runs.

## Reproduce

From the repository root, install the pinned numerical dependencies and run the checks.

```sh
python -m pip install -r requirements.txt
python cluster_dephasing.py --outdir run_outputs/cluster
python cp_tdvp/validate_tdvp.py --outdir run_outputs/tdvp-validation
python cp_tdvp/cp_tdvp_evolution.py --part clock --outdir run_outputs/clock
python local_cp_large/local_cp_large.py --help
python cat_buffer_counterexample/cat_buffer_counterexample.py
```

The individual reports provide the larger benchmark commands. Some runs require substantially more time or memory than the validation commands. Compare global vector errors, physical marginal errors and doubled-site marginal errors using the definitions in each report.

## Recorded results and corrections

The reports retain the experimental settings and numerical limitations of their studies. Generic Ising and single-excitation stress tests exercise numerical components, and the explicit cluster recovery supplies a benchmark within the local reversibility promise. A recovery radius for each generic stress-test trajectory remains to be established.

The restricted reference-compression Kraus update in the historical handoff study has a different optimization domain from a general unitary on doubled sites. Its GHZ failure concerns a step outside the local reversibility promise.

The uploaded handoff implementation includes the sweep-cache correction described in the cat regression. The recorded handoff tables were generated before that correction. Rechecks of the four-spin ten-step buffer-two, six-spin ten-step buffer-two, and six-spin 32-step buffer-three Kraus controls preserve their reported CP errors at the quoted precision. Other recorded rows remain historical measurements.

The recorded large-chain quantum CP checks compare finite-window doubled marginals. The classical Ising test additionally supplies an exterior-CMI bound. A generic efficient global CP certificate and a generic thermodynamic-limit implementation remain research targets.
