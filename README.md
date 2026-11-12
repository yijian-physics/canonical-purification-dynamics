# Canonical purification dynamics

Research implementations for evolving the canonical purification of a mixed spin state under local noise.

The current scope assumes a specified path of physical channels whose steps are locally reversible on the states encountered. The two approaches are tensor TDVP with explicit rank-opening treatment and finite local lifts on buffered physical and reference sites. The new half-Petz implementation constructs the latter directly from physical marginals and the channel.

Read [the algorithm note](two_ways_to_evolve_canonical_purification.md) for the assumptions, algorithms and numerical evidence. The note proposes a common-trajectory comparison at matched global CP error. That comparison remains open.

**Latest comparison.** On a tilted-noise benchmark, fitting a three-site unitary gives about **289 times lower global CP error** than a three-site half-Petz update at tilt 0.1. A five-site active window restores numerical precision for both. The accuracy difference persists through 256 spins. The unitary fit uses five-site physical data, and half-Petz uses its active window. These are single-update results. [Comparison report and figure](tilted_half_petz_comparison/report.md)

**TDVP at 512 spins.** A positive MPO witness and an independent density-residual check bounded the global CP error by $2.97\times10^{-6}$ on a correlated mixed chain under uniform dephasing. On an exactly solvable noncommuting parity ensemble at the same size, the measured CP error was $2.32\times10^{-7}$. The full-rank extension and its scaling limits are described in the [witness study](scalable_cp_witness/report.md).

## Implementations

| Directory or entry point | Content |
|---|---|
| [cp_tdvp](cp_tdvp/report.md) | MPS tensor TDVP, square-root-time integration, dense and single-excitation force backends |
| [scalable_cp_witness](scalable_cp_witness/report.md) | Two-site TDVP, positive-factor correction, global CP bounds and a long-range parity benchmark through 512 spins |
| [tilted_half_petz_comparison](tilted_half_petz_comparison/report.md) | Half-Petz versus local-data fitting and the exact-target unitary optimum, through 256 spins |
| [half_petz_evolution](half_petz_evolution/report.md) | Explicit half-Petz MPS evolution, local Kraus-factor caches and global CP-error benchmarks through 256 spins |
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
python half_petz_evolution/validate_half_petz_evolution.py
python half_petz_evolution/test_half_petz_evolution.py --n 6 64 256 --out run_outputs/half-petz
python local_cp_large/local_cp_large.py --help
python cat_buffer_counterexample/cat_buffer_counterexample.py
```

The individual reports provide the larger benchmark commands. Some runs require substantially more time or memory than the validation commands. Compare global vector errors, physical marginal errors and doubled-site marginal errors using the definitions in each report.

## Recorded results and corrections

The reports retain the experimental settings and numerical limitations of their studies. Generic Ising and single-excitation stress tests exercise numerical components, and the explicit cluster recovery supplies a benchmark within the local reversibility promise. A recovery radius for each generic stress-test trajectory remains to be established.

The restricted reference-compression Kraus update in the historical handoff study has a different optimization domain from a general unitary on doubled sites. Its GHZ failure concerns a step outside the local reversibility promise.

The uploaded handoff implementation includes the sweep-cache correction described in the cat regression. The recorded handoff tables were generated before that correction. Rechecks of the four-spin ten-step buffer-two, six-spin ten-step buffer-two, and six-spin 32-step buffer-three Kraus controls preserve their reported CP errors at the quoted precision. Other recorded rows remain historical measurements.

The recorded large-chain quantum CP checks compare finite-window doubled marginals. The classical Ising test additionally supplies an exterior-CMI bound. A generic efficient global CP certificate and a generic thermodynamic-limit implementation remain research targets.


## Explicit half-Petz evolution

The new engine applies the supported half-Petz forward map directly to MPS tensors. Local physical factors evolve with the known onsite channels independently of the numerical CP. The local factors are compressed by SVD to preserve small populations at the root scale. The update uses the active window alone and performs no variational unitary solve or post-update branch correction.

Starting from pure cluster and non-Clifford controlled-phase states, 20 sweeps to t=1 with a three-site window and bond four give global CP errors of 3.00e-13 and 5.40e-13 at 256 spins. Analytic MPS references enter only when scoring the runs. The report includes undersized-window and bond controls, cutoff and time-step checks, and the inverse-conditioning error at very small startup times. Generic onsite qubit Kraus maps are supported by the interface. The large-chain accuracy measurements cover the two explicitly locally reversible families described in the report.
