# Recovered GitHub Actions benchmark

The complete `factor-vs-witness-raw` artifact (ID `14174167664`, 165 entries)
was downloaded from [run 43745844867](https://github.com/xiaoyanghuangphys-create/canonical-purification-dynamics/actions/runs/43745844867)
on 2026-11-15. Its extracted original contents are unchanged.
`RECOVERY_MANIFEST.json` records the archive SHA256 and hashes of every
original extracted file. This note and `JOB_LOG.txt` are additional provenance,
not original artifact entries.

The run executed source commit `5cfb7dda1f231ae57a7206801ef5ed2460edeaa2`.
Benchmark execution and artifact upload both succeeded. Publication failed
because another commit advanced `research/factor-vs-witness-scaling`; the log
records an ordinary `fetch first` push rejection. No force push was used.

There are 31 trajectory measurement rows: 30 completed and one failed before
an accepted step. The failed D6 matrix-free row reports excessive compression
at work bond 32; its elapsed time is failure latency, not a completed runtime.
The native witness succeeds through N=256 at target bound 1e-5. Actual global
errors are measured only through N=8; larger rows report upper bounds.

These are single-run Linux CI observations. Python 3.11, NumPy 2.1.2 and SciPy
1.16.2 are recorded in the environment/job logs. The exact CPU model and RAM
capacity were not recorded, and are unknown. Do not pool these timings with
new local repetitions or interpret them as repeated estimates.

The artifact's original generated report/frontiers are preserved verbatim;
they are historical output, not the final interpretation. See the updated
[scaling report](../../scaling_report.md) for corrected method/resource
comparisons, repeated timings, and explicit coverage limitations.
