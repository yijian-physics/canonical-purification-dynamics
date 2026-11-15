"""Run isolated D-family factor/root/witness trajectories at increasing chain size.

Factor/root use the historical dense finite-size trajectory adapter. Witness
uses the native tensor-network solver directly so large N is NOT dense scored.
Never infer an asymptotic result from this finite, unequal-information test.
"""
from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
FACTOR = ROOT / "factor_tdvp_accelerated/run_benchmarks.py"
WITNESS = ROOT / "scalable_cp_witness/code/scalable_cp_witness.py"
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"


def cases(suite):
    fs = (3, 4, 5, 6) if suite == "full" else (3, 4)
    ws = (3, 4, 5, 6, 8, 16, 32, 64, 128, 256) if suite == "full" else (3, 4, 8)
    for n in fs:
        # N4 uses an unrestricted factor cap; N5/6 test restricted chi=4.
        cap = 16 if n == 4 else 4
        work = {3: 4, 4: 16, 5: 16, 6: 32}[n]
        grids = (5, 10, 20) if n == 3 else ((5, 10) if n == 4 else (5,))
        for steps in grids:
            for method in ("root", "matrix_free"):
                yield dict(kind="trajectory", family="D", method=method, n=n,
                           steps=steps, chi=cap, work_bond=work, total=.1,
                           solver_tolerance=1e-8)
    for n in ws:
        methods = ("witness", "correction_only") if n in (3, 4, 8) else ("witness",)
        for method in methods:
            tolerances = (1e-5, 1e-7) if n in (3, 4) else (1e-5,)
            for tol in tolerances:
                yield dict(kind="trajectory", family="D", method=method,
                           n=n, steps=5, total=.1, tolerance=tol)


def save_table(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(rows)


def native_witness(folder, returncode):
    sources = [p for p in folder.glob("*.csv") if not p.name.endswith("_trials.csv")]
    if len(sources) != 1:
        raise RuntimeError("Expected exactly one witness endpoint CSV")
    with sources[0].open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError("Missing native endpoint data")
    last = rows[-1]
    def val(k):
        return float(last[k]) if last.get(k) else None
    success = last.get("accepted", "").lower() == "true" and returncode == 0
    return dict(status="completed" if success else "failed_certificate",
                global_cp_error=val("actual"), global_cp_bound=val("bound"),
                reached_time=val("last_accepted"), native_seconds=val("seconds"),
                root_bond=int(last["root_bond"]), factor_bond=int(last["factor_bond"]),
                accuracy_kind="measured" if last.get("actual") else "upper_bound_only")


def run_case(config, out, timeout):
    n, method, steps = config["n"], config["method"], config["steps"]
    folder = out / f'D{n:03d}_{method}_{steps:02d}_tol{config.get("tolerance",0):g}'
    folder.mkdir(parents=True)
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    if method in ("root", "matrix_free"):
        command = [sys.executable, str(FACTOR), "--worker", json.dumps(config),
                   "--out", str(folder)]
    else:
        command = [sys.executable, str(WITNESS), "--n", str(n), "--times",
                   *[f"{0.1*i/steps:.12g}" for i in range(1, steps+1)],
                   "--tol", str(config["tolerance"]), "--passes", "8",
                   "--levels", "4,6,10:6,8,12:8,12,16", "--out", str(folder)]
        if method == "witness":
            command.append("--tdvp")
    started = perf_counter()
    try:
        with (folder / "stdout.txt").open("w") as f:
            p = subprocess.run(command, cwd=ROOT, env=env, stdout=f,
                               stderr=subprocess.STDOUT, timeout=timeout)
        if method in ("root", "matrix_free"):
            summary = folder/"summary.json"
            if not summary.exists():
                raise RuntimeError(f"Missing trajectory summary, code {p.returncode}")
            raw = json.loads(summary.read_text())
            extra = dict(status=raw.get("status", "failed"),
                         global_cp_error=raw.get("global_cp_error"),
                         global_cp_bound=None, native_seconds=raw.get("total_seconds"),
                         root_bond=raw.get("x_bond") if method == "matrix_free"
                                   else raw.get("chi"),
                         factor_bond=raw.get("h_bond") if method == "matrix_free" else None,
                         error_message=raw.get("error_message"),
                         peak_process_memory_mib=raw.get("peak_process_memory_mib"),
                         accuracy_kind="measured")
        else:
            extra = native_witness(folder,p.returncode)
    except subprocess.TimeoutExpired:
        extra = dict(status="timeout", error_message=f"Exceeded {timeout} s")
    except Exception as exc:
        extra = dict(status="failed",error_message=f"{type(exc).__name__}: {exc}")
    row = dict(config, **extra, wall_seconds=perf_counter()-started,
               directory=folder.name)
    (folder/"measurement.json").write_text(json.dumps(row,indent=2)+"\n")
    return row


def frontier(rows, target):
    result = []
    for n in sorted({r["n"] for r in rows}):
        for method in ("root", "matrix_free", "witness", "correction_only"):
            choices = []
            for r in rows:
                if r["n"] != n or r["method"] != method or r["status"] != "completed":
                    continue
                error = r.get("global_cp_error")
                bound = r.get("global_cp_bound")
                if (error is not None and error <= target) or (
                    error is None and bound is not None and bound <= target):
                    choices.append(r)
            best = min(choices,key=lambda r:r["wall_seconds"]) if choices else None
            result.append(dict(n=n, method=method, target=target,
                               attained=best is not None,
                               wall_seconds=best["wall_seconds"] if best else "",
                               evidence=("measured" if best and
                                         best.get("global_cp_error") is not None else
                                         "bound" if best else ""),
                               directory=best["directory"] if best else ""))
    return result


def report(out,rows,suite):
    lines = [
        "# Increasing-size canonical-purification TDVP comparison",
        "",
        "Executed benchmarks, **not a proof of an asymptotic ordering**. All use "
        "correlated full-rank mixed-chain family D, final time 0.1 and uniform "
        "grids. Each case runs in a separate one-BLAS-thread process. The wall "
        "times include subprocess startup and scoring. Single observations "
        "are not confidence intervals.",
        "",
        "Root+witness and witness-only corrections have an independently propagated "
        "density and inverse-metric bound unavailable to plain root/factor "
        "trajectories: this is an unequal-information comparison. The witness "
        "global CP quantity is a measured error through N=8 and otherwise an "
        "upper bound (conditional on correct reference/metric). Never replace "
        "the latter by a measured error.",
        "",
        "Direct factor uses the accelerated MPO-LSQR force but its driver retains "
        "dense support audits and finite-size scoring. N=4 factor cap is 16; "
        "N=5,6 cap is 4 and hence restricted, not a full-capacity comparison. "
        "The factor-implied raw root bond can be as large as chi_H squared.",
        "",
        "## Trajectory measurements",
        "",
        "| N | Method | Steps | Status | Measured CP error | CP upper bound | Wall s |",
        "|--:|:--|--:|:--|--:|--:|--:|",
    ]
    def fmt(v):
        return "—" if v is None else f"{v:.3g}"
    for r in rows:
        lines.append(f'| {r["n"]} | {r["method"]} | {r["steps"]} '
                     f'| {r["status"]} | {fmt(r.get("global_cp_error"))} '
                     f'| {fmt(r.get("global_cp_bound"))} '
                     f'| {fmt(r["wall_seconds"])} |')
    lines.extend([
        "", "## Cheapest observed qualifying run at target accuracy",
        "",
        "A completed row qualifies if its measured global error or separately "
        "reported global upper bound is below target. A dash means not attained "
        "among the executed rows, not that the method can never succeed.",
        "",
    ])
    for target in (1e-5, 1e-7):
        lines += [f"### Target {target:g}", "",
                  "| N | Plain root | Direct factor | Root + witness | Correction only |",
                  "|--:|--:|--:|--:|--:|"]
        data = frontier(rows,target)
        for n in sorted({r["n"] for r in rows}):
            def cell(method):
                item = next(r for r in data if r["n"] == n and r["method"] == method)
                return (f'{item["wall_seconds"]:.3g}s ({item["evidence"]})'
                        if item["attained"] else "—")
            lines.append(f"| {n} | {cell('root')} | {cell('matrix_free')} "
                         f"| {cell('witness')} | {cell('correction_only')} |")
        lines.append("")
    lines.extend([
        "## Scientific interpretation and limitations",
        "",
        "An advantage of root+witness at matched global accuracy can arise from "
        "fewer time steps thanks to cheap positive-factor endpoint correction. "
        "It does not show that each corrected TDVP step is faster. The factor "
        "method spends time on real-linear Jacobian/adjoint Krylov solves "
        "and can be ill-conditioned at small physical populations. "
        "Unsupported, timed-out and inaccurate factor trajectories cannot be "
        "converted into invented runtime or an asymptotic lower bound.",
        "",
        "The strongest defensible conclusion is implementation- and "
        "benchmark-specific; a universal or asymptotic speed ordering would "
        "require matched-information, matched-accuracy families and genuine "
        "polynomial-cost support handling for the factor method.",
        "",
        "Reproduction: python -m factor_tdvp_accelerated.size_scaling "
        f"--suite {suite} --out <new-directory>",
        "",
    ])
    (out/"report.md").write_text("\n".join(lines))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--suite",choices=("smoke","full"),default="full")
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--timeout",type=float,default=90.)
    a = p.parse_args()
    if a.timeout<=0:
        p.error("Timeout must be positive")
    a.out.mkdir(parents=True,exist_ok=False)
    (a.out/"environment.json").write_text(json.dumps({
        "suite":a.suite,"command":sys.argv,"python":sys.version,
        "timeout_per_worker":a.timeout,"BLAS_threads":1},indent=2)+"\n")
    rows=[]
    for config in cases(a.suite):
        row=run_case(config,a.out,a.timeout)
        rows.append(row)
        save_table(a.out/"measurements.csv",rows)
        (a.out/"measurements.json").write_text(json.dumps(rows,indent=2)+"\n")
        report(a.out,rows,a.suite)
        print(f'{row["directory"]}: {row["status"]}, '
              f'error={row.get("global_cp_error")}, '
              f'bound={row.get("global_cp_bound")}, '
              f'wall={row["wall_seconds"]:.3g}s',flush=True)
    for target in (1e-5,1e-7):
        save_table(a.out/f"frontier_{target:g}.csv",frontier(rows,target))


if __name__=="__main__":
    main()
