#!/usr/bin/env python3
"""Append execution metadata without replacing the benchmark environment file."""
import argparse
import contextlib
from datetime import datetime, timezone
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', required=True, type=Path)
    parser.add_argument('--address-space-limit-bytes', required=True, type=int)
    args = parser.parse_args()
    if not args.results.is_dir() or args.address_space_limit_bytes < 1:
        parser.error('An existing results directory and positive address-space limit are required')
    cpuinfo = Path('/proc/cpuinfo')
    cpu_model = None
    if cpuinfo.exists():
        cpu_model = next((line.split(':', 1)[1].strip() for line in cpuinfo.read_text().splitlines()
                          if line.startswith('model name')), None)
    dependencies = {}
    for name in ('numpy', 'scipy', 'matplotlib', 'threadpoolctl'):
        try:
            dependencies[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            dependencies[name] = None
    metadata = {
        'recorded_at_utc': datetime.now(timezone.utc).isoformat(),
        'source_sha_executed': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'python': sys.version, 'python_executable': sys.executable,
        'platform': platform.platform(), 'uname': platform.uname()._asdict(),
        'cpu_model': cpu_model, 'logical_cpu_count': os.cpu_count(),
        'cpu_affinity': sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
        'physical_memory_bytes': os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES'),
        'dependencies': dependencies,
        'thread_environment': {key: os.environ.get(key) for key in (
            'OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')},
        'benchmark_resource_limit': {
            'kind': 'Linux RLIMIT_AS inherited by each benchmark subprocess',
            'address_space_limit_bytes': args.address_space_limit_bytes,
            'scope': 'per process virtual address space; not aggregate RSS',
            'enforcement': 'workflow shell ulimit -v before launching the suite'},
        'github_run_id': os.environ.get('GITHUB_RUN_ID'),
        'github_run_attempt': os.environ.get('GITHUB_RUN_ATTEMPT'),
    }
    try:
        import numpy as np
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            np.show_config()
        metadata['numpy_blas_configuration'] = captured.getvalue()
    except Exception as exc:
        metadata['numpy_blas_configuration_error'] = f'{type(exc).__name__}: {exc}'
    try:
        from threadpoolctl import threadpool_info
        metadata['threadpools'] = threadpool_info()
    except ImportError:
        metadata['threadpools'] = None
    # Exclusive creation preserves an existing historical record on retries.
    with (args.results / 'hardware.json').open('x') as stream:
        json.dump(metadata, stream, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
