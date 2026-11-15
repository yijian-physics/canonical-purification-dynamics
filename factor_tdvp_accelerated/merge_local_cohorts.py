"""Merge immutable measured cohorts and add separate provenance annotations."""
from pathlib import Path
import argparse
import collections
import hashlib
import json
from factor_tdvp_accelerated.run_benchmarks import save, table
from factor_tdvp_accelerated.repeated_scaling import repeated_frontier


def merge(out):
    sources=[('measurements_main.json',''),('supplemental/measurements.json','supplemental/'),('parity_eight/measurements.json','parity_eight/')]
    rows=[];checksums={}
    for filename,prefix in sources:
        path=out/filename
        checksums[filename]=hashlib.sha256(path.read_bytes()).hexdigest()
        for raw in json.loads(path.read_text()):
            row=dict(raw)
            if 'directory' in row:row['directory']=prefix+row['directory']
            rows.append(row)
    save(out/'measurements.json',rows);table(out/'measurements.csv',rows)
    for target in (1e-5,1e-6,1e-7):
        data=repeated_frontier(rows,target)
        save(out/f'frontier_{target:g}.json',data);table(out/f'frontier_{target:g}.csv',data)
    annotations=[]
    for row in rows:
        if row['status']=='unsupported':guard='not_executed'
        elif row.get('memory_enforcement'):guard='child_watchdog_reported_by_worker'
        elif row.get('peak_process_memory_mib') is not None:guard='no_verified_active_memory_guard; normal_exit_high_water_only'
        else:guard='no_completion_guard_metadata; time_budget_enforced'
        digest=row.get('worker_instrumentation_sha256')
        annotations.append(dict(directory=row.get('directory'),config_id=row['config_id'],repeat=row['repeat'],
                                memory_guard_evidence=guard,parent_rss_monitor_available=row.get('peak_observed_rss_mib',0)>0,
                                peak_process_memory_mib=row.get('peak_process_memory_mib'),
                                worker_instrumentation_sha256=digest,
                                exact_worker_source_snapshot_available=bool(digest and (out/'instrumentation_sources'/f'{digest}.py').exists())))
    save(out/'resource_guard_audit.json',dict(note='Sidecar supersedes any implied parent RSS enforcement in original row text; original measurements unchanged.',counts=dict(collections.Counter(a['memory_guard_evidence'] for a in annotations)),rows=annotations))
    save(out/'collection_provenance.json',dict(source_tables_sha256=checksums,row_count=len(rows),status_counts=dict(collections.Counter(r['status'] for r in rows)),
         row_transform='Only prefix supplemental/ or parity_eight/ to appended directory paths. No numerical values, runtimes, original statuses or original monitor text changed.',
         source_note='Initial runner exact snapshot not retained; environment keeps source-at-start hash. Later source snapshots identified by hash are retained. Numerical solver source unchanged.',
         resource_note='Sandbox denied parent ps monitor. Later sampled child RSS guard was validated; early normal-exit high-water measurements do not prove cap enforcement.',
         verification='Each immutable source table checksum retained. No overlapping or duplicate constituent table is included twice.'))
    print(json.dumps(dict(rows=len(rows),statuses=dict(collections.Counter(r['status'] for r in rows)))))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);a=p.parse_args();merge(a.directory)
