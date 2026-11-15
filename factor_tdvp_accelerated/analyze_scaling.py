"""Audit immutable recovered data and summarize repeated, same-host timings.

No failed run is converted into a performance observation. Repeated frontiers
require every requested replicate to complete and meet the error target; they
select the lowest MEDIAN worker time across executed configurations only.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np

HERE = Path(__file__).resolve().parent
METHODS = ('root', 'witness', 'matrix_free', 'correction_only')
LABELS = dict(root='Root TDVP', witness='Root + witness', matrix_free='Factor LSQR',
              correction_only='Correction only')
COLORS = dict(root='#38598b', witness='#16817a', matrix_free='#ba4638', correction_only='#9b6ab6')
TARGETS = (1e-5, 1e-6, 1e-7)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def save_csv(path, rows):
    if not rows:
        path.write_text('')
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        for row in rows:
            w.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k,v in row.items()})


def audit_recovered(folder):
    manifest = json.loads((folder/'RECOVERY_MANIFEST.json').read_text())
    files = manifest['files']
    failures, inventory = [], []
    for name, expected in sorted(files.items()):
        path = folder/name
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
        if actual != expected:
            failures.append(f'Hash mismatch or missing file: {name}')
        item = dict(path=name, bytes=path.stat().st_size if path.exists() else None,
                    sha256=actual, unchanged=actual==expected)
        # Parse every structured artifact, not just top-level measurements.
        if path.suffix == '.json':
            obj = json.loads(path.read_text())
            item['json_kind'] = type(obj).__name__
            item['entries'] = len(obj)
        elif path.suffix == '.csv':
            with path.open(newline='') as f:
                parsed = list(csv.DictReader(f))
            item['rows'] = len(parsed)
        elif path.suffix == '.npz':
            with np.load(path, allow_pickle=False) as arrays:
                item['arrays'] = {k: list(arrays[k].shape) for k in arrays.files}
                item['all_finite'] = all(np.isfinite(arrays[k]).all() for k in arrays.files)
                if not item['all_finite']:
                    failures.append(f'Nonfinite saved arrays: {name}')
        else:
            content = path.read_text()
            item['lines'] = len(content.splitlines())
            item['traceback_lines'] = [s for s in content.splitlines() if 'Traceback' in s]
        inventory.append(item)
    rows = json.loads((folder/'measurements.json').read_text())
    cases = []
    for row in rows:
        directory = folder/row['directory']
        case = dict(directory=row['directory'], n=row['n'], method=row['method'],
                    steps=row['steps'], status=row['status'], wall_seconds=row['wall_seconds'],
                    global_cp_error=row.get('global_cp_error'), global_cp_bound=row.get('global_cp_bound'))
        if json.loads((directory/'measurement.json').read_text()) != row:
            failures.append(f'Individual/top-level measurement mismatch: {directory.name}')
        stdout = (directory/'stdout.txt').read_text()
        case['stdout_lines'] = len(stdout.splitlines())
        summary = directory/'summary.json'
        if summary.exists():
            raw = json.loads(summary.read_text())
            # The final structured stdout is the source for the summary.
            objects = []
            for line in stdout.splitlines():
                try:
                    value = json.loads(line)
                    if isinstance(value, dict):
                        objects.append(value)
                except (ValueError, TypeError):
                    pass
            case['stdout_summary_agrees'] = bool(objects and objects[-1] == raw)
            if not case['stdout_summary_agrees']:
                failures.append(f'Stdout/summary disagreement: {directory.name}')
            if row['status'] != raw['status'] or row.get('global_cp_error') != raw.get('global_cp_error'):
                failures.append(f'Summary/measurement disagreement: {directory.name}')
            for key in ('accepted_steps','reached_time','error_message','setup_seconds',
                        'evolution_seconds','scoring_seconds','force_seconds','total_seconds',
                        'max_krylov_iterations','max_conditioning','max_stationarity_relative',
                        'h_bond','x_bond','peak_process_memory_mib'):
                case[key] = raw.get(key)
            solves = directory/'solves.json'
            if solves.exists():
                solve_records = json.loads(solves.read_text())
                case['solve_records'] = len(solve_records)
        else:
            endpoints = [p for p in directory.glob('*.csv') if not p.name.endswith('_trials.csv')]
            if len(endpoints) != 1:
                failures.append(f'Missing or ambiguous endpoint CSV: {directory.name}')
                continue
            with endpoints[0].open(newline='') as f:
                data = list(csv.DictReader(f))
            last = data[-1]
            case.update(endpoint_rows=len(data), all_endpoints_accepted=all(r['accepted']=='True' for r in data),
                        reached_time=float(last['last_accepted']), total_seconds=float(last['seconds']),
                        h_bond=int(last['factor_bond']), x_bond=int(last['root_bond']),
                        prediction_bound=float(last['prediction_bound']) if last.get('prediction_bound') else None,
                        witness_defect=float(last['witness']), weighted_residual=float(last['weighted_residual']))
            for field, column in (('global_cp_error','actual'), ('global_cp_bound','bound')):
                value = float(last[column]) if last.get(column) else None
                if value != row.get(field):
                    failures.append(f'Endpoint/measurement {field} disagreement: {directory.name}')
            case['endpoint_corrections'] = 0
            trials = list(directory.glob('*_trials.csv'))
            if trials:
                with trials[0].open(newline='') as f:
                    trial_rows = list(csv.DictReader(f))
                case['correction_trials'] = len(trial_rows)
                case['improving_correction_trials'] = sum(x['improved']=='True' for x in trial_rows)
                case['endpoint_corrections'] = len({x['time'] for x in trial_rows})
            else:
                case['correction_trials'] = 0
            if row['status']=='completed' and (not case['all_endpoints_accepted'] or case['reached_time'] < row['total']-1e-12):
                failures.append(f'False completion: {directory.name}')
        cases.append(case)
    return dict(run_id=manifest['run_id'], artifact_id=manifest['artifact_id'],
                archive_sha256=manifest['archive_sha256'], files_checked=len(files),
                case_count=len(cases), status_counts=dict(Counter(r['status'] for r in rows)),
                all_original_hashes_match=not failures, failures=failures,
                environment=json.loads((folder/'environment.json').read_text()),
                files=inventory, cases=cases)


def summarize_local(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['config_id']].append(row)
    result = []
    for config_id, group in grouped.items():
        first = group[0]
        success = [r for r in group if r['status']=='completed']
        item = {k:first.get(k) for k in ('family','method','n','steps','chi','work_bond','tolerance',
                                        'solver_tolerance','total','requested_repeats')}
        item.update(config_id=config_id, observations=len(group), completed=len(success),
                    status_counts=dict(Counter(r['status'] for r in group)),
                    directories=[r['directory'] for r in group if r.get('directory')],
                    errors=[r.get('error_message') or r.get('unsupported_reason') for r in group
                            if r.get('error_message') or r.get('unsupported_reason')])
        item['repeat_complete'] = (len(success) == len(group) == first.get('requested_repeats',1)
                                   and len(success)>=3 and
                                   {r.get('repeat') for r in group}==set(range(1,len(group)+1)))
        chosen=[r.get('global_cp_error') if r.get('global_cp_error') is not None else r.get('global_cp_bound') for r in success]
        item['qualification_complete']=bool(success and all(v is not None and np.isfinite(v) and v>=0 for v in chosen))
        item['qualification_error_max']=max(chosen) if item['qualification_complete'] else None
        kinds={'measured' if r.get('global_cp_error') is not None else 'conditional_bound' for r in success}
        item['qualification_evidence']=next(iter(kinds)) if len(kinds)==1 else 'mixed_measured_and_bound'
        item['runtime_complete']=bool(success and all(r.get('total_seconds') is not None and
                                      np.isfinite(r['total_seconds']) and r['total_seconds']>0 for r in success))
        for key in ('total_seconds','wall_seconds','global_cp_error','global_cp_bound',
                    'peak_process_memory_mib','root_bond','factor_bond','max_conditioning',
                    'max_krylov_iterations','max_stationarity_relative','max_relative_force_residual',
                    'krylov_iterations_total','force_solve_count','correction_trial_count',
                    'accepted_correction_trials','used_predictions'):
            vals = [float(r[key]) for r in success if r.get(key) is not None and np.isfinite(r[key])]
            for stat, func in (('median',statistics.median),('min',min),('max',max)):
                item[key+'_'+stat] = func(vals) if vals else None
        categories = sorted({k for r in success for k in r.get('timing_exclusive_seconds',{})})
        item['exclusive_seconds_median'] = {k:statistics.median([r.get('timing_exclusive_seconds',{}).get(k,0.) for r in success])
                                            for k in categories}
        result.append(item)
    return result


def audit_local(folder, rows):
    failures=[]
    checks={}
    provenance=folder/'collection_provenance.json'
    if provenance.exists():
        saved=json.loads(provenance.read_text())
        for name,expected in saved['source_tables_sha256'].items():
            path=folder/name
            checks[name]=path.exists() and hashlib.sha256(path.read_bytes()).hexdigest()==expected
            if not checks[name]:failures.append('Constituent table hash mismatch: '+name)
        if saved['row_count']!=len(rows):failures.append('Collection row count mismatch')
    for row in rows:
        if not row.get('directory'):
            if row['status'] not in ('unsupported','not_run_budget'):
                failures.append('Missing run directory: '+row['config_id'])
            continue
        path=folder/row['directory']/'measurement.json'
        original=json.loads(path.read_text()) if path.exists() else None
        if original and '/' in row['directory'] and original.get('directory')==Path(row['directory']).name:
            # Supplemental immutable measurement paths are relative to their
            # own output root; combined-table paths add that root prefix.
            original=dict(original,directory=row['directory'])
        if original!=row:
            failures.append('Measurement mismatch: '+row['directory'])
        if row['status']=='completed':
            if row.get('total_seconds') is None or not np.isfinite(row['total_seconds']) or row['total_seconds']<=0:
                failures.append('Invalid completed runtime: '+row['directory'])
            costs=row.get('timing_exclusive_seconds',{})
            if costs and abs(sum(costs.values())-row['total_seconds'])>1e-8:
                failures.append('Exclusive costs do not partition total: '+row['directory'])
            if row.get('global_cp_error') is None and row.get('global_cp_bound') is None:
                failures.append('Completed without accuracy evidence: '+row['directory'])
            for key in ('global_cp_error','global_cp_bound'):
                val=row.get(key)
                if val is not None and (not np.isfinite(val) or val<0):
                    failures.append('Invalid '+key+': '+row['directory'])
    return dict(rows=len(rows),status_counts=dict(Counter(r['status'] for r in rows)),
                constituent_table_hashes=checks,failures=failures,all_checks_pass=not failures)


def frontier(groups, repeats=True):
    out = []
    for family,n,total in sorted({(r['family'],r['n'],r.get('total')) for r in groups},key=lambda k:(k[0],k[1],k[2] or 0.)):
        for target in TARGETS:
            for method in METHODS:
                options = []
                for r in groups:
                    if (r['family'],r['n'],r['method'],r.get('total')) != (family,n,method,total):
                        continue
                    if repeats and not r['repeat_complete']:
                        continue
                    if r['completed'] != r['observations'] or not r['completed']:
                        continue
                    if r['runtime_complete'] and r['qualification_complete'] and r['qualification_error_max']<=target:
                        options.append(r)
                best = min(options,key=lambda r:r['total_seconds_median']) if options else None
                row=dict(family=family,n=n,total=total,target=target,method=method,attained=best is not None)
                if best:
                    row.update(config_id=best['config_id'], repeats=best['completed'],
                               seconds=best['total_seconds_median'], minimum=best['total_seconds_min'],
                               maximum=best['total_seconds_max'], wall_seconds=best['wall_seconds_median'],
                               error=best['global_cp_error_max'], bound=best['global_cp_bound_max'],
                               evidence=best['qualification_evidence'],
                               steps=best['steps'], tolerance=best['tolerance'], directories=best['directories'])
                out.append(row)
    return out


def fmt(value):
    return '—' if value is None else f'{value:.4g}'


def tables(out,audit,groups,matched):
    lines=['# Generated scaling tables','',
           'Regenerate with `python3 -m factor_tdvp_accelerated.analyze_scaling`. '
           'Recovered Actions observations and local repeated observations are separate cohorts.','',
           '## All recovered artifact measurements','',
           '| N | Method | Steps | Status | CP error | CP bound | Wall s | Raw root bond | Factor bond |',
           '|--:|---|--:|---|--:|--:|--:|--:|--:|']
    for r in audit['cases']:
        time=fmt(r['wall_seconds']) if r['status']=='completed' else 'failed; see raw log'
        lines.append(f"| {r['n']} | [{r['method']}](../scaling_runs/actions_43745844867/{r['directory']}/measurement.json) | {r['steps']} | {r['status']} | {fmt(r.get('global_cp_error'))} | {fmt(r.get('global_cp_bound'))} | {time} | {fmt(r.get('x_bond'))} | {fmt(r.get('h_bond'))} |")
    lines += ['', '## Local repeated matched accuracy','',
              'Each cell is median worker seconds [minimum, maximum], over three completed repeats. '
              'Every repeat must meet the target. A dash means no qualifying repeated configuration in this sweep. '
              'Worker totals include all measured auxiliary work and scoring, and exclude process/import startup. '
              'For measured-error cells, passing this table does not imply the online certificate also meets the target.','',
              '| Family | N | Final time | CP target | Root | Root + witness | Factor LSQR | Correction only |',
              '|---|--:|--:|--:|---:|---:|---:|---:|']
    for family,n,total in sorted({(r['family'],r['n'],r.get('total')) for r in matched},key=lambda k:(k[0],k[1],k[2] or 0.)):
        if not any(r['repeat_complete'] for r in groups if r['family']==family and r['n']==n and r.get('total')==total):
            continue
        for target in TARGETS:
            cells=[]
            for method in METHODS:
                row=next(r for r in matched if (r['family'],r['n'],r['method'],r['target'],r.get('total'))==(family,n,method,target,total))
                cell='—'
                if row['attained']:
                    cell=f"{row['seconds']:.4g} [{row['minimum']:.4g}, {row['maximum']:.4g}]"
                    if row['evidence']=='conditional_bound':cell+=' (bound)'
                cells.append(cell)
            lines.append(f"| {family} | {n} | {fmt(total)} | {target:g} | "+' | '.join(cells)+' |')
    lines += ['', '## Repeated factor solve diagnostics','',
              'Condition values are LSQR bidiagonal estimates, not exact singular-value condition numbers. '
              'Reported bonds are represented bonds, not equal-expressivity budgets.','',
              '| Family | N | Steps | Factor / raw-root bond | Median s | Median force s | Max solve iterations | Max condition estimate | Worst CP error |',
              '|---|--:|--:|---|--:|--:|--:|--:|--:|']
    for r in groups:
        if r['method']=='matrix_free' and r['repeat_complete']:
            lines.append(f"| {r['family']} | {r['n']} | {r['steps']} | {fmt(r['factor_bond_max'])} / {fmt(r['root_bond_max'])} | {fmt(r['total_seconds_median'])} | {fmt(r['exclusive_seconds_median'].get('factor_tangent_force'))} | {fmt(r['max_krylov_iterations_max'])} | {fmt(r['max_conditioning_max'])} | {fmt(r['global_cp_error_max'])} |")
    lines += ['', '## Unsuccessful or unexecuted local configurations','',
              'Timeouts, numerical failures, unsupported adapters and budget omissions are distinct; none supplies a completed runtime.','',
              '| Configuration | Status counts | Reason |','|---|---|---|']
    for r in groups:
        if r['completed'] != r['observations']:
            lines.append(f"| {r['config_id']} | {r['status_counts']} | "+'; '.join(dict.fromkeys(r['errors'])).replace('|','/')+' |')
    (out/'tables.md').write_text('\n'.join(lines)+'\n')


def recovered_frontier(rows):
    out=[]
    for n in sorted({r['n'] for r in rows}):
        for target in TARGETS:
            for method in METHODS:
                options=[r for r in rows if r['n']==n and r['method']==method and r['status']=='completed'
                         and ((r.get('global_cp_error') is not None and r['global_cp_error']<=target)
                         or (r.get('global_cp_error') is None and r.get('global_cp_bound') is not None
                             and r['global_cp_bound']<=target))]
                best=min(options,key=lambda r:r['wall_seconds']) if options else None
                out.append(dict(n=n,target=target,method=method,attained=bool(best),
                                wall_seconds=best['wall_seconds'] if best else None,
                                directory=best['directory'] if best else None))
    return out


def plots(out, audit, groups, matched):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'figure.dpi':140,'savefig.bbox':'tight'})
    fig,axs=plt.subplots(1,2,figsize=(11,4.2))
    for method in METHODS:
        rows=[r for r in audit['cases'] if r['method']==method and r['status']=='completed']
        axs[0].scatter([r['n'] for r in rows],[r['wall_seconds'] for r in rows],label=LABELS[method],
                       color=COLORS[method],s=30)
        measured=[r for r in rows if r.get('global_cp_error') is not None]
        bounds=[r for r in rows if r.get('global_cp_error') is None]
        axs[1].scatter([r['n'] for r in measured],[r['global_cp_error'] for r in measured],color=COLORS[method],s=30)
        axs[1].scatter([r['n'] for r in bounds],[r['global_cp_bound'] for r in bounds],color=COLORS[method],marker='^',s=45)
    for ax in axs:
        ax.set_xscale('log',base=2);ax.set_yscale('log');ax.set_xlabel('Spins N');ax.grid(alpha=.2)
    axs[0].set_ylabel('Process wall time (s)');axs[1].set_ylabel('Global CP error / conditional bound (triangles)')
    axs[0].legend(fontsize=8)
    fig.suptitle('Recovered Actions cohort: single observations; failed N=6 factor excluded')
    fig.tight_layout();fig.savefig(out/'actions_scaling.png');plt.close(fig)
    if not groups:
        return
    families=sorted({r['family'] for r in groups if r['repeat_complete']})
    if not families:
        return
    fig,axs=plt.subplots(1,len(families),figsize=(5*len(families),4.6),squeeze=False)
    markers={3:'o',4:'s',5:'^',6:'D',8:'v',12:'P',16:'X'}
    for ax,family in zip(axs[0],families):
        for method in METHODS:
            records=[r for r in groups if r['repeat_complete'] and r['family']==family and r['method']==method
                     and r['global_cp_error_max'] is not None]
            for r in records:
                ax.errorbar(r['global_cp_error_max'],r['total_seconds_median'],
                            yerr=[[r['total_seconds_median']-r['total_seconds_min']],
                                  [r['total_seconds_max']-r['total_seconds_median']]],fmt=markers.get(r['n'],'o'),color=COLORS[method],capsize=2)
            if records:
                ax.plot([],[],marker='o',linestyle='',color=COLORS[method],label=LABELS[method])
        ax.set(xscale='log',yscale='log',xlabel='Worst repeated global CP error',ylabel='Median worker time (s)',title=family)
        for n in sorted({r['n'] for r in groups if r['family']==family and r['repeat_complete'] and r['global_cp_error_max'] is not None}):
            ax.plot([],[],marker=markers.get(n,'o'),linestyle='',color='#777777',label=f'N={n}')
        ax.grid(alpha=.2);ax.legend(fontsize=7,ncol=2,loc='upper left')
    fig.suptitle('Local cohort: three repeats; bars show observed min–max, not confidence intervals')
    fig.tight_layout();fig.savefig(out/'local_error_runtime.png');plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(14,4.3))
    for ax,target in zip(axs,TARGETS):
        for method in METHODS:
            vals=[r for r in matched if r['family']=='D' and r['method']==method and r['target']==target and r['attained']]
            if vals:
                ax.plot([r['n'] for r in vals],[r['seconds'] for r in vals],color=COLORS[method],label=LABELS[method])
                for r in vals:
                    ax.errorbar(r['n'],r['seconds'],yerr=[[r['seconds']-r['minimum']],[r['maximum']-r['seconds']]],
                                marker='o',color=COLORS[method],capsize=3,
                                markerfacecolor=COLORS[method] if r['evidence']=='measured' else 'white')
        ax.set(yscale='log',xlabel='Spins N',ylabel='Median worker time (s)',title=f'D: error ≤ {target:g}')
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Local repeated matched accuracy: cheapest observed qualifying configuration\nFilled markers: measured global error; hollow markers: conditional global bound')
    fig.tight_layout();fig.savefig(out/'local_matched_scaling.png');plt.close(fig)
    # Same target and size: each bar includes the independently propagated
    # reference, metric, certification and scoring when those are needed.
    chosen=[r for r in matched if r['family']=='D' and r['n']==3 and r['target']==1e-7 and r['attained']]
    if chosen:
        lookup={g['config_id']:g for g in groups}
        buckets={
            'Factor tangent solve':['factor_tangent_force'],
            'Root predictor':['root_predictor_force','prediction_other'],
            'Witness fit':['witness_fit'],
            'Correction direction':['correction_direction'],
            'Certification':['certification'],
            'Auxiliary reference / metric':['reference_metric_initialization','physical_or_inverse_metric_propagation'],
            'Retraction / root construction':['factor_retraction','root_retraction','hermitian_retraction','factor_root_construction_and_normalization'],
            'Dense support audit':['dense_support_audit'],
            'Other, scoring, I/O':['other_worker_seconds','independent_reference_scoring','dense_model_initialization']}
        values={name:np.array([sum(lookup[r['config_id']]['exclusive_seconds_median'].get(k,0.) for k in keys) for r in chosen])
                for name,keys in buckets.items()}
        totals=sum(values.values())
        fig,axs=plt.subplots(1,2,figsize=(12,4.8))
        for ax,normalized in zip(axs,(False,True)):
            bottom=np.zeros(len(chosen))
            for name,raw in values.items():
                val=raw/totals*100 if normalized else raw
                ax.bar([LABELS[r['method']].replace(' ','\n',1) for r in chosen],val,bottom=bottom,label=name)
                bottom+=val
            ax.set_ylabel('Share of summed category medians (%)' if normalized else 'Median category time (s)')
        fig.suptitle('Local D, N=3, CP target 1e-7: exclusive measured costs')
        axs[1].legend(fontsize=8,loc='upper left',bbox_to_anchor=(1,1));fig.tight_layout()
        fig.savefig(out/'local_cost_breakdown.png');plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recovered',type=Path,default=HERE/'scaling_runs/actions_43745844867')
    parser.add_argument('--local',type=Path,default=HERE/'scaling_runs/local_20261115')
    parser.add_argument('--out',type=Path,default=HERE/'scaling_figures')
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    audit=audit_recovered(args.recovered)
    write_json(args.out/'recovery_audit.json',audit)
    save_csv(args.out/'recovered_cases.csv',audit['cases'])
    if audit['failures']:
        raise RuntimeError('Recovered artifact failed integrity audit: '+str(audit['failures']))
    raw=json.loads((args.recovered/'measurements.json').read_text())
    save_csv(args.out/'recovered_frontiers.csv',recovered_frontier(raw))
    local=args.local/'measurements.json'
    rows=json.loads(local.read_text()) if local.exists() else []
    local_audit=audit_local(args.local,rows)
    write_json(args.out/'local_audit.json',local_audit)
    if local_audit['failures']:
        raise RuntimeError('Local data consistency audit failed: '+str(local_audit['failures']))
    groups=summarize_local(rows)
    matched=frontier(groups)
    for name,data in (('local_groups',groups),('repeated_matched_accuracy',matched),
                      ('exploratory_matched_accuracy',frontier(groups,repeats=False))):
        write_json(args.out/(name+'.json'),data);save_csv(args.out/(name+'.csv'),data)
    plots(args.out,audit,groups,matched)
    tables(args.out,audit,groups,matched)
    print(json.dumps(dict(artifact_cases=len(audit['cases']),verified_files=audit['files_checked'],
                          local_rows=len(rows),local_groups=len(groups),
                          repeated_complete=sum(g['repeat_complete'] for g in groups))))


if __name__=='__main__':
    main()
