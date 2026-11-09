"""Sequential matched-runtime controls. All settings are selected without exact errors."""
from pathlib import Path
import csv
import handoff_trial as trial


DATA = Path(__file__).parent / 'handoff_results'


def load_summary(name):
    with (DATA / (name + '.csv')).open() as stream:
        return next(csv.DictReader(stream))


def fixed(n, chi, steps, buffer, label):
    summary, history = trial.run(n, chi, 'fixed', steps, 1e-4, 4, .1, buffer, 24 if n==6 else 16)
    trial.observables.save(DATA / (label + '.csv'), [summary])
    trial.observables.save(DATA / (label + '_history.csv'), history)
    return summary


def main():
    rows = []
    for n in (4, 6):
        force_chi = 16 if n == 4 else 24
        base = f'n{n}_c16_f{force_chi}_T0.1_s10_b2_tol0.0001_a4_r0'
        hybrid = load_summary('mpo_' + base)
        budget = float(hybrid['seconds'])
        for buffer in (2, 3):
            prefix = f'cost_n{n}_b{buffer}'
            pilot = fixed(n, 16, 20, buffer, prefix + '_pilot')
            steps = max(10, round(20 * .94 * budget / pilot['seconds']))
            actual = fixed(n, 16, steps, buffer, prefix + '_run')
            row = dict(hybrid='mpo_' + base, wall_budget=budget, **actual)
            row['time_ratio'] = actual['seconds'] / budget
            rows.append(row)
            trial.observables.save(DATA/'matched_cost.csv', rows)
            print(row, flush=True)


if __name__=='__main__':
    main()
