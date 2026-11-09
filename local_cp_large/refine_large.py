"""End-to-end initialization refinement for the uniform mixed-field benchmark."""
import copy
from pathlib import Path
import argparse
import local_cp_large as study


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir', type=Path, default=Path(__file__).parent/'refinement_results')
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    histories = []
    rows = []
    for tau in (.025, .0125):
        initial = study.thermal_cp(64, .8, order=4, tau_step=tau, chi=24)
        state = copy.deepcopy(initial)
        engine = study.LocalEvolution(state, 2, 1, lambda a,b: study.previous.local_density(initial,a,b))
        history = {0: (study.doubled_marginal(state,30,2), study.previous.local_density(state,30,32))}
        for step in range(1,11):
            engine.sweep(.1, 'Z')
            if step in (2,5,10):
                history[step] = (study.doubled_marginal(state,30,2), study.previous.local_density(state,30,32))
        histories.append(history)
        print('completed tau', tau, flush=True)
    for step in histories[0]:
        a,b = histories[0][step],histories[1][step]
        row = dict(n=64,order=4,coarse_tau=.025,fine_tau=.0125,sweep=step,
                   local_cp_output_change=study.trace_distance(a[0],b[0]),
                   physical_output_change=study.trace_distance(a[1],b[1]))
        rows.append(row)
        print(row, flush=True)
    study.csv_write(args.outdir/'refinement.csv',rows)


if __name__ == '__main__':
    main()
