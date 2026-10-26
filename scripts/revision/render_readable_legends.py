#!/usr/bin/env python3
"""Enlarge legends using archived curves and baseline values; no model fitting."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import render_figures as archived


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--style', type=Path, required=True)
    parser.add_argument('--samples', type=Path, required=True)
    parser.add_argument('--baseline-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    code = Path(__file__).resolve().parents[2]
    data, out = args.data_root.resolve(), args.output.resolve()
    if out.exists() or out.is_relative_to(data):
        raise ValueError('Use a new output directory outside historical data.')
    inputs = [args.style.resolve(), args.samples.resolve(), args.baseline_manifest.resolve()]
    baselines = {}
    for tag, _, n in archived.SPEC:
        for kind, name in [('learning_curve',f'{tag}_anomalyaware_on_latent_off_n{n}.json'),
                           ('zero_shot',f'{tag}_anomalyaware_on.json'),
                           ('screening',f'policy_selection_diagnostic_{tag}_anomalyaware_on_latent_off_n{n}.json')]:
            inputs.append(data/'results'/kind/name)
        rows = json.loads((data/'results/zero_shot'/f'{tag}_anomalyaware_on.json').read_text())['stage_rows']
        baselines[tag] = next(r['rmse'] for r in rows if r['stage']=='Matched subset / anomaly-aware selection')
    inputs.append(data/'results/screening/oc20_dense_ocxhx_anomalyaware_on_latent_off_n2367.json')
    hashes = {str(p):archived.sha(p) for p in inputs}
    saved = json.loads(args.baseline_manifest.read_text())
    # Use already audited uncertainty baselines; do not recalculate diagnostics.
    uq_values = [saved['uq_baselines'][tag] for tag, _, _ in archived.SPEC]
    os.environ['OASIS_PLOT_STYLE_PATH'] = str(args.style.resolve())
    archived.reset_plot_style_cache()
    original_plot, original_save, original_legend = archived.p.oracle_learning_curve_plot, Figure.savefig, Axes.legend
    original_metric = archived.miscalibration_area
    reports = {}
    out.mkdir(parents=True)
    for phase, size in [('before',8),('after',12)]:
        folder = out/phase
        folder.mkdir()
        capture = {}
        metric_values = iter(uq_values)
        archived.miscalibration_area = lambda *a, **kw: next(metric_values)
        archived.p._DEFAULT_LEGEND_FONTSIZE = size
        def oracle(frame, **kw):
            frame = frame.copy()
            frame['zero_shot_rmse'] = frame.dataset.map(baselines)
            assert frame.zero_shot_rmse.notna().all()
            return original_plot(frame, **kw)
        def legend(ax, *a, **kw):
            result = original_legend(ax, *a, **kw)
            if phase == 'after':
                for text in result.get_texts():
                    text.set_text(text.get_text().replace('mean-MLIP RMSE','mean-MLIP\nRMSE').replace('mean-MLIP baseline','mean-MLIP\nbaseline').replace('Screening-selected method','Screening-selected\nmethod'))
            return result
        def save(fig, filename, *a, **kw):
            name = Path(filename).name
            if name not in ('accuracy.png','uq.png','comparison.png','screening.png'):
                ax = fig.axes[0]
                if name == 'aa.png':
                    ax.yaxis.labelpad = 0
                if name in ('ac.png','ad.png'):
                    arrows = [t for t in ax.texts if t.get_text() == '↑']
                    for i,t in enumerate(arrows):
                        t.set_x(i*0.04)
                    squares = [c for c in ax.collections if c.get_zorder()==5]
                    occupied=[]
                    for c in sorted(squares,key=lambda c:float(c.get_offsets()[0,1])):
                        y=float(c.get_offsets()[0,1]); lane=0
                        while any(abs(y-oy)<0.04 and lane==ol for oy,ol in occupied):
                            lane+=1
                        c.set_offsets([[lane*0.025,y]]); occupied.append((y,lane))
                    assert len(arrows)+len(squares)==(8 if name=='ac.png' else 7)
                capture[name] = archived.snapshot(fig)
                capture[name][0]['title'] = ax.get_title()
                capture[name][0]['legend_fontsizes'] = [] if ax.get_legend() is None else [t.get_fontsize() for t in ax.get_legend().get_texts()]
                # Layout only: preserve tick locations, but avoid label crowding.
                if phase=='after' and name in ('ab.png','ub.png'):
                    for label in ax.get_xticklabels()[1::2]:
                        label.set_visible(False)
            return original_save(fig, filename, *a, **kw)
        archived.p.oracle_learning_curve_plot, Figure.savefig, Axes.legend = oracle, save, legend
        try:
            for render in (archived.render_accuracy,archived.render_uq,archived.render_comparison,archived.render_screening):
                render(data/'results', folder, args.samples, {})
        finally:
            archived.p.oracle_learning_curve_plot, Figure.savefig, Axes.legend = original_plot, original_save, original_legend
            archived.miscalibration_area = original_metric
        reports[phase] = capture
    for name in reports['before']:
        old, new = reports['before'][name][0].copy(), reports['after'][name][0].copy()
        before_size, after_size = old.pop('legend_fontsizes'), new.pop('legend_fontsizes')
        assert all(v==8 for v in before_size), (name,before_size)
        assert all(v==12 for v in after_size), (name,after_size)
        assert old == new, name
    assert hashes == {str(p):archived.sha(p) for p in inputs}
    (out/'coordinate_checks.json').write_text(json.dumps(reports,indent=2)+'\n')
    sources=[Path(__file__),code/'scripts/revision/render_figures.py',code/'src/oasis/plot.py',code/'src/oasis/plot_style.py',code/'src/oasis/figure.py']
    manifest=dict(command=[sys.executable,*sys.argv],oasis_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=code,text=True).strip(),
        source_status=subprocess.check_output(['git','status','--short'],cwd=code,text=True),source_hashes={str(p.relative_to(code)):archived.sha(p) for p in sources},
        inputs=hashes,accuracy_baselines=baselines,uq_baselines=saved['uq_baselines'],
        validation=f'{len(reports["after"])} panels: curves, bands, axes, ticks, annotations, titles and normalized legend labels unchanged; legend font sizes 8 to 12.',
        display_changes='Legends enlarged; long labels wrapped; alternate x tick labels hidden in Figures 3b/4b with tick positions retained. Complete Figure 3 baselines preserved. Figure 4 titles preserved.',
        outputs={str(p.relative_to(out)):archived.sha(p) for p in sorted(out.rglob('*')) if p.is_file()})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(manifest['validation'])


if __name__=='__main__':
    main()
