#!/usr/bin/env python3
"""Complete Figure 3 baseline annotations from saved results, without fitting."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import render_figures as archived
from matplotlib.figure import Figure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--style', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    data, output = args.data_root.resolve(), args.output.resolve()
    if output.exists() or output.is_relative_to(data):
        raise ValueError('Use a new output directory outside historical data.')
    inputs = [args.style.resolve()]
    values = {}
    for tag, _, n in archived.SPEC:
        zero = data/'results/zero_shot'/f'{tag}_anomalyaware_on.json'
        inputs += [zero, data/'results/learning_curve'/f'{tag}_anomalyaware_on_latent_off_n{n}.json']
        rows = json.loads(zero.read_text())['stage_rows']
        matches = [r['rmse'] for r in rows if r['stage'] == 'Matched subset / anomaly-aware selection']
        if len(matches) != 1:
            raise ValueError(f'Ambiguous baseline for {tag}')
        values[tag] = float(matches[0])
    hashes = {str(p): archived.sha(p) for p in inputs}
    sources = [Path(__file__), root/'scripts/revision/render_figures.py',
               root/'src/oasis/plot.py', root/'src/oasis/figure.py', root/'src/oasis/plot_style.py']
    source_hashes = {str(p.relative_to(root)): archived.sha(p) for p in sources}
    os.environ['OASIS_PLOT_STYLE_PATH'] = str(args.style.resolve())
    archived.reset_plot_style_cache()
    archived.p._DEFAULT_LEGEND_FONTSIZE = 8
    output.mkdir(parents=True)
    original_plot = archived.p.oracle_learning_curve_plot
    original_save = Figure.savefig
    reports, annotation_reports = {}, {}
    for phase in ('sparse', 'complete'):
        folder = output/phase
        folder.mkdir()
        capture, annotations = {}, {}

        def plot(frame, **kwargs):
            frame = frame.copy()
            if phase == 'complete':
                frame['zero_shot_rmse'] = frame.dataset.map(values)
                if frame.zero_shot_rmse.isna().any():
                    raise ValueError('Missing dataset baseline')
            return original_plot(frame, **kwargs)

        def save(fig, filename, *a, **kw):
            name = Path(filename).name
            if name in ('aa.png', 'ab.png', 'ac.png', 'ad.png'):
                ax = fig.axes[0]
                if name == 'aa.png':
                    ax.yaxis.labelpad = 0  # Preserve the manuscript's unclipped y-axis label.
                if phase == 'complete' and name in ('ac.png', 'ad.png'):
                    # Separate off-scale arrows along the upper margin.
                    arrows = [t for t in ax.texts if t.get_text() == '↑']
                    for i, text in enumerate(arrows):
                        text.set_x(i * 0.04)
                    # Stagger squares only when nearby baselines would overlap.
                    squares = [c for c in ax.collections if c.get_zorder() == 5]
                    occupied = []
                    for square in sorted(squares, key=lambda c: float(c.get_offsets()[0, 1])):
                        y = float(square.get_offsets()[0, 1])
                        lane = 0
                        while any(abs(y-old_y) < 0.04 and lane == old_lane for old_y, old_lane in occupied):
                            lane += 1
                        square.set_offsets([[lane * 0.025, y]])
                        occupied.append((y, lane))
                    annotations[name] = dict(squares=len(squares), arrows=len(arrows))
                    assert len(squares) + len(arrows) == (8 if name == 'ac.png' else 7)
                capture[name] = archived.snapshot(fig)
            return original_save(fig, filename, *a, **kw)
        archived.p.oracle_learning_curve_plot = plot
        Figure.savefig = save
        try:
            archived.render_accuracy(data/'results', folder, None, {})
        finally:
            archived.p.oracle_learning_curve_plot = original_plot
            Figure.savefig = original_save
        reports[phase], annotation_reports[phase] = capture, annotations
    # Only baseline scatters and arrow annotations may differ.
    for name in ('aa.png', 'ab.png', 'ac.png', 'ad.png'):
        old, new = reports['sparse'][name][0], reports['complete'][name][0]
        for key in ('xlim', 'ylim', 'xscale', 'yscale', 'xlabel', 'ylabel', 'xticks', 'yticks', 'lines', 'legend_labels'):
            assert old[key] == new[key], (name, key)
        if name in ('aa.png', 'ab.png'):
            assert old == new, name
        else:
            # Historical oracle plots have one square in c and none in d.
            n_old = 1 if name == 'ac.png' else 0
            n_new = annotation_reports['complete'][name]['squares']
            old_bands = old['collections'][:-n_old] if n_old else old['collections']
            new_bands = new['collections'][:-n_new] if n_new else new['collections']
            assert old_bands == new_bands, (name, 'non-baseline collections')
    assert hashes == {str(p): archived.sha(p) for p in inputs}
    assert source_hashes == {str(p.relative_to(root)): archived.sha(p) for p in sources}
    (output/'coordinate_checks.json').write_text(json.dumps(reports, indent=2)+'\n')
    manifest = dict(command=[sys.executable, *sys.argv],
        oasis_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        source_hashes=source_hashes,
        source_status=subprocess.check_output(['git','status','--short'],cwd=root,text=True),
        inputs=hashes, baselines=values, annotations=annotation_reports['complete'],
        validation='All four panels retain identical curves, bands, axes, ticks, and legend labels. Inputs unchanged.',
        outputs={str(p.relative_to(output)):archived.sha(p) for p in sorted(output.rglob('*')) if p.is_file()})
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(baselines=values,annotations=annotation_reports['complete'],output=str(output)),indent=2))


if __name__ == '__main__':
    main()
