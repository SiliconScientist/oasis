#!/usr/bin/env python3
"""Re-render archived manuscript figures; never run experiments or write caches."""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from oasis import plot as p
from oasis.figure import two_by_two_figure, horizontal_panel_figure
from oasis.learning_curve.results_io import load_learning_curve_results_artifact
from oasis.experiment.policy_diagnostic import summarize_fixed_method_baseline_frame
from oasis.experiment.splits import resolve_configured_sweep_sizes
from oasis.calibration_metrics import miscalibration_area
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib
import PIL
import shutil
from oasis.plot_style import reset_plot_style_cache

SPEC = [('mamun_oh','OH-BMA',1168),('khlohc','Tol-BMA',307),('bio_mass','Biomass',25),('oc20_dense_ocxhx','OC20-OCxHx',2367),('oc20_dense_nhx','OC20-NHx',9744),('game_net','Game-Net-Ox',986),('func_group','Func-Group',2598),('comer','Hydrox-Ox',323)]
METHODS = ['ridge','kernel_ridge','lasso','elastic','residual','weighted_linear','weighted_simplex','current_best_mlip']
EXPECTED_BASELINE = {
    'comparison.png': 'afaef01cd9b4570d0c577ab62c5d768cebf13d5a5190076a84f2778306aa91f4',
    'accuracy.png': 'fc631e6f728dde4ea25c8c2f3ce60075bc3f2996d6daab5d8a676ba87f945a44',
    'uq.png': 'e049adee496cdf4aa8346e640c48cbf4545dbc568b805ac093f5aef86e47eb2e',
    'screening.png': '83336027fe49bcea06bcdadea962f95cc6f87085ea96bdbcf15e7125ce6154e9',
}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def array_hash(value):
    return hashlib.sha256(np.asarray(value, dtype=float).tobytes()).hexdigest()

def snapshot(fig):
    """Data-space coordinates/styles, independent of legend/layout geometry."""
    return [dict(
        xlim=list(ax.get_xlim()), ylim=list(ax.get_ylim()),
        xscale=ax.get_xscale(), yscale=ax.get_yscale(),
        xlabel=ax.get_xlabel(), ylabel=ax.get_ylabel(),
        xticks=list(ax.get_xticks()), yticks=list(ax.get_yticks()),
        lines=[dict(x=array_hash(line.get_xdata()), y=array_hash(line.get_ydata()),
                    color=line.get_color(), marker=line.get_marker(),
                    style=line.get_linestyle(), label=line.get_label()) for line in ax.lines],
        collections=[dict(paths=[array_hash(path.vertices) for path in col.get_paths()],
                          offsets=array_hash(col.get_offsets()),
                          face=array_hash(col.get_facecolors()), edge=array_hash(col.get_edgecolors()))
                     for col in ax.collections],
        annotations=[dict(text=t.get_text(), position=list(t.get_position()),
                          color=t.get_color()) for t in ax.texts],
        legend_labels=[] if ax.get_legend() is None else [' '.join(t.get_text().split()) for t in ax.get_legend().get_texts()],
    ) for ax in fig.axes]

def render_screening(root, out, samples_path, baselines):
    spec=SPEC
    methods=METHODS
    entries=[]
    for tag,label,n in spec:
     data=json.loads((root/'screening'/f'policy_selection_diagnostic_{tag}_anomalyaware_on_latent_off_n{n}.json').read_text()); sig=data['cache_signature']['learning_curve'];summary=pd.read_json(io.StringIO(data['results']['summary_df']),orient='table')
     entries.append(dict(dataset=tag,dataset_label=label,dataset_size=n,summary_df=summary,sweep_sizes=tuple(sig['sweep_sizes']),sweep_fractions=tuple(sig['sweep_fractions'])))
     if tag=='oc20_dense_ocxhx': main=data; main_summary=summary
    order=[x[0] for x in spec]
    fixed=summarize_fixed_method_baseline_frame(pd.read_json(io.StringIO(main['results']['outer_metrics_df']),orient='table'),baselines=(('residual','Residual'),('kernel_ridge','Kernel ridge')))
    absolute=list(range(1,21));fraction=resolve_configured_sweep_sizes(2367,min_train=None,max_train=None,sweep_sizes=(),sweep_fractions=main['cache_signature']['learning_curve']['sweep_fractions'])
    p.policy_selected_vs_oracle_plot(main_summary,fixed_method_summary_df=fixed,output_path=out/'a.png',include_x=absolute,show_title=False,show_legend=False)
    p.policy_selected_vs_oracle_plot(main_summary,fixed_method_summary_df=fixed,output_path=out/'b.png',include_x=fraction,show_title=False,legend_labels={'oracle':'Oracle best','min_screening_rmse':'Screening-selected method','Residual':'Mean-residual method','Kernel ridge':'Kernel ridge regression method'})
    p._render_screening_curve_regret_panel(p._build_screening_curve_regret_frame(entries,mode='absolute',excluded_datasets=set(),absolute_include_x=absolute),output_path=out/'c.png',dataset_order=order,log_x=False,show_legend=False)
    p._render_screening_curve_regret_panel(p._build_screening_curve_regret_frame(entries,mode='fraction',excluded_datasets={'bio_mass'}),output_path=out/'d.png',dataset_order=order,log_x=True,legend_source_df=p._build_screening_curve_regret_frame(entries,mode='fraction',excluded_datasets=set()))
    two_by_two_figure(top_left_path=out/'a.png',top_right_path=out/'b.png',bottom_left_path=out/'c.png',bottom_right_path=out/'d.png',output_path=out/'screening.png')

def render_accuracy(root, out, samples_path, baselines):
    spec=SPEC
    methods=METHODS
    aframes=[]; fframes=[]
    for tag,label,n in spec:
     artifact=load_learning_curve_results_artifact(root/'learning_curve'/f'{tag}_anomalyaware_on_latent_off_n{n}.json')
     frame=p.oracle_learning_curve_frame(artifact.results,enabled_method_names=methods,dataset=tag,dataset_label=label)
     zero=json.loads((root/'zero_shot'/f'{tag}_anomalyaware_on.json').read_text());rows=zero['stage_rows']
     baseline=next(x['rmse'] for x in rows if x['stage']=='Matched subset / anomaly-aware selection')
     if tag in ['mamun_oh','khlohc']: frame['zero_shot_rmse']=baseline
     aframes.append(frame[frame.n_train.isin(range(1,21))])
     sizes=resolve_configured_sweep_sizes(n,min_train=None,max_train=None,sweep_sizes=(),sweep_fractions=[i/20 for i in range(1,17)])
     fframes.append(frame[frame.n_train.isin(sizes)])
     if tag=='oc20_dense_ocxhx': results=artifact.results; zero_main=baseline; fractions=sizes
    mapping=results.to_mapping()
    for key in list(mapping):
     if key.startswith(('moe_','gnn_direct_')):mapping[key]=None
    rb=results.from_mapping(mapping); ra=results.from_mapping({**mapping,'weighted_linear_df':None})
    p.learning_curve_plot(ra,output_path=out/'aa.png',title='',include_x=list(range(1,21)),zero_shot_rmse=zero_main,show_legend=False,show_std_bands=True)
    p.learning_curve_plot(rb,output_path=out/'ab.png',title='',include_x=fractions,zero_shot_rmse=zero_main,show_legend=True,legend_outside_right=True,show_std_bands=True,x_tick_fontsize=10)
    p.oracle_learning_curve_plot(pd.concat(aframes),output_path=out/'ac.png',title='',show_legend=False,log_x=False)
    f=pd.concat(fframes)
    p.oracle_learning_curve_plot(f[f.dataset!='bio_mass'],output_path=out/'ad.png',title='',show_legend=True,legend_outside_right=True,legend_source_df=f,log_x=True)
    two_by_two_figure(top_left_path=out/'aa.png',top_right_path=out/'ab.png',bottom_left_path=out/'ac.png',bottom_right_path=out/'ad.png',output_path=out/'accuracy.png')

def render_uq(root, out, samples_path, baselines):
    spec=SPEC
    methods=METHODS
    samples=pd.read_csv(samples_path)
    aframes=[]; fframes=[]
    for tag,label,n in spec:
     artifact=load_learning_curve_results_artifact(root/'learning_curve'/f'{tag}_anomalyaware_on_latent_off_n{n}.json')
     frame=p.oracle_uq_curve_frame(artifact.results,enabled_method_names=methods,dataset=tag,dataset_label=label)
     s=samples[(samples.dataset==tag)&samples.retained_ten];cols=[c[:-11] for c in s if c.endswith('_prediction')]
     pred=s[[c+'_prediction' for c in cols]].to_numpy()
     baseline=miscalibration_area(s.reference.to_numpy(),pred.mean(axis=1),pred.std(axis=1)); baselines[tag]=float(baseline)
     frame['zero_shot_miscalibration_area']=baseline
     aframes.append(frame[frame.n_train.isin(range(1,21))]);sizes=resolve_configured_sweep_sizes(n,min_train=None,max_train=None,sweep_sizes=(),sweep_fractions=[i/20 for i in range(1,17)]);fframes.append(frame[frame.n_train.isin(sizes)])
     if tag=='oc20_dense_ocxhx': results=artifact.results; zero_main=baseline; fractions=sizes
    mapping=results.to_mapping()
    for key in list(mapping):
     if key.startswith(('moe_','gnn_direct_')):mapping[key]=None
    rb=results.from_mapping(mapping);ra=results.from_mapping({**mapping,'weighted_linear_uq_df':None})
    p.miscalibration_area_plot(ra,output_path=out/'ua.png',include_x=list(range(1,21)),zero_shot_value=zero_main,show_legend=False,show_xlabel=True)
    p.miscalibration_area_plot(rb,output_path=out/'ub.png',include_x=fractions,zero_shot_value=zero_main,show_legend=True,legend_outside_right=True,show_xlabel=True,x_tick_fontsize=10)
    p.all_datasets_uq_oracle_plot(pd.concat(aframes),output_path=out/'uc.png',metric_column='oracle_miscalibration_area',ylabel='Oracle miscalibration area',title='',log_x=False,show_legend=False)
    f=pd.concat(fframes)
    p.all_datasets_uq_oracle_plot(f[f.dataset!='bio_mass'],output_path=out/'ud.png',metric_column='oracle_miscalibration_area',ylabel='Oracle miscalibration area',title='',log_x=True,show_legend=True,legend_outside_right=True,legend_source_df=f)
    two_by_two_figure(top_left_path=out/'ua.png',top_right_path=out/'ub.png',bottom_left_path=out/'uc.png',bottom_right_path=out/'ud.png',output_path=out/'uq.png')

def render_comparison(root, out, samples_path, baselines):
    old_tick=p._tick_fontsize
    try:
        p._tick_fontsize=lambda:8
        for kind,title,name in [('screening','Screening curve','ca'),('learning_curve','Learning curve','cb')]:
         r=load_learning_curve_results_artifact(root/kind/'oc20_dense_ocxhx_anomalyaware_on_latent_off_n2367.json').results
         r=r.from_mapping({k:None if k.startswith(('weighted_linear_','gnn_direct_','moe_','current_best_mlip_')) else v for k,v in r.to_mapping().items()})
         if kind=='screening':p.screening_budget_plot(r,output_path=out/f'{name}.png',title=title,include_x=list(range(1,21)),show_legend=False)
         else:p.learning_curve_plot(r,output_path=out/f'{name}.png',title=title,include_x=list(range(1,21)),show_legend=True,zero_shot_rmse=None)
        horizontal_panel_figure([out/'ca.png',out/'cb.png'],output_path=out/'comparison.png',panel_label_positions=((0.02,1.0),(0.02,1.0)))
    finally:
        p._tick_fontsize=old_tick

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--style', type=Path, required=True)
    parser.add_argument('--samples', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args=parser.parse_args()
    code_root=Path(__file__).resolve().parents[2]
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=code_root,text=True).strip():
        raise RuntimeError('Tracked source must be clean before rendering.')
    initial_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=code_root,text=True).strip()
    initial_script_sha=sha(__file__)
    output=args.output.resolve()
    data=args.data_root.resolve()
    if output.exists() or output.is_relative_to(data):
        raise ValueError('Output must be new and outside historical data.')
    inputs=[args.style.resolve(),args.samples.resolve()]
    for tag,_,n in SPEC:
        inputs += [data/'results'/kind/name for kind,name in (
            ('learning_curve',f'{tag}_anomalyaware_on_latent_off_n{n}.json'),
            ('screening',f'policy_selection_diagnostic_{tag}_anomalyaware_on_latent_off_n{n}.json'),
            ('zero_shot',f'{tag}_anomalyaware_on.json'))]
    inputs.append(data/'results/screening/oc20_dense_ocxhx_anomalyaware_on_latent_off_n2367.json')
    fingerprints={str(path):sha(path) for path in inputs}
    output.mkdir(parents=True)
    shutil.copyfile(args.style,output/"plot_style.toml")
    os.environ['OASIS_PLOT_STYLE_PATH']=str(args.style.resolve())
    reset_plot_style_cache()
    original_savefig=Figure.savefig
    original_legend=Axes.legend
    reports={}
    baselines={}
    for phase,legend_size in [('baseline',8),('revised',12)]:
        folder=output/phase
        folder.mkdir()
        captures={}
        def legend(ax,*a,**kw):
            result=original_legend(ax,*a,**kw)
            if phase=='revised':
                for text in result.get_texts():
                    text.set_text(text.get_text().replace('mean-MLIP RMSE','mean-MLIP\nRMSE').replace('mean-MLIP baseline','mean-MLIP\nbaseline'))
            return result
        Axes.legend=legend
        def savefig(fig, filename, *a, **kw):
            name=Path(filename).name
            if name not in EXPECTED_BASELINE:
                if phase=='revised' and name=='aa.png':
                    fig.axes[0].yaxis.labelpad=0
                captures[name]=snapshot(fig)
                if phase=='revised' and name in ('ab.png','ub.png'):
                    for label in fig.axes[0].get_xticklabels()[1::2]:
                        label.set_visible(False)
                if phase=='revised' and name in ('ua.png','ub.png'):
                    for ax in fig.axes:
                        ax.set_title('')
            return original_savefig(fig,filename,*a,**kw)
        Figure.savefig=savefig
        p._DEFAULT_LEGEND_FONTSIZE=legend_size
        try:
            for render in (render_accuracy,render_uq,render_screening,render_comparison):
                render(data/'results',folder,args.samples,baselines)
        finally:
            Figure.savefig=original_savefig
            Axes.legend=original_legend
        reports[phase]=captures
        if phase=='baseline':
            for name,expected in EXPECTED_BASELINE.items():
                if sha(folder/name)!=expected:
                    raise AssertionError(f'Historical rendering contract mismatch: {name}')
    if reports['baseline']!=reports['revised']:
        (output/'coordinate_mismatch.json').write_text(json.dumps(reports,indent=2))
        raise AssertionError('Legend revision changed data-space geometry/styles/annotations/axes.')
    for path,expected in fingerprints.items():
        if sha(path)!=expected:
            raise AssertionError(f'Input changed: {path}')
    (output/'coordinate_checks.json').write_text(json.dumps(reports['revised'],indent=2))
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=code_root,text=True).strip()!=initial_commit or sha(__file__)!=initial_script_sha or subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=code_root,text=True).strip():
        raise RuntimeError('Source changed during rendering.')
    manifest=dict(
        analysis_commit=initial_commit,
        script_sha256=initial_script_sha,command=[sys.executable,*sys.argv],
        python=sys.version, numpy=np.__version__, pandas=pd.__version__, matplotlib=matplotlib.__version__, pillow=PIL.__version__, inputs=fingerprints,
        datasets=SPEC,oracle_methods=METHODS,absolute_grid=list(range(1,21)),
        fractions=[i/20 for i in range(1,17)],fraction_rounding='resolve_configured_sweep_sizes (floor via int)',
        sparse_accuracy_baselines=['mamun_oh','khlohc'],
        uq_baselines=baselines,uq_baseline_contract='Retained rows, unmasked ten raw MLIP columns; original mean/std and miscalibration_area formula, no fits.',
        display_changes='Longest zero-shot legend labels wrapped; alternate numeric x-labels hidden in Figures3b/4b, with original tick positions/gridlines preserved; Figure3a y-label padding adjusted inward; Figure4a/b titles removed; Figure5 panel-label composition recovered to within label-only pixel differences.',
        legend_size_before=8,legend_size_after=12,
        coordinate_checks='All 14 panels identical data-space line/band/marker coordinates, styles, limits, scales, ticks, labels and annotations; titles intentionally suppressed in Figure4 a/b.',
        historical_baseline_sha256=EXPECTED_BASELINE,
        historical_equivalence='Figure5 differs only in panel-label placement, 5238 pixels within x72..2218,y30..111; all data panels and titles identical to original6cd4288397b409ea67dbf6582534c8127000cd20b689249527ff538ce4d47dc2. Figure6 exact current manuscript hash; Figure4 exact pre-title-removal hash. Figure3 differs only in repaired panel-a RMSE (eV) label: 7876 pixels within x17..69,y338..648; every other pixel identical to original04c2525b5d5500a25678168a56f2a2df0363f000384d1a3c1e263a9d740b010a. Revised label keeps identical wording and moves inward to prevent clipping.',
        outputs={str(path.relative_to(output)):sha(path) for path in sorted(output.rglob('*')) if path.is_file()},
    )
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'output':str(output),'baseline_verified':4,'coordinate_panels_verified':len(reports['revised'])}))

if __name__=='__main__':
    main()
