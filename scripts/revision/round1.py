"""Read-only reconstruction of round-one zero-shot evidence from saved results.

Never imports experiment orchestration or writes caches. A fresh output directory
is mandatory. Cache stage counts/RMSE and all ten model identities must agree
before an ablation is emitted. No fitting, random sampling or relaxation occurs.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TAGS = ('mamun_oh','khlohc','bio_mass','oc20_dense_ocxhx','oc20_dense_nhx','game_net','func_group','comer')
EXCLUDE = ('allegro-oam-l-0.1','aqcat25-ev2','grace-2l-omat-medium-ft-am')
EXPECTED_MODELS = ('7net-omni','allegro-oam-l-0.1','alphanet','aqcat25-ev2','chgnet-r2scan','grace-2l-omat-medium-ft-am','mace-mh-1','mattersim-v1-5m','orb-v3-conservative-inf-omat','uma-s-1p1')
FLAGS = ('slab_conv','ads_conv','slab_move','ads_move','slab_seed','ads_seed','ads_eng_seed','adsorbate_migration')

def validate_models(models):
    if len(models)!=10 or set(models)!=set(EXPECTED_MODELS):
        raise ValueError('expected exactly ten unique manuscript models')

def validate_output(data_root,output):
    if output.resolve().is_relative_to(data_root.resolve()):
        raise ValueError('output must be outside historical data tree')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def percent_reduction(before, after):
    return None if before == 0 else 100*(before-after)/before

def error_summary(pred, target):
    residual=np.asarray(pred)-np.asarray(target)
    return dict(rmse=float(np.sqrt(np.mean(residual**2))),bias=float(np.mean(residual)),centered_rmse=float(np.std(residual)))

def compare(x,y,valid,reduced):
    x=np.asarray(x); y=np.asarray(y); valid=np.asarray(valid,dtype=bool)
    if x.ndim!=2 or valid.shape!=x.shape or y.shape!=(len(x),):raise ValueError('shape mismatch')
    if not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('nonfinite input')
    if len(set(reduced))!=len(reduced) or len(reduced)<2:raise ValueError('invalid reduced model set')
    all_mask=valid.sum(1)>=2; reduced_mask=valid[:,reduced].sum(1)>=2
    common=all_mask & reduced_mask
    if not common.any():raise ValueError('empty common population')
    out=dict(n_full=len(y),n_retained_all=int(all_mask.sum()),n_retained_reduced=int(reduced_mask.sum()),n_common=int(common.sum()))
    for name,cols in [('all',list(range(x.shape[1]))),('reduced',reduced)]:
        a=x[common][:,cols]; v=valid[common][:,cols]
        raw=a.mean(1); filtered=np.where(v,a,0).sum(1)/v.sum(1)
        for mode,pred in [('unfiltered',raw),('filtered',filtered)]:
            out[f'{name}_{mode}_rmse']=error_summary(pred,y[common])['rmse']
        out[f'{name}_filtering_reduction_percent']=percent_reduction(out[f'{name}_unfiltered_rmse'],out[f'{name}_filtered_rmse'])
    for mode in ['unfiltered','filtered']:
        before=out[f'all_{mode}_rmse'];after=out[f'reduced_{mode}_rmse']
        out[f'exclusion_{mode}_rmse_difference']=after-before
        out[f'exclusion_{mode}_reduction_percent']=percent_reduction(before,after)
    return out

def write_csv(path, rows):
    if not rows:
        path.write_text("dataset,model,rmse\n");return
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def run(data_root,output,tags=TAGS):
    validate_output(data_root,output)
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True).strip():
        raise ValueError('commit all tracked source changes before analysis')
    sys.path[:0]=[str(ROOT/'src'),str(ROOT/'vendor/catbench')]
    from oasis.mlip.result_parsing import detect_anomalies_from_result_dict
    from catbench.utils.analysis_utils import get_calculator_keys
    output.mkdir(parents=True,exist_ok=False)
    inputs={}; summaries=[];ablation=[];anomalies=[];bias=[];offscale=[];samples=[];datasets=[]
    def record(p):
        p=p.resolve();inputs[str(p)]=dict(sha256=sha(p),size=p.stat().st_size);return p
    config=record(data_root.parent/'experiment.toml')
    for tag in tags:
        cache=record(data_root/'results/zero_shot'/f'{tag}_anomalyaware_on.json')
        cached=json.loads(cache.read_text());sig=cached['cache_signature']
        if sig['minimum_quorum']!=2 or not sig['exclude_anomalous_mlips']:raise ValueError('unexpected cache policy')
        parsed=[];models=[];run_info=[]
        for item in sig['result_files']:
            path=record(data_root.parent/item['path'])
            if path.stat().st_size!=item['size']:raise ValueError(f'cache source size changed: {path}')
            raw=json.loads(path.read_text()); models.append(path.name.removesuffix('_result.json'))
            parsed.append(detect_anomalies_from_result_dict(raw))
            repeats=Counter(len(get_calculator_keys(v)) for k,v in raw.items() if k!='calculation_settings')
            run_info.append(dict(path=str(path),settings=raw.get('calculation_settings'),repeat_counts=dict(repeats),mtime_matches_signature=path.stat().st_mtime_ns==item['mtime_ns']))
        validate_models(models)
        ids=sorted(set.intersection(*(set(d) for d in parsed)))
        ids=[i for i in ids if all(d[i]['dft_ads_eng'] is not None and d[i]['mlip_ads_eng_median'] is not None for d in parsed)]
        y=np.array([parsed[0][i]['dft_ads_eng'] for i in ids],float)
        for d in parsed:
            if not np.array_equal(y,np.array([d[i]['dft_ads_eng'] for i in ids],float)):raise ValueError('reference mismatch')
        x=np.array([[d[i]['mlip_ads_eng_median'] for d in parsed] for i in ids],float)
        valid=np.array([[not any(d[i]['details'][f] for f in FLAGS) for d in parsed] for i in ids])
        if not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('nonfinite historical values')
        retained=valid.sum(1)>=2
        vals=[error_summary(x.mean(1),y)['rmse'],error_summary(x[retained].mean(1),y[retained])['rmse'],error_summary(np.where(valid[retained],x[retained],0).sum(1)/valid[retained].sum(1),y[retained])['rmse']]
        stages=('Full / all MLIPs','Matched subset / all MLIPs','Matched subset / anomaly-aware selection')
        for stage,value,n in zip(stages,vals,[len(ids),int(retained.sum()),int(retained.sum())]):
            row=next(r for r in cached['stage_rows'] if r['stage']==stage and 'mlip' not in r)
            if row['n_samples']!=n or not np.isclose(value,row['rmse'],rtol=0,atol=1e-10):raise ValueError(f'{tag} does not reproduce {stage}: {value}, {row}')
        summary=dict(dataset=tag,n_full=len(ids),n_retained=int(retained.sum()),excluded_percent=100*(1-retained.mean()),full_rmse=vals[0],matched_all_rmse=vals[1],matched_valid_rmse=vals[2],full_comparison_percent=percent_reduction(vals[0],vals[2]),matched_filtering_percent=percent_reduction(vals[1],vals[2]))
        summaries.append(summary)
        reduced=[i for i,m in enumerate(models) if m not in EXCLUDE]
        comparison=compare(x,y,valid,reduced);ablation.append(dict(dataset=tag,**comparison));common=retained & (valid[:,reduced].sum(1)>=2)
        for j,model in enumerate(models):
            entry=dict(dataset=tag,model=model,n_full=len(ids),invalid=int((~valid[:,j]).sum()))
            for f in FLAGS:entry[f]=sum(bool(parsed[j][i]['details'][f]) for i in ids)
            for category in ['normal','energy_anomaly','adsorbate_migration','unphysical_relaxation','reproduction_failure']:
                entry['exclusive_'+category]=sum(parsed[j][i]['label']==category for i in ids)
            anomalies.append(entry)
            for mode,mask in [('common_unfiltered',common),('common_model_valid',common & valid[:,j])]:
                if mask.any():bias.append(dict(dataset=tag,model=model,population=mode,n=int(mask.sum()),**error_summary(x[mask,j],y[mask])))
            single=error_summary(x[:,j],y)['rmse']
            saved=next(r for r in cached['stage_rows'] if r.get('mlip')==model)
            if not np.isclose(single,saved['rmse'],rtol=0,atol=1e-10):raise ValueError('single-model cache mismatch')
            if single>3:offscale.append(dict(dataset=tag,model=model,rmse=single))
        for i,identity in enumerate(ids):
            row=dict(dataset=tag,sample_id=identity,reference=y[i],retained_ten=bool(retained[i]),retained_seven=bool(common[i]))
            for j,model in enumerate(models):row[model+'_prediction']=x[i,j];row[model+'_valid']=bool(valid[i,j])
            samples.append(row)
        bundle=record(data_root/'results/learning_curve'/f'{tag}_anomalyaware_on_latent_off_n{int(retained.sum())}.json')
        b=json.loads(bundle.read_text());meta=b['metadata']
        if meta['dataset_tag']!=tag or set(meta['mlip_feature_names'])!=set(models):raise ValueError('bundle metadata mismatch')
        alternatives=[]
        for other in sorted(bundle.parent.glob(f'{tag}_anomalyaware_on_latent_off_n*.json')):
            if 'repeat_metrics' in other.name:continue
            record(other);alternatives.append(dict(path=str(other),metadata=json.loads(other.read_text()).get('metadata')))
        datasets.append(dict(dataset=tag,models=models,n_full=len(ids),n_retained=int(retained.sum()),source_cache=str(cache),learning_curve=str(bundle),alternative_metadata=alternatives,raw_results=run_info))
        print(tag,'verified cache; common ablation n=',comparison['n_common'],flush=True)
    for name,rows in [('filtering',summaries),('ablation',ablation),('anomaly_counts',anomalies),('descriptive_errors',bias),('offscale',offscale),('samples',samples)]:write_csv(output/f'{name}.csv',rows)
    manifest=dict(analysis_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),script_sha256=sha(Path(__file__)),catbench_tree=subprocess.check_output(['git','rev-parse','HEAD:vendor/catbench'],cwd=ROOT,text=True).strip(),command=sys.argv,python=sys.version,numpy=np.__version__,configuration=str(config),thresholds=dict(displacement_angstrom=1.0,reproduction_ev=0.2,bond_change_fraction=0.2,energy_anomaly_ev=2.0),threshold_evidence='Pinned parser defaults, no overrides in load_wide_predictions; reconstructed stages match saved cache within 1e-10. Not proof of original relaxation software identity.',population_contract='Intersection of complete finite ten-model rows with >=2 inference-valid predictions in each model set; independent operational coverage on the same complete-row universe.',random_sampling='None; deterministic saved-result reanalysis.',excluded_models=EXCLUDE,inputs=inputs,datasets=datasets,outputs={p.name:sha(p) for p in output.glob('*.csv')})
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--datasets',nargs='+',choices=TAGS,default=TAGS);args=p.parse_args();run(args.data_root.resolve(),args.output.resolve(),args.datasets)
