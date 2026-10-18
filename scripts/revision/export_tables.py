"""Format audited CSV evidence as manuscript tables; no scientific recomputation."""
import argparse,csv
from pathlib import Path
LABELS={'mamun_oh':'OH-BMA','khlohc':'Tol-BMA','bio_mass':'Biomass','oc20_dense_ocxhx':'OC20-OCxHx','oc20_dense_nhx':'OC20-NHx','game_net':'Game-Net-Ox','func_group':'Func-Group','comer':'Hydrox-Ox'}
MODELS={'7net-omni':'SevenNet','allegro-oam-l-0.1':'Allegro','alphanet':'AlphaNet','aqcat25-ev2':'AQCat25','chgnet-r2scan':'CHGNet','grace-2l-omat-medium-ft-am':'GRACE','mace-mh-1':'MACE','mattersim-v1-5m':'MatterSim','orb-v3-conservative-inf-omat':'ORB-V3','uma-s-1p1':'UMA'}
def read(root,name):return list(csv.DictReader((root/(name+'.csv')).open()))
def table(caption,label,columns,header,rows,alt):
 return '\n'.join(['% Alt text: '+alt,r'\begin{table}[ht]',r'\centering',r'\caption{'+caption+'}',r'\label{'+label+'}',r'\small',r'\begin{tabular}{'+columns+'}',r'\toprule',header+r' \\',r'\midrule',*[r+' \\' for r in rows],r'\bottomrule',r'\end{tabular}',r'\end{table}',''])
def export(root,out):
 if out.exists():raise ValueError('refusing to overwrite table artifact')
 f=read(root,'filtering');a=read(root,'ablation');counts=read(root,'anomaly_counts')
 text=r'''\section{Filtering and Model-Exclusion Comparisons}
\label{sec:si-revision-comparisons}

The full population contains structures with reference energies and predictions from all ten MLIPs. The retained population additionally requires at least two inference-valid predictions. Filtering improvements compare the all-model mean and the mean of valid predictions on identical retained structure IDs. Missing predictions are distinct from predictions marked anomalous. Negative improvement denotes an increase in RMSE. All RMSE values are in eV.

'''
 text+=table('Full and retained population counts, excluded fractions, and matched-population filtering comparisons. $R_0$ is the full-population unfiltered RMSE; $R_1$ and $R_2$ are the retained-population unfiltered and filtered RMSEs. Improvement is $100(R_1-R_2)/R_1$.','tab:si-matched-filtering','lrrrrrrr',r'Dataset & Full & Retained & Excluded (\%) & $R_0$ & $R_1$ & $R_2$ & Gain (\%)',[f"{LABELS[r['dataset']]} & {r['n_full']} & {r['n_retained']} & {float(r['excluded_percent']):.2f} & {float(r['full_rmse']):.3f} & {float(r['matched_all_rmse']):.3f} & {float(r['matched_valid_rmse']):.3f} & {float(r['matched_filtering_percent']):.2f}" for r in f],'Counts and RMSEs for eight datasets; filtering improves matched-population RMSE for seven datasets and worsens it for Tol-BMA.')
 text+=r'''
For the model-exclusion comparison, the reduced set omits Allegro, AQCat25, and GRACE. Both sets are evaluated on the intersection of structures satisfying the minimum-two-valid-model criterion for each set. The seven-model retained population is a subset of the ten-model retained population. The comparison therefore holds the evaluated structures fixed; the separate retained counts describe operational coverage. No fitting or use of test labels for bias correction is involved.

'''
 text+=table('Ten-model versus seven-model zero-shot ensemble RMSE on the common population. $N_{10}$ and $N_7$ are operational retained counts; $N_7$ is also the common-population size. U and F denote unfiltered and inference-filtered means.','tab:si-model-exclusion','lrrrrrr',r'Dataset & $N_{10}$ & $N_7$ & 10 U & 10 F & 7 U & 7 F',[f"{LABELS[r['dataset']]} & {r['n_retained_all']} & {r['n_common']} & {float(r['all_unfiltered_rmse']):.3f} & {float(r['all_filtered_rmse']):.3f} & {float(r['reduced_unfiltered_rmse']):.3f} & {float(r['reduced_filtered_rmse']):.3f}" for r in a],'Paired zero-shot errors for eight datasets: excluding three models improves filtered RMSE in five datasets and worsens it in three.')
 text+=r'''
Table~\ref{tab:si-anomaly-counts} counts each model's inference-invalid predictions once per structure on the full complete-prediction population. The archived analysis tables additionally distinguish the eight overlapping detail flags from the mutually exclusive category labels assigned by detector precedence. DFT-dependent energy-anomaly labels do not enter the inference-validity mask.

'''
 header='MLIP & '+ ' & '.join(str(i+1) for i in range(8))
 rows=[]
 for m,label in MODELS.items():rows.append(label+' & '+' & '.join(next(r['invalid'] for r in counts if r['model']==m and r['dataset']==tag) for tag in LABELS))
 text+=table('Inference-invalid predictions by MLIP and dataset. Columns 1--8 follow the dataset order in Table~\\ref{tab:si-matched-filtering}; denominators are the full-population counts in that table.','tab:si-anomaly-counts','lrrrrrrrr',header,rows,'Counts of inference-invalid predictions for ten MLIPs across all eight benchmark datasets.')
 out.write_text(text)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('input',type=Path);p.add_argument('output',type=Path);a=p.parse_args();export(a.input,a.output)
