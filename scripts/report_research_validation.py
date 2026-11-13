"""Report only the complete27-job research validation matrix.

Reads completed inference/analysis artifacts. It never starts GPU work and never
substitutes forecasts for missing jobs. Existing exploratory analyses stay clearly
separate from the corrected-split and multiple-training-seed experiments.
"""
from __future__ import annotations
import argparse,csv,json,sys
from pathlib import Path
from functools import lru_cache
from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from core import digest_file,dump_json,digest_json
DATASETS=['gc10_det','neu_det','pcb_ind_board']
TITLES={'gc10_det':'GC10-DET','neu_det':'NEU-DET','pcb_ind_board':'PCB-IND 숫자 ID 분할'}
SEEDS=[11,22,33]
CONDITIONS=['bright_k25','dark_k25','stuck_k25','bright_k1','bright_k9','bright_partial25','bright_cluster25']
LABELS={'bright_k25':'밝은25·분산','dark_k25':'어두운25·분산','stuck_k25':'128고정25·분산','bright_k1':'밝은1·분산','bright_k9':'밝은9·분산','bright_partial25':'부분강도25·분산','bright_cluster25':'밝은25·5×5군집'}
METRICS={'recall':'recall','precision':'precision','map50':'coco_map50','map50_95':'coco_map50_95'}
COND_SPECS={
 'bright_k25':(255,1.,25,'dispersed',1), 'dark_k25':(0,1.,25,'dispersed',1),
 'stuck_k25':(128,1.,25,'dispersed',1), 'bright_k1':(255,1.,1,'dispersed',1),
 'bright_k9':(255,1.,9,'dispersed',1), 'bright_partial25':(255,.5,25,'dispersed',1),
 'bright_cluster25':(255,1.,25,'cluster',5)}
FONT=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams['axes.unicode_minus']=False


def resolve(path):
 p=Path(path)
 return p if p.is_absolute() else ROOT/p


def read_json(path):
 return json.loads(Path(path).read_text(encoding='utf-8-sig'))


@lru_cache(maxsize=64)
def file_digest(path):return digest_file(path)


def validate_manifest(manifest,state):
 jobs=manifest.get('jobs')
 if not isinstance(jobs,list):raise ValueError('evaluation_manifest must contain jobs:list')
 wanted={(d,s,c) for d in DATASETS for s in SEEDS for c in (CONDITIONS if s==11 else ['bright_k25'])}
 observed=[(j.get('dataset'),j.get('seed'),j.get('condition')) for j in jobs]
 if len(jobs)!=27 or len(set(observed))!=27 or set(observed)!=wanted:raise ValueError('Expected complete27-job dataset/seed/condition matrix')
 keys=[j.get('key') for j in jobs]
 if any(not isinstance(k,str) or not k for k in keys) or len(set(keys))!=27:raise ValueError('Each job needs a unique key')
 statuses=state.get('jobs',state)
 if not isinstance(statuses,dict):raise ValueError('State must map job keys to status records')
 incomplete=[j['key'] for j in jobs if not isinstance(statuses.get(j['key']),dict) or statuses[j['key']].get('status')!='complete']
 if incomplete:raise ValueError('Not all jobs complete: '+', '.join(incomplete[:8]))
 for job in jobs:
  record=statuses[job['key']]
  if 'run_path' in record and 'run_path' in job and resolve(record['run_path']).resolve()!=resolve(job['run_path']).resolve():raise ValueError('Completed state refers to another run path')
 return jobs


def validate_grid_rows(m,condition,n_images,n_gt):
 required={'repeat','cell','condition_id','n_images','n_gt','fp_per_image',*METRICS.values()}
 if not required<=set(m):raise ValueError('Missing metric columns: '+str(required-set(m)))
 if len(m)!=640 or set(map(tuple,m[['repeat','cell']].to_numpy()))!={(r,b) for r in range(10) for b in range(64)}:raise ValueError('Need exactly64cells×10patterns with no duplicates')
 if not m.condition_id.eq(condition).all():raise ValueError('Mixed condition IDs')
 if not m.n_images.eq(n_images).all() or not m.n_gt.eq(n_gt).all():raise ValueError('Image/GT census differs across cases')
 values=m[list(METRICS.values())].to_numpy(float)
 if not np.isfinite(values).all() or (values<0).any() or (values>1+1e-10).any():raise ValueError('Undefined or out-of-range metrics cannot silently be averaged')


def summarize_metrics(m,base):
 patterns=[]
 for rep,g in m.groupby('repeat',sort=True):
  r={'pattern_id':int(rep)+1,'n_locations':len(g)}
  for label,key in METRICS.items():
   r[label+'_pct']=float(g[key].mean()*100);r[label+'_drop_pp']=float((base[key]-g[key].mean())*100)
  r['fp_per_image']=float(g.fp_per_image.mean());patterns.append(r)
 p=pd.DataFrame(patterns);result={}
 for label,key in METRICS.items():
  result['clean_'+label+'_pct']=float(base[key]*100);result['mean_'+label+'_pct']=float(m[key].mean()*100);result[label+'_drop_pp']=float((base[key]-m[key].mean())*100)
  if not np.isclose(result['mean_'+label+'_pct'],p[label+'_pct'].mean(),atol=1e-10,rtol=0):raise ValueError('Equal pattern averaging differs from all-case mean')
 result['pattern_sd_recall_drop_pp']=float(p.recall_drop_pp.std(ddof=1));result['mean_fp_per_image']=float(m.fp_per_image.mean())
 return result,patterns


def validate_config(cfg,condition):
 if cfg.get('grid')!={'rows':8,'cols':8} or cfg.get('repeats')!=10:raise ValueError('Expected8x8grid and10masks')
 cs=cfg.get('conditions',[])
 if len(cs)!=1 or cs[0].get('id')!=condition:raise ValueError('Run must contain its declared single condition')
 c=cs[0];actual=tuple(c.get(k) for k in ['target','alpha','count','layout','side'])
 if actual!=COND_SPECS[condition] or c.get('channels','all')!='all':raise ValueError('Condition does not match fixed matched-budget design')


@lru_cache(maxsize=8)
def validation_ids(dataset):
 with (ROOT/f'data/canonical/{dataset}/manifest.jsonl').open() as f:
  return frozenset(r['image_id'] for r in map(json.loads,f) if r['split']=='val')


def validate_checkpoint_control(meta,job,control,script_sha,config_sha,valid_ids):
 source=meta['source_sha256'];compiled='run_compiled.py' in source
 expected={'dataset':job['dataset'],'pattern_key':meta['pattern_key'],'weights_sha256':meta['weights_sha256'],
  'manifest_sha256':meta['manifest_sha256'],'classes_sha256':meta['classes_sha256'],
  'condition_config_sha256':config_sha,'control_script_sha256':script_sha,'source_sha256':source}
 if not control.get('all_pass') or control.get('fingerprints')!=expected:raise ValueError('Checkpoint control has stale model/data/source fingerprints')
 runner='run_compiled.py' if compiled else 'run_fast.py'
 if control.get('runner_sha256')!=source[runner] or control.get('weights_sha256')!=meta['weights_sha256']:raise ValueError('Checkpoint control identifies another model/runner')
 if compiled and control.get('nms_sha256')!=source['safe_compiled_nms.py']:raise ValueError('Checkpoint NMS source differs')
 ids=control.get('selected_validation_ids',[])
 if len(ids)!=17 or len(set(ids))!=17 or not set(ids)<=set(valid_ids):raise ValueError('Checkpoint controls must use17unique validation images')
 checks=control.get('checks',[])
 if len(checks)!=8 or {c.get('condition') for c in checks}!={'clean',*CONDITIONS}:raise ValueError('Checkpoint control coverage is incomplete')
 if not all(c.get('exact_predictions') and c.get('n_images')==17 for c in checks):raise ValueError('Checkpoint controls do not preserve every prediction')
 return {'n_validation_images':17,'n_condition_checks':8}


def load_job(job,verify_prediction_sha=False):
 root=resolve(job['run_path']);a=root/'analysis';meta=read_json(root/'run_meta.json');base=read_json(a/'clean_metrics.json');ameta=read_json(a/'analysis_meta.json')
 validate_config(meta['config'],job['condition'])
 if ameta.get('run_hash')!=meta['run_hash']:raise ValueError('Analysis is from another run: '+job['key'])
 if meta.get('batch')!=16:raise ValueError('Validated batch16 must be preserved')
 if meta.get('pattern_key')!=('pcb_ind' if job['dataset']=='pcb_ind_board' else job['dataset']):raise ValueError('Pattern namespace differs from fixed dataset masks')
 if base['n_images']<=0 or base['n_gt']<=0 or len(meta['selected_ids'])!=base['n_images'] or len(set(meta['selected_ids']))!=base['n_images']:raise ValueError('Invalid selected image census')
 for key,name in [('manifest_sha256','manifest.jsonl'),('classes_sha256','classes.json')]:
  if file_digest(str(ROOT/f'data/canonical/{job["dataset"]}'/name))!=meta[key]:raise ValueError('Canonical data/class file changed since evaluation')
 weights=resolve(job['weights_path'])
 if weights.resolve()!=Path(meta['weights_path']).resolve() or file_digest(str(weights))!=meta['weights_sha256']:raise ValueError('Weights differ from declared job')
 train_dir=weights.parent.parent;train_args_path=train_dir/'args.yaml';train_args=read_json_or_yaml(train_args_path)
 if int(train_args.get('seed',-1))!=int(job['seed']):raise ValueError('Training seed differs from declared evaluation job')
 if resolve(train_args['data']).resolve()!=(ROOT/f'data/canonical/{job["dataset"]}/dataset.yaml').resolve():raise ValueError('Model was trained on another dataset split')
 initial_weights=resolve(train_args['model'])
 recipe_keys=['epochs','patience','batch','imgsz','optimizer','lr0','lrf','mosaic','mixup','copy_paste','degrees','translate','scale','shear','perspective','flipud','fliplr','hsv_h','hsv_s','hsv_v','deterministic','amp']
 recipe_hash=digest_json({k:train_args.get(k) for k in recipe_keys});initial_hash=file_digest(str(initial_weights))
 training_results_path=train_dir/'results.csv';n_training_epochs=len(pd.read_csv(training_results_path))
 if n_training_epochs<1:raise ValueError('Training results are empty')
 if 'config_path' in job and read_json_or_yaml(resolve(job['config_path']))!=meta['config']:raise ValueError('Config file differs from frozen run')
 checkpoint_info={};checkpoint_path=None
 if 'model_control_path' in job:
  checkpoint_path=resolve(job['model_control_path']);control=read_json(checkpoint_path);script='validate_compiled_model_control.py' if 'run_compiled.py' in meta['source_sha256'] else 'validate_fast_model_control.py'
  checkpoint_info=validate_checkpoint_control(meta,job,control,file_digest(str(ROOT/'scripts'/script)),file_digest(str(ROOT/'configs/research_validation/all_conditions.yaml')),validation_ids(job['dataset']))
 elif 'run_compiled.py' in meta['source_sha256']:raise ValueError('Compiled job manifest lacks per-checkpoint control certificate')
 m=pd.read_csv(a/'condition_metrics.csv');s=pd.read_csv(a/'grid_summary.csv').sort_values('cell')
 validate_grid_rows(m,job['condition'],base['n_images'],base['n_gt'])
 if len(s)!=64 or s.cell.tolist()!=list(range(64)) or not s.condition_id.eq(job['condition']).all():raise ValueError('Incomplete location summary')
 for b,g in m.groupby('cell'):
  direct=(base['recall']-g.recall.mean())*100;stored=float(s.loc[s.cell==b,'sensitivity_pp'].iloc[0])
  if not np.isclose(direct,stored,atol=1e-10,rtol=0):raise ValueError('Location summary differs from per-case metrics')
 expected={'clean'}|{f'{job["condition"]}__r{r:03d}__g{b:03d}' for r in range(10) for b in range(64)}
 marker_paths=list((root/'predictions').glob('*.done.json'));actual={p.name.removesuffix('.done.json') for p in marker_paths}
 if actual!=expected:raise ValueError('Incomplete641prediction shards for '+job['key'])
 n_eval=n_new=n_reused=0;seconds=0.;devices=set();nms_counts={}
 for p in marker_paths:
  v=read_json(p);prediction=p.with_name(p.name.removesuffix('.done.json')+'.jsonl')
  if v['run_hash']!=meta['run_hash'] or v['n_images']!=base['n_images'] or not prediction.is_file():raise ValueError('Invalid prediction completion record')
  if verify_prediction_sha and digest_file(prediction)!=v['sha256']:raise ValueError('Prediction SHA mismatch: '+str(prediction))
  n_eval+=int(v['n_images']);seconds+=float(v.get('seconds_wall',0));devices.add(str(v.get('worker_device','unknown')))
  if 'run_compiled.py' in meta['source_sha256'] and 'nms_stats' not in v:raise ValueError('Compiled completion marker lacks NMS counters')
  for key,value in v.get('nms_stats',{}).items():
   if int(value)!=value or value<0:raise ValueError('Invalid per-shard NMS count')
   nms_counts[key]=nms_counts.get(key,0)+int(value)
  if 'cache_reuse' in v:n_reused+=int(v['n_images'])
  else:n_new+=int(v['n_images'])
 pairs=a/'paired_images.csv'
 # Count independently of quoted content: this CSV contains no multiline fields.
 with pairs.open('rb') as f:pair_lines=sum(block.count(b'\n') for block in iter(lambda:f.read(1<<20),b''))-1
 if pair_lines!=640*base['n_images'] or list(a.glob('*.partial')):raise ValueError('Paired-image output is partial')
 result,patterns=summarize_metrics(m,base);peak=s.loc[s.sensitivity_pp.idxmax()]
 row={'key':job['key'],'dataset':job['dataset'],'dataset_title':TITLES[job['dataset']],'seed':int(job['seed']),'condition':job['condition'],
  'n_test':int(base['n_images']),'n_gt':int(base['n_gt']),'n_training_epochs':n_training_epochs,'initial_weights_sha256':initial_hash,'training_recipe_sha256':recipe_hash,**result,'max_location':str(peak.grid_label),'max_location_drop_pp':float(peak.sensitivity_pp),
  'mean_actual_changed_pixels':float(s.mean_actual_changed_pixels.mean()),'weights_sha256':meta['weights_sha256'],'manifest_sha256':meta['manifest_sha256'],
  'classes_sha256':meta['classes_sha256'],'selected_ids_sha256':digest_json(meta['selected_ids']),'pattern_key':meta['pattern_key'],
  'checkpoint_control_path':str(checkpoint_path) if checkpoint_path is not None else '',**checkpoint_info,'execution_backend':meta.get('backend','unspecified'),'n_prediction_shards':len(marker_paths),'nms_calls':nms_counts.get('calls',0) if 'run_compiled.py' in meta['source_sha256'] else None,'nms_compiled_calls':nms_counts.get('compiled_rank_preserved',0) if 'run_compiled.py' in meta['source_sha256'] else None,'nms_near_threshold_fallback_calls':nms_counts.get('near_threshold',0) if 'run_compiled.py' in meta['source_sha256'] else None,'nms_guard_counts':json.dumps(nms_counts,sort_keys=True),
  'logical_image_evaluations':n_eval,'new_image_evaluations':n_new,'reused_image_evaluations':n_reused,'pair_rows':pair_lines,'shard_seconds_sum':seconds,
  'worker_devices':','.join(sorted(devices)),'run_path':str(root),'run_hash':meta['run_hash']}
 for p in patterns:p.update(dataset=job['dataset'],seed=int(job['seed']),condition=job['condition'],key=job['key'])
 provenance={'run_meta_sha256':digest_file(root/'run_meta.json'),'condition_metrics_sha256':digest_file(a/'condition_metrics.csv'),
  'training_args_sha256':digest_file(train_args_path),'training_results_sha256':digest_file(training_results_path),'grid_summary_sha256':digest_file(a/'grid_summary.csv'),'analysis_meta_sha256':digest_file(a/'analysis_meta.json'),'paired_images_sha256':digest_file(pairs),'checkpoint_control_sha256':digest_file(checkpoint_path) if checkpoint_path is not None else None}
 return row,patterns,s,meta,base,provenance


def read_json_or_yaml(path):
 import yaml
 return yaml.safe_load(Path(path).read_text())


def validate_backend_certificate(meta,certificate):
 if not certificate.get('all_pass') or certificate.get('selected_batch')!=16 or not certificate.get('checks'):
  raise ValueError('Validated batch16 backend certificate is missing')
 for c in certificate['checks']:
  if not (c.get('accepted') and c.get('exact_predictions') and c.get('per_image_matches_equal') and c.get('ap_max_abs_difference',float('inf'))<=1e-6):
   raise ValueError('Backend certificate failed exact-prediction controls')
 source=meta['source_sha256'];compiled='run_compiled.py' in source
 if compiled:
  if meta.get('backend')!='bgr-packed-safe-compiled-nms-fp32-bs16-v4':raise ValueError('Unsupported compiled backend identity')
  if source.get('run_compiled.py')!=certificate.get('runner_sha256') or source.get('safe_compiled_nms.py')!=certificate.get('nms_sha256'):
   raise ValueError('Compiled runner or NMS module differs from certificate')
  reference=certificate.get('reference_runner_sha256')
  if reference not in {source.get('run.py'),source.get('run_fast.py')}:raise ValueError('Compiled certificate references another eager runner')
  if 'reference_fast_runner_sha256' in certificate and source.get('run_fast.py')!=certificate['reference_fast_runner_sha256']:
   raise ValueError('Compiled certificate fast-reference fingerprint differs')
 else:
  if source.get('run_fast.py')!=certificate.get('runner_sha256'):raise ValueError('Fast runner differs from certificate')
  if certificate.get('reference_runner_sha256')!=source.get('run.py'):raise ValueError('Fast reference runner differs from certificate')
 return 'compiled' if compiled else 'fast'


def validate_model_consistency(loaded):
 models={};datasets={};logical=None;backend_sources={};backend_versions=None;training_identity=None
 for job,row,patterns,grid,meta,base,prov in loaded:
  current_training=(row['initial_weights_sha256'],row['training_recipe_sha256'])
  if training_identity is not None and current_training!=training_identity:raise ValueError('Initial checkpoint or training recipe differs across models')
  training_identity=current_training
  model_key=(job['dataset'],int(job['seed']))
  fingerprint=(meta['weights_sha256'],meta['manifest_sha256'],meta['classes_sha256'],tuple(meta['selected_ids']),meta['pattern_key'])
  if model_key in models:
   prev_fp,prev_base=models[model_key]
   if fingerprint!=prev_fp:raise ValueError('Mixed model/data within dataset/seed')
   for k in ['n_images','n_gt','tp','fp','fn',*METRICS.values()]:
    if not np.isclose(base[k],prev_base[k],atol=1e-10,rtol=0):raise ValueError('Clean baseline differs across conditions')
  else:models[model_key]=(fingerprint,base)
  data_fp=(meta['manifest_sha256'],meta['classes_sha256'],tuple(meta['selected_ids']),meta['pattern_key'],base['n_gt'])
  if job['dataset'] in datasets and data_fp!=datasets[job['dataset']]:raise ValueError('Training seeds use different test/data cohorts')
  datasets[job['dataset']]=data_fp
  cfg={k:v for k,v in meta['config'].items() if k!='conditions'}
  if logical is not None and cfg!=logical:raise ValueError('Shared evaluation parameters differ across jobs')
  logical=cfg
  backend=meta.get('backend','unspecified')
  if backend in backend_sources and meta['source_sha256']!=backend_sources[backend]:raise ValueError('Inference source versions differ within the same backend')
  if backend_versions is not None and meta['versions']!=backend_versions:raise ValueError('Package versions differ across jobs')
  backend_sources[backend]=meta['source_sha256'];backend_versions=meta['versions']
 return models


def seed_summaries(condition_rows,grids):
 full=pd.DataFrame(condition_rows);bright=full[full.condition=='bright_k25'].sort_values(['dataset','seed']);summaries=[];corrs=[]
 for d in DATASETS:
  ds=bright[bright.dataset==d]
  if ds.seed.tolist()!=SEEDS:raise ValueError('Need3independent bright-condition models')
  r={'dataset':d,'n_training_seeds':3,'n_patterns_per_seed':10,'n_locations_per_pattern':64}
  for c in ['mean_recall_pct','recall_drop_pp','mean_precision_pct','precision_drop_pp','mean_map50_pct','map50_drop_pp','mean_map50_95_pct','map50_95_drop_pp']:
   r[c+'_seed_mean']=float(ds[c].mean());r[c+'_seed_sd']=float(ds[c].std(ddof=1))
  r['peak_cells']=';'.join(f"s{int(x.seed)}:{x.max_location}" for x in ds.itertuples());summaries.append(r)
  for sa,sb in combinations(SEEDS,2):
   ga=grids[d,sa,'bright_k25'].sort_values('cell');gb=grids[d,sb,'bright_k25'].sort_values('cell');xa=ga.sensitivity_pp.to_numpy();xb=gb.sensitivity_pp.to_numpy()
   pearson=float(np.corrcoef(xa,xb)[0,1]) if np.ptp(xa)>0 and np.ptp(xb)>0 else np.nan
   spearman=float(pd.Series(xa).corr(pd.Series(xb),method='spearman')) if np.ptp(xa)>0 and np.ptp(xb)>0 else np.nan
   ta=set(ga.sort_values(['sensitivity_pp','cell'],ascending=[False,True]).head(4).cell);tb=set(gb.sort_values(['sensitivity_pp','cell'],ascending=[False,True]).head(4).cell)
   corrs.append({'dataset':d,'seed_a':sa,'seed_b':sb,'pearson_spatial_map':pearson,'spearman_spatial_map':spearman,
    'same_peak_cell':int(ga.loc[ga.sensitivity_pp.idxmax(),'cell'])==int(gb.loc[gb.sensitivity_pp.idxmax(),'cell']),
    'top4_cell_jaccard':len(ta&tb)/len(ta|tb),'top4_seed_a':','.join(f'G{b+1}' for b in sorted(ta)),'top4_seed_b':','.join(f'G{b+1}' for b in sorted(tb))})
 return bright,pd.DataFrame(summaries),pd.DataFrame(corrs)


def matched_contrasts(patterns):
 ps=pd.DataFrame(patterns);rows=[]
 definitions=[('count9_minus1','화소수9−1','bright_k9','bright_k1'),('count25_minus9','화소수25−9','bright_k25','bright_k9'),
  ('count25_minus1','화소수25−1','bright_k25','bright_k1'),('dark_minus_bright','어두움−밝음','dark_k25','bright_k25'),
  ('stuck_minus_bright','128고정−밝음','stuck_k25','bright_k25'),('full_minus_partial','완전−부분강도','bright_k25','bright_partial25'),
  ('cluster_minus_dispersed','군집−분산','bright_cluster25','bright_k25')]
 for d in DATASETS:
  p=ps[(ps.dataset==d)&(ps.seed==11)]
  for slug,label,left,right in definitions:
   x=p[p.condition==left].sort_values('pattern_id');y=p[p.condition==right].sort_values('pattern_id')
   if x.pattern_id.tolist()!=list(range(1,11)) or y.pattern_id.tolist()!=list(range(1,11)):raise ValueError('Missing matched patterns')
   r={'dataset':d,'seed':11,'comparison':slug,'comparison_label':label,'lhs_condition':left,'rhs_condition':right,'n_patterns':10}
   for metric in METRICS:
    delta=x[metric+'_drop_pp'].to_numpy()-y[metric+'_drop_pp'].to_numpy()
    r[metric+'_drop_contrast_pp']=float(delta.mean());r[metric+'_contrast_pattern_sd_pp']=float(delta.std(ddof=1))
   rows.append(r)
 return pd.DataFrame(rows)


def old_pcb_comparison(condition_rows):
 oldroot=ROOT/'runs/evaluation/pcb_ind_s11_bright25_g8_p10';base=read_json(oldroot/'analysis/clean_metrics.json');m=pd.read_csv(oldroot/'analysis/condition_metrics.csv');r,_=summarize_metrics(m,base)
 old={'cohort':'기존 공식 분할','dataset':'pcb_ind','seed':11,'n_test':base['n_images'],'n_gt':base['n_gt'],**r,'run_path':str(oldroot)}
 new=next(row for row in condition_rows if row['dataset']=='pcb_ind_board' and row['seed']==11 and row['condition']=='bright_k25')
 return pd.DataFrame([old,{'cohort':'숫자 ID 분할',**new}])


def md_table(rows,columns):
 text='| '+' | '.join(title for _,title,_ in columns)+' |\n| '+' | '.join('---' for _ in columns)+' |\n'
 for row in rows:
  fields=[]
  for key,title,fmt in columns:
   v=row[key]
   fields.append(fmt(v) if callable(fmt) else format(v,fmt) if fmt else str(v))
  text+='| '+' | '.join(fields)+' |\n'
 return text


def analysis_inputs(base_dir,distribution_dir,positions_dir,corrected_distribution,corrected_positions):
 result={}
 for key,path in [('distribution',distribution_dir),('positions',positions_dir),('corrected_distribution',corrected_distribution),('corrected_positions',corrected_positions)]:
  if path is None:continue
  p=resolve(path)
  if not (p/'REPORT_KO.md').is_file():
   if key.startswith('corrected'):continue
   raise ValueError('Required existing-cache analysis missing: '+str(p))
  csvname='dataset_summary.csv' if 'distribution' in key else 'position_effect_summary.csv'
  if not (p/csvname).is_file():raise ValueError('Analysis summary missing: '+str(p/csvname))
  result[key]={'path':p,'frame':pd.read_csv(p/csvname),'report_sha256':digest_file(p/'REPORT_KO.md'),'summary_sha256':digest_file(p/csvname)}
 return result


def figures(out,conditions,bright,seed_summary,grids,patterns):
 colors=['#4677a9','#ef9b39','#bd4855'];x=np.arange(len(CONDITIONS));width=.25
 fig,ax=plt.subplots(figsize=(17,7))
 for i,d in enumerate(DATASETS):
  vals=conditions[(conditions.dataset==d)&(conditions.seed==11)].set_index('condition').loc[CONDITIONS]
  bars=ax.bar(x+(i-1)*width,vals.recall_drop_pp,width,label=TITLES[d],color=colors[i]);ax.bar_label(bars,fmt='%.2f',fontsize=8,padding=3)
 ax.axhline(0,color='#5f6974',lw=.7);ax.set_xticks(x,[LABELS[c] for c in CONDITIONS],fontproperties=FONT,fontsize=10);ax.set_ylabel('평균 Recall 감소 (%p)',fontproperties=FONT);ax.set_title('조건별 전체64위치·10패턴 평균 — 학습 시드11',fontproperties=FONT,fontsize=18);ax.legend(prop=FONT);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);fig.tight_layout();fig.savefig(out/'rq3_conditions_recall_drop.png',dpi=190);plt.close(fig)
 fig,axes=plt.subplots(1,3,figsize=(16,5),sharey=True)
 for ax,d,color in zip(axes,DATASETS,colors):
  a=bright[bright.dataset==d].sort_values('seed');bars=ax.bar([str(s) for s in a.seed],a.recall_drop_pp,color=color);ax.bar_label(bars,fmt='%.3f',padding=3,fontsize=10);ax.axhline(0,color='#5f6974',lw=.7);ax.set_title(TITLES[d],fontproperties=FONT);ax.set_xlabel('학습 시드',fontproperties=FONT);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
 axes[0].set_ylabel('평균 Recall 감소 (%p)',fontproperties=FONT);fig.suptitle('동일 핫픽셀25개·분산 조건의 학습 시드 반복',fontproperties=FONT,fontsize=18);fig.tight_layout();fig.savefig(out/'training_seed_recall_drop.png',dpi=190);plt.close(fig)
 fig,axes=plt.subplots(3,3,figsize=(14,13));lim=max(.01,float(np.ceil(max(abs(grids[d,s,'bright_k25'].sensitivity_pp).max() for d in DATASETS for s in SEEDS)*2)/2))
 for i,d in enumerate(DATASETS):
  for j,seed in enumerate(SEEDS):
   ax=axes[i,j];ar=grids[d,seed,'bright_k25'].sensitivity_pp.to_numpy().reshape(8,8);im=ax.imshow(ar,cmap='RdBu_r',vmin=-lim,vmax=lim)
   for y in range(8):
    for xx in range(8):ax.text(xx,y,f'{ar[y,xx]:.1f}',ha='center',va='center',fontsize=6.5,color='white' if abs(ar[y,xx])>lim*.55 else '#17212b')
   ax.set_title(f'{TITLES[d]} | seed{seed}',fontproperties=FONT,fontsize=11);ax.set_xticks([]);ax.set_yticks([])
 fig.subplots_adjust(left=.04,right=.89,top=.92,bottom=.04,wspace=.12,hspace=.17);cb=fig.colorbar(im,cax=fig.add_axes([.92,.19,.015,.6]));cb.set_label('Recall 감소 (%p), 공통 색상 범위',fontproperties=FONT);fig.suptitle('밝은 화소25개 — 3학습 시드의 위치별 평균 지도',fontproperties=FONT,fontsize=20);fig.savefig(out/'training_seed_spatial_maps.png',dpi=190);plt.close(fig)


def main(argv=None):
 parser=argparse.ArgumentParser();base=ROOT/'reports/research_validation_2026-11-11'
 parser.add_argument('--manifest',default=str(base/'evaluation_manifest.json'));parser.add_argument('--state',default=str(base/'evaluation_state.json'));parser.add_argument('--out',default=str(base/'summary'))
 parser.add_argument('--distribution',default=str(base/'distribution'));parser.add_argument('--positions',default=str(base/'positions'))
 parser.add_argument('--corrected-distribution',default=str(base/'corrected_pcb_distribution'));parser.add_argument('--corrected-positions',default=str(base/'corrected_pcb_positions'))
 parser.add_argument('--verify-prediction-sha',action='store_true',help='Rehash every prediction JSONL; original analyzer already checks them')
 args=parser.parse_args(argv);manifest_path=resolve(args.manifest);state_path=resolve(args.state);manifest=read_json(manifest_path);state=read_json(state_path);jobs=validate_manifest(manifest,state)
 jobs=sorted(jobs,key=lambda j:(DATASETS.index(j['dataset']),int(j['seed']),CONDITIONS.index(j['condition'])))
 loaded=[]
 for job in jobs:
  for field in ['run_path','config_path','weights_path']:
   if field not in job:raise ValueError('Manifest job missing '+field)
  row,patterns,grid,meta,clean,provenance=load_job(job,args.verify_prediction_sha)
  state_record=state.get('jobs',state)[job['key']]
  for name,expected_sha in state_record.get('analysis_output_sha256',{}).items():
   if digest_file(resolve(job['run_path'])/'analysis'/name)!=expected_sha:raise ValueError('Analysis output differs from completed-state fingerprint')
  row['legacy_fullcase_max_abs_difference']=state_record.get('original_fullcase_max_abs_difference')
  loaded.append((job,row,patterns,grid,meta,clean,provenance))
 validate_model_consistency(loaded)
 certificates={}
 for *_,meta,clean,prov in loaded:
  kind='compiled' if 'run_compiled.py' in meta['source_sha256'] else 'fast'
  if kind not in certificates:certificates[kind]=read_json(base/('compiled_backend_validation.json' if kind=='compiled' else 'fast_backend_validation.json'))
  validate_backend_certificate(meta,certificates[kind])
 analysis=analysis_inputs(base,args.distribution,args.positions,args.corrected_distribution,args.corrected_positions)
 split=read_json(base/'pcb_split/split_audit.json')
 if split['new_overlap']['n_groups_crossing_splits']!=0:raise ValueError('Corrected PCB split still has numeric-group overlap')
 condition_rows=[x[1] for x in loaded];pattern_rows=[p for x in loaded for p in x[2]];grids={(x[0]['dataset'],int(x[0]['seed']),x[0]['condition']):x[3] for x in loaded}
 conditions=pd.DataFrame(condition_rows);bright,seeds,corrs=seed_summaries(condition_rows,grids);contrasts=matched_contrasts(pattern_rows);pcb=old_pcb_comparison(condition_rows)
 baselines=conditions.sort_values(['dataset','seed','condition']).drop_duplicates(['dataset','seed'])[['dataset','seed','n_test','n_gt','n_training_epochs',*[f'clean_{m}_pct' for m in METRICS],'weights_sha256','manifest_sha256']]
 out=resolve(args.out);out.mkdir(parents=True,exist_ok=True)
 for name,frame in [('condition_means',conditions),('pattern_means',pd.DataFrame(pattern_rows)),('clean_baselines',baselines),('bright25_seed_results',bright),('bright25_seed_summary',seeds),('seed_spatial_correlations',corrs),('matched_budget_contrasts',contrasts),('pcb_official_vs_numeric_group',pcb)]:frame.to_csv(out/(name+'.csv'),index=False)
 figures(out,conditions,bright,seeds,grids,pd.DataFrame(pattern_rows))
 # Assemble a concise Korean report; all numerical tables originate from completed jobs.
 btable=md_table(baselines.to_dict('records'),[('dataset','데이터셋',lambda x:TITLES[x]),('seed','시드','d'),('n_training_epochs','완료 epoch','d'),('n_test','시험 영상','d'),('n_gt','GT','d'),('clean_recall_pct','원본 Recall (%)','.3f'),('clean_precision_pct','원본 Precision (%)','.3f'),('clean_map50_pct','원본 mAP50 (%)','.3f'),('clean_map50_95_pct','원본 mAP50:95 (%)','.3f')])
 ctable=md_table(conditions[conditions.seed==11].to_dict('records'),[('dataset','데이터셋',lambda x:TITLES[x]),('condition','주입 조건',lambda x:LABELS[x]),('mean_recall_pct','평균 Recall (%)','.3f'),('recall_drop_pp','Recall 감소 (%p)','.3f'),('mean_precision_pct','평균 Precision (%)','.3f'),('mean_map50_pct','평균 mAP50 (%)','.3f'),('mean_map50_95_pct','평균 mAP50:95 (%)','.3f')])
 seedtable=md_table(bright.to_dict('records'),[('dataset','데이터셋',lambda x:TITLES[x]),('seed','시드','d'),('recall_drop_pp','평균 Recall 감소 (%p)','.3f'),('map50_drop_pp','평균 mAP50 감소 (%p)','.3f'),('max_location','관찰 최대 칸',''),('max_location_drop_pp','최대 칸 평균 감소 (%p)','.3f')])
 seedsummary=md_table(seeds.to_dict('records'),[('dataset','데이터셋',lambda x:TITLES[x]),('recall_drop_pp_seed_mean','Recall 감소 시드평균 (%p)','.3f'),('recall_drop_pp_seed_sd','시드 표준편차 (%p)','.3f'),('map50_drop_pp_seed_mean','mAP50 감소 시드평균 (%p)','.3f'),('map50_drop_pp_seed_sd','시드 표준편차 (%p)','.3f'),('peak_cells','시드별 최대 칸','')])
 corrtable=md_table(corrs.to_dict('records'),[('dataset','데이터셋',lambda x:TITLES[x]),('seed_a','시드A','d'),('seed_b','시드B','d'),('pearson_spatial_map','지도 Pearson r','.3f'),('spearman_spatial_map','지도 Spearman ρ','.3f'),('top4_cell_jaccard','상위4칸 Jaccard','.3f')])
 contrasttable=md_table(contrasts.to_dict('records'),[('dataset','데이터셋',lambda x:TITLES[x]),('comparison_label','첫 조건−둘째 조건',''),('recall_drop_contrast_pp','감소량 차이 (%p)','.3f'),('recall_contrast_pattern_sd_pp','패턴 차이 표준편차 (%p)','.3f'),('map50_drop_contrast_pp','mAP50 감소량 차이 (%p)','.3f')])
 pcbtable=md_table(pcb.to_dict('records'),[('cohort','분할',''),('n_test','시험 영상','d'),('n_gt','GT','d'),('clean_recall_pct','원본 Recall (%)','.3f'),('mean_recall_pct','주입 후 평균 Recall (%)','.3f'),('recall_drop_pp','Recall 감소 (%p)','.3f'),('clean_map50_pct','원본 mAP50 (%)','.3f'),('mean_map50_pct','주입 후 mAP50 (%)','.3f')])
 analysis_sections=[]
 for key,entry in analysis.items():
  rel=Path(__import__('os').path.relpath(entry['path'],out));df=entry['frame'];title={'distribution':'기존 공식분할 단일시드의 분포·겹침 분석','positions':'기존 공식분할 단일시드의 위치 공동 밴드','corrected_distribution':'새 PCB 숫자 ID 분할의 분포·겹침 분석','corrected_positions':'새 PCB 숫자 ID 분할의 위치 공동 밴드'}[key]
  if 'distribution' in key:
   t=md_table(df.to_dict('records'),[('title','데이터셋',''),('central_center_fraction','중앙 결함 중심 비율',lambda x:f'{100*x:.2f}%'),('peak_cell','관찰 최대 칸',''),('peak_drop_pp','최대 감소 (%p)','.3f'),('peak_clean_tp_exposure_fraction','최대 칸 cleanTP 겹침',lambda x:f'{100*x:.2f}%'),('spearman_drop_clean_tp_exposure','감소−노출 순위상관','.3f'),('crossfit_residual_spatial_rms_pp','교차적합 기술잔차 RMS (%p)','.3f')])
   note='중앙은 가로·세로 중앙25%, 영상면적6.25%다. 상자 겹침·클래스·객체 크기를 이용한 교차적합 진단은 인과적 위치 효과를 제거한 결과가 아니다. 잔차 감소나 상관만으로 원인을 확정하지 않는다.'
  else:
   t=md_table(df.to_dict('records'),[('dataset','자료',''),('grouping','재표집 그룹',''),('grand_mean_drop_pp','전체평균 감소 (%p)','.3f'),('grand_mean_ci_low_pp','평균 구간 하한 (%p)','.3f'),('grand_mean_ci_high_pp','평균 구간 상한 (%p)','.3f'),('above_grand_mean_cells','평균보다 높은 칸',lambda x:'없음' if pd.isna(x) or not str(x) else str(x)),('constant_location_effect_vector_in_joint_region','동일 위치효과 벡터가 공동영역 안에 있음','')])
   note='같은 이미지 그룹과 패턴의 재표집 가중치를64위치에 공통 적용한 명목95% 근사 동시 밴드다. 위치별 편차는 전체 위치 평균과의 차이다. 확증적 p-value 또는 센서 고유 위치의 인과 효과를 뜻하지 않는다. 숫자 ID 재표집만으로 학습–시험 중복이 제거되지는 않는다.'
  analysis_sections.append(f'## {title}\n\n{t}\n{note}\n\n[상세 방법·수치]({rel.as_posix()}/REPORT_KO.md)\n')
 total_evals=int(conditions.logical_image_evaluations.sum());new_evals=int(conditions.new_image_evaluations.sum());reused_evals=int(conditions.reused_image_evaluations.sum());pair_rows=int(conditions.pair_rows.sum())
 source_checks=sum(len(c['checks']) for c in certificates.values());max_apdiff=max(x['ap_max_abs_difference'] for c in certificates.values() for x in c['checks']);prediction_shards=int(conditions.n_prediction_shards.sum())
 compiled_jobs=conditions[conditions.execution_backend=='bgr-packed-safe-compiled-nms-fp32-bs16-v4'];compiled_calls=int(compiled_jobs.nms_compiled_calls.sum());nms_calls=int(compiled_jobs.nms_calls.sum());near_calls=int(compiled_jobs.nms_near_threshold_fallback_calls.sum())
 checkpoint_rows=conditions[conditions.checkpoint_control_path!=''].drop_duplicates(['dataset','seed']) if 'checkpoint_control_path' in conditions else conditions.iloc[:0];n_checkpoint_controls=len(checkpoint_rows)
 certificate_links=' · '.join(f"[{kind} 경로 인증서](../{ 'compiled_backend_validation.json' if kind=='compiled' else 'fast_backend_validation.json'})" for kind in certificates)
 report=f'''# 제조 화소 손상 연구 — 추가 검증 결과

**실행 완료 범위:** 3개 데이터셋, 학습 시드11·22·33, 총27개 모델/조건 평가를 완료했다. 각 평가는8×8위치·분산형 배치10개(군집 조건은5×5블록10개)와 고정된 시험자료를 사용한다. 모든640개 위치/패턴 평가를 평균했으며 결과에 따라 위치나 패턴을 제외하지 않았다.

## 핵심 비교의 정의

- 모든 모델의 학습 args에서 시드와 데이터 분할을 확인했고, 초기 COCO체크포인트 해시와 학습 recipe가 같음을 검사했다.
- 각 학습 모델·전처리·평가 임계값을 고정한 뒤 손상을 원본 영상 좌표에 주입한다. 각 조건마다 clean1+64×10=641개 예측 묶음이다.
- Recall/Precision은 고정 confidence0.25, 동일 클래스·IoU0.5 대응으로 계산한다. AP는 floor0.001로 보존한 예측을 동일 COCO evaluator로 평가한다.
- 각 위치의 지표는 시험자료 전체를 평가한 값이다. 이를64위치와10패턴에 동일 가중치로 평균한다. 서로 다른 조건의 예측을 한 영상에 합쳐 AP를 계산하지 않는다.
- **감소량=원본−주입 후**이므로 양수는 악화, 음수는 개선이다. 패턴 번호는 데이터셋 내부의 난수 배치 번호이며 데이터셋 간 같은 형상을 뜻하지 않는다.
- 유형 비교는 모두25화소, 강도 비교도25화소를 유지한다. 군집은5×5=25화소로 분산25와 개수를 맞췄다. 실제 변경 수는 원래 값과 같으면 줄 수 있어 CSV에 보존했다.

## 원본 시험 성능

{btable}

## 유형·화소수·강도·군집 비교: 시드11

{ctable}

![조건별 평균 Recall 감소](rq3_conditions_recall_drop.png)

## 조건을 맞춘 차이

표의 값은 첫 조건의 감소량−둘째 조건의 감소량이다. 양수면 첫 조건의 악화가 더 크다. 같은 패턴 번호의64위치 평균끼리 차이를 구했다. 유형(밝음·어두움·128고정)과 강도 비교는 동일한25개 분산 좌표를 사용한다. 화소수1·9는 같은 순서로 생성한25개 패턴의 앞부분을 사용하므로 좌표가 중첩된다. 군집5×5는 별도의layout/side seed를 사용해 실제 좌표와GT노출도 달라진다. 군집−분산 대비는 명령 화소 수와큰 격자 위치를 맞춘 비교이며, 노출을 고정한 연결성만의 인과적 효과가 아니다. 표준편차는10개 패턴 차이의 변동으로, 신뢰구간 또는 학습 시드 변동이 아니다.

{contrasttable}

## 학습 시드 반복: 밝은25화소·분산

{seedtable}

{seedsummary}

![학습 시드별 감소](training_seed_recall_drop.png)
![학습 시드별 위치지도](training_seed_spatial_maps.png)

세 시드의 평균과 표본 표준편차(ddof=1)를 보고했다. 시드3개만으로 모든 모델·촬영환경의 재현성을 보장하지 않는다. 최대 칸은 각 지도에서 사후 선택한 기술통계다.

{corrtable}

지도 상관과 상위4칸 겹침은 기술적 일치도다. Jaccard는 두 상위4칸 집합의 교집합/합집합이며, 동점은 작은 칸 번호를 우선했다. 이는 위치 차이의 유의성 검정이 아니다.

## PCB 기존 공식분할과 숫자 ID 분할

기존 공식분할에서 숫자 ID {split['old_overlap']['n_numeric_groups']}개 중 {split['old_overlap']['n_groups_crossing_splits']}개가 여러 분할에 걸쳤고, {split['old_overlap']['n_groups_in_train_and_test']}개는 학습과 시험에 함께 있었다. 새 분할은 숫자 ID별로 t/b양면과 모든 패치를 묶었고 분할 간 교집합은0이다. 기존 양성{split['n_images']}장과 주석을 보존했다.

{pcbtable}

**시험 코호트가 서로 다르다.** 클래스 구성·객체수·영상 조건도 바뀌므로 성능 차이를 정보누수 제거만의 인과적 효과라고 해석하지 않는다. 숫자 ID 역시 실제 물리 보드임이 확인된 값이 아니라 보수적인 대리 그룹이다. 새 PCB의 패턴 namespace는pcb_ind로 고정해 같은 원본 해상도에서 기존과 같은 마스크 형상을 사용했다.

[PCB 재분할 검증](../pcb_split/REPORT_KO.md)

{''.join(analysis_sections)}

## 실행 검증과 수치 동등성

실행 경로 인증서의 고정 validation control검사는 총{source_checks}건이다. 원래batch16을 유지했고, 전체 예측 사전과 이미지별TP/FP/FN·GT매칭이 일치했으며 COCO AP 차이의 최댓값은{max_apdiff:g}였다. 더 큰 batch의 작은 수치 차이는 채택하지 않았다. 이는 제한된 고정 control의 수치 동등성 검증이며 모든 새 체크포인트에 대한 수학적 보장은 아니다. 내부 전처리·inference·후처리는 고정 버전의 같은 Ultralytics predictor 단계를 사용했다.

{certificate_links}

새 체크포인트별 인증서도{n_checkpoint_controls}개를 확인했다. 각 인증서는 해당 가중치와 현재 데이터·클래스·조건·실행 소스를 지문으로 묶고, validation17장(16장전체배치+1장부분배치)에서clean+7조건 전체 예측 일치를 검사한다. 선택한영상ID가실제validation분할에속함도확인했다. 이것도 제한된control검사이며 임의입력의 수학적동등성보장은아니다.

Compiled 경로는 원래 scores.argsort 순서를 rank로 보존하고 원본 confidence는 바꾸지 않는다. 후보수2048초과·비정상/퇴화·비유한값 또는 IoU기준±1e-5근방이면 원래TorchNMS로 복귀한다. 최종 완료 표식에 기록한 NMS호출{nms_calls:,}회 중 compiled호출{compiled_calls:,}회, IoU경계fallback{near_calls:,}회다. 이 수는 완료 예측의 호출만 합산한다.

완료 상태와 실제 파일을 확인한 총{len(jobs)}개 평가, 예측 묶음{prediction_shards:,}개, 논리적 이미지 평가{total_evals:,}회다. 완료 표식 기준 신규{new_evals:,}회, 명시적 재사용{reused_evals:,}회, 쌍체 이미지 행{pair_rows:,}개다. 원래 분석기가 각 예측의 SHA256을 검사했다. 본 보고서는 완료 상태·run/analysis hash·전체 위치/패턴 수·모델/데이터 일관성을 추가 검사했다. 예측 파일 재해시 옵션 사용 여부는{args.verify_prediction_sha}다. 원본·조건별 예측과 코드·데이터·가중치·수치표 해시를 보존했다. 위 횟수는 최종27작업의 완료 예측만 합산하며 보존된eager부분실행·수치검증·프로토타입의 중간계산은 포함하지 않는다.

## 결과를 해석하는 범위

이번 유형·강도 비교는 처리된RGB영상에서의 합성 손상이다. hot-like는255치환, dead-like는0치환, stuck-like는128치환이며 온도·노출 의존 센서의 물리적 고장 모델은 아니다. 실제 RAW·ISP·DPC 또는 출하 불량률을 검증하지 않았다.

GC10은2048×1000, NEU는200×200, PCB는300×300 원본을 사용하므로 같은25화소라도 면적 비율과 YOLO전처리 후 크기가 다르다. 데이터셋 간 차이를 제조 대상이나 센서의 본질적인 민감도 차이 하나로 설명할 수 없다. 기존과 새 PCB 자료는 양성 영상만 사용하며 정상제품 오거부율을 의미하지 않는다. 실제 코일·제품의 독립성도 확인된 범위까지만 해석한다.

시드 반복, 조건 비교, 공동밴드, 겹침 진단은 서로 다른 질문을 다룬다. 탐색에 사용한 자료의 추가 분석이므로 독립 확증 실험 또는 모든 비교에 걸친 전체 오류율 보장이라고 주장하지 않는다.

## 파일

- [27개 모델/조건 평균](condition_means.csv)
- [패턴별 평균 전체](pattern_means.csv)
- [원본 기준 성능9개](clean_baselines.csv)
- [밝은25화소 시드별 결과](bright25_seed_results.csv)
- [3시드 평균·표준편차](bright25_seed_summary.csv)
- [시드 간 지도 상관·최대위치 비교](seed_spatial_correlations.csv)
- [조건을 맞춘 대비](matched_budget_contrasts.csv)
- [PCB 두 시험 코호트의 별도 비교](pcb_official_vs_numeric_group.csv)
'''
 (out/'REPORT_KO.md').write_text(report)
 dump_json(out/'report_meta.json',{'manifest_sha256':digest_file(manifest_path),'state_sha256':digest_file(state_path),'source_sha256':digest_file(__file__),
  'n_complete_jobs':len(jobs),'n_models':len(baselines),'n_checkpoint_control_certificates':n_checkpoint_controls,'n_prediction_shards':prediction_shards,'logical_image_evaluations':total_evals,'new_image_evaluations':new_evals,'reused_image_evaluations':reused_evals,'paired_rows':pair_rows,
  'prediction_sha_rechecked_by_report':args.verify_prediction_sha,'inputs':{x[0]['key']:x[6] for x in loaded},
  'analysis_inputs':{k:{'path':str(v['path']),'report_sha256':v['report_sha256'],'summary_sha256':v['summary_sha256']} for k,v in analysis.items()},'backend_validation_sha256':{kind:digest_file(base/('compiled_backend_validation.json' if kind=='compiled' else 'fast_backend_validation.json')) for kind in certificates}})
 print('Complete validation report:',out/'REPORT_KO.md');print('Verified jobs:',len(jobs),'models:',len(baselines),'image evaluations:',total_evals)

if __name__=='__main__':
 try:main()
 except (ValueError,FileNotFoundError,KeyError) as exc:
  print('Report not generated: '+str(exc),file=sys.stderr);raise SystemExit(2)
