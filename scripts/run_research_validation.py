"""Durable full research-validation schedule; original experiments remain intact."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json,os,subprocess,sys,time,threading,collections
import yaml
ROOT=Path(__file__).resolve().parents[1];os.chdir(ROOT);sys.path.insert(0,str(ROOT/'src'))
from core import dump_json,digest_file,digest_json,load_jsonl,stable_seed,fault_pattern,cell_bounds
from run import versions
OUT=ROOT/'reports/research_validation_2026-11-11';LOG=ROOT/'logs/research_validation';PYTHON=str(ROOT/'.venv/bin/python')
ENV=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',YOLO_CONFIG_DIR=str(ROOT/'.ultralytics'))
DATASETS=['gc10_det','neu_det','pcb_ind_board'];SEEDS=[11,22,33];CONDITIONS=['bright_k25','dark_k25','stuck_k25','bright_k1','bright_k9','bright_partial25','bright_cluster25']
lock=threading.Lock();state_path=OUT/'evaluation_state.json';state=json.loads(state_path.read_text()) if state_path.exists() else {}
def save(key,**kwargs):
 with lock:state[key]={**state.get(key,{}),**kwargs};dump_json(state_path,state)
def wait_prerequisites():
 while True:
  train=json.loads((OUT/'training_state.json').read_text()) if (OUT/'training_state.json').exists() else {}
  failed=[k for k,v in train.items() if v['status']=='failed']
  if failed:raise RuntimeError(f'Training failed: {failed}')
  vpath=OUT/'fast_backend_validation.json';valid=json.loads(vpath.read_text()) if vpath.exists() else {}
  validation_ok=valid.get('all_pass') and valid.get('selected_batch')==16 and valid.get('runner_sha256')==digest_file(ROOT/'src/run_fast.py')
  if len(train)==7 and all(v['status']=='complete' for v in train.values()) and validation_ok:return train,valid
  print('Waiting prerequisites; complete models',sum(v['status']=='complete' for v in train.values()),'/7; final numerical validation',bool(validation_ok),flush=True);time.sleep(30)
def make_jobs(train):
 jobs=[]
 for d in DATASETS:
  for seed in SEEDS:
   if seed==11 and d!='pcb_ind_board':
    old='gc10_s11_frozen' if d=='gc10_det' else 'neu_det_s11_bright';weights=json.loads((ROOT/f'runs/evaluation/{old}/run_meta.json').read_text())['weights_path']
   else:weights=train[f'{d}_s{seed}']['weights']
   for condition in CONDITIONS if seed==11 else ['bright_k25']:
    key=f'{d}_s{seed}__{condition}';jobs.append({'key':key,'dataset':d,'seed':seed,'condition':condition,'run_path':str(ROOT/f'runs/research_validation_20261111/{key}'),'config_path':str(ROOT/f'configs/research_validation/{condition}.yaml'),'weights_path':weights,'pattern_key':'pcb_ind' if d=='pcb_ind_board' else d,'weights_sha256':digest_file(weights),'manifest_sha256':digest_file(ROOT/f'data/canonical/{d}/manifest.jsonl'),'classes_sha256':digest_file(ROOT/f'data/canonical/{d}/classes.json')})
 return jobs

def preflight(jobs):
 for job in jobs:
  cfg=yaml.safe_load(Path(job['config_path']).read_text());c=cfg['conditions'][0]
  rows=[r for r in load_jsonl(ROOT/f"data/canonical/{job['dataset']}/manifest.jsonl") if r['split']=='test'];job['n_test']=len(rows);job['n_gt']=sum(len(r['gt']) for r in rows)
  for h,w in {(r['height'],r['width']) for r in rows}:
   for rep in range(10):
    seed=stable_seed('pattern-v1',job['pattern_key'],h,w,8,8,c['layout'],c['side'],301,rep);off=fault_pattern(h,w,8,8,c['count'],c['layout'],c['side'],seed)
    if len(off)!=c['count']:raise ValueError('Invalid fault budget')
    for cell in range(64):
     x0,y0,x1,y1=cell_bounds(h,w,8,8,cell)
     if off[:,0].max()>=y1-y0 or off[:,1].max()>=x1-x0:raise ValueError('Pattern outside assigned cell')


def model_control(job):
 p=OUT/'model_controls'/(f"{job['dataset']}_s{job['seed']}.json");p.parent.mkdir(exist_ok=True)
 expected_control={'dataset':job['dataset'],'pattern_key':job['pattern_key'],'weights_sha256':job['weights_sha256'],'manifest_sha256':job['manifest_sha256'],'classes_sha256':job['classes_sha256'],'condition_config_sha256':digest_file(ROOT/'configs/research_validation/all_conditions.yaml'),'control_script_sha256':digest_file(ROOT/'scripts/validate_fast_model_control.py'),'source_sha256':{name:digest_file(ROOT/'src'/name) for name in ['run_fast.py','run.py','core.py']}}
 if p.exists():
  v=json.loads(p.read_text())
  if v.get('all_pass') and v.get('fingerprints')==expected_control:return
 cmd=[PYTHON,'scripts/validate_fast_model_control.py','--dataset',job['dataset'],'--weights',job['weights_path'],'--pattern-key',job['pattern_key'],'--device','0','--out',str(p)]
 with (LOG/(f"control_{job['dataset']}_s{job['seed']}.log")).open('w') as f:subprocess.run(cmd,env=ENV,stdout=f,stderr=subprocess.STDOUT,check=True)


def worker_command(job,device,index):
 return [PYTHON,'src/run_fast.py','--manifest',f"data/canonical/{job['dataset']}/manifest.jsonl",'--classes',f"data/canonical/{job['dataset']}/classes.json",'--weights',job['weights_path'],'--config',job['config_path'],'--pattern-key',job['pattern_key'],'--split','test','--out',job['run_path'],'--device',str(device),'--batch','16','--threads','2','--devices','0,1','--worker-count','8','--worker-index',str(index)]
def gpu_wave(job,cap):
 def one(device,index):
  with (LOG/(job['key']+f'_worker{index}.log')).open('a') as f:
   return subprocess.run(worker_command(job,device,index),env=ENV,stdout=f,stderr=subprocess.STDOUT).returncode
 def gpu(device):
  with ThreadPoolExecutor(max_workers=cap) as pool:return list(pool.map(lambda i:one(device,i),range(device,8,2)))
 with ThreadPoolExecutor(max_workers=2) as pool:
  fs=[pool.submit(gpu,d) for d in [0,1]];codes=[c for f in fs for c in f.result()]
 return codes

def analyze(job):
 key=job['key'];save(key,status='analysis_running');print('Starting analysis',key,flush=True)
 with (LOG/(key+'_analysis.log')).open('w') as f:subprocess.run([PYTHON,'src/analyze.py','--run',job['run_path'],'--classes',f"data/canonical/{job['dataset']}/classes.json",'--ap','--plots','--bootstrap','2000'],env=ENV,stdout=f,stderr=subprocess.STDOUT,check=True)
 # Compare original seed11 bright25 full-case metrics to demonstrate parity beyond finite validation controls.
 if job['seed']==11 and job['dataset'] in ['gc10_det','neu_det'] and job['condition']=='bright_k25':
  import pandas as pd,numpy as np
  old=pd.read_csv(ROOT/f"runs/evaluation/{job['dataset']}_s11_bright25_g8_p10/analysis/condition_metrics.csv").sort_values(['repeat','cell']);new=pd.read_csv(Path(job['run_path'])/'analysis/condition_metrics.csv').sort_values(['repeat','cell'])
  columns=['tp','fp','fn','recall','precision','coco_map50','coco_map50_95'];diff=float(np.abs(old[columns].to_numpy()-new[columns].to_numpy()).max());save(key,original_fullcase_max_abs_difference=diff)
  if diff>1e-6:raise RuntimeError(f'Full-case execution parity differs: {key} {diff}')
 output_hashes={name:digest_file(Path(job['run_path'])/'analysis'/name) for name in ['clean_metrics.json','condition_metrics.csv','grid_summary.csv','paired_images.csv','occupancy.csv','analysis_meta.json']}
 save(key,status='complete',analysis_output_sha256=output_hashes);print('Completed analysis',key,flush=True)


def validate_completed_job(job):
    root=Path(job['run_path']);meta=json.loads((root/'run_meta.json').read_text());cfg=yaml.safe_load(Path(job['config_path']).read_text())
    selected=sorted([r for r in load_jsonl(ROOT/f"data/canonical/{job['dataset']}/manifest.jsonl") if r['split']=='test'],key=lambda r:r['image_id'])
    identity={'source_sha256':{name:digest_file(ROOT/'src'/name) for name in ['run_fast.py','run.py','core.py']},
      'backend':'bgr-packed-direct-predictor-fp32-batch16-v3','pattern_key':job['pattern_key'],
      'manifest_sha256':digest_file(ROOT/f"data/canonical/{job['dataset']}/manifest.jsonl"),'classes_sha256':digest_file(ROOT/f"data/canonical/{job['dataset']}/classes.json"),
      'weights_sha256':digest_file(job['weights_path']),'weights_path':str(Path(job['weights_path']).resolve()),'config':cfg,'batch':16,'threads':2,'split':'test',
      'selected_ids':[r['image_id'] for r in selected],'device':'0,1','worker_count':8,'versions':versions()}
    if meta['run_hash']!=digest_json(identity):raise ValueError('Completed job is stale for current model/data/config/source/environment: '+job['key'])
    ameta=json.loads((root/'analysis/analysis_meta.json').read_text())
    if ameta['run_hash']!=meta['run_hash'] or ameta['source_sha256']!={name:digest_file(ROOT/'src'/name) for name in ['analyze.py','core.py','coco_metrics.py']}:raise ValueError('Analysis provenance differs: '+job['key'])
    if ameta['bootstrap']!=2000:raise ValueError('Bootstrap protocol differs')
    expected={'clean'}|{f"{job['condition']}__r{rep:03d}__g{cell:03d}" for rep in range(10) for cell in range(64)}
    markers=list((root/'predictions').glob('*.done.json'))
    if {p.name.removesuffix('.done.json') for p in markers}!=expected:raise ValueError('Completed job missing prediction cases')
    for path in markers:
      mark=json.loads(path.read_text());prediction=path.with_name(path.name.replace('.done.json','.jsonl'))
      if mark['run_hash']!=meta['run_hash'] or mark['n_images']!=len(selected) or mark['sha256']!=digest_file(prediction):raise ValueError('Prediction hash mismatch: '+str(path))
    for name,h in state[job['key']].get('analysis_output_sha256',{}).items():
      if digest_file(root/'analysis'/name)!=h:raise ValueError('Analysis output changed: '+name)
    if not state[job['key']].get('analysis_output_sha256'):raise ValueError('Completed state lacks analysis fingerprints')


def main():
 train,validation=wait_prerequisites();jobs=make_jobs(train);preflight(jobs)
 dump_json(OUT/'evaluation_manifest.json',{'jobs':jobs,'logical_workers_per_job':8,'max_concurrent_per_gpu':2,'execution_batch':16,'config_source_sha256':digest_file(ROOT/'configs/research_validation/all_conditions.yaml'),'master_sha256':digest_file(__file__),'backend_validation_sha256':digest_file(OUT/'fast_backend_validation.json'),'runner_sha256':digest_file(ROOT/'src/run_fast.py')})
 print('Frozen evaluation jobs',len(jobs),'image evaluations',sum(641*j['n_test'] for j in jobs),flush=True)
 with ThreadPoolExecutor(max_workers=3) as pool:
  analyses=[]
  for job in jobs:
   key=job['key']
   if state.get(key,{}).get('status')=='complete':
    model_control(job);validate_completed_job(job)
    print('cached complete',key,flush=True);continue
   if __import__('shutil').disk_usage(ROOT).free<10*1024**3:raise RuntimeError('Insufficient free disk for validated output')
   model_control(job);save(key,status='inference_running',**{k:v for k,v in job.items() if k!='key'});print('Starting inference',key,flush=True)
   codes=gpu_wave(job,2)
   if any(codes):
    print('Retrying incomplete workers at lower concurrency',key,codes,flush=True);codes=gpu_wave(job,1)
   if any(codes):save(key,status='failed',worker_codes=codes);raise RuntimeError(f'Inference failed: {key} {codes}')
   save(key,status='inference_complete');print('Completed inference',key,flush=True);analyses.append(pool.submit(analyze,job))
   for f in analyses:
    if f.done():f.result()
  for f in analyses:f.result()
 pcbjob=next(j for j in jobs if j['dataset']=='pcb_ind_board' and j['seed']==11 and j['condition']=='bright_k25')
 extra=[('distribution',[PYTHON,'scripts/analyze_validation_distribution.py','--run','pcb_ind_board='+pcbjob['run_path'],'--classes','pcb_ind_board='+str(ROOT/'data/canonical/pcb_ind_board/classes.json'),'--out',str(OUT/'corrected_pcb_distribution')]),
 ('positions',[PYTHON,'scripts/analyze_validation_positions.py','--run','pcb_ind_board_s11='+pcbjob['run_path'],'--out',str(OUT/'corrected_pcb_positions')])]
 with ThreadPoolExecutor(max_workers=2) as pool:
  def run_extra(item):
   name,cmd=item
   with (LOG/('corrected_pcb_'+name+'.log')).open('w') as f:subprocess.run(cmd,env=ENV,stdout=f,stderr=subprocess.STDOUT,check=True)
  list(pool.map(run_extra,extra))
 with (LOG/'final_report.log').open('w') as f:subprocess.run([PYTHON,'scripts/report_research_validation.py','--corrected-distribution',str(OUT/'corrected_pcb_distribution'),'--corrected-positions',str(OUT/'corrected_pcb_positions')],env=ENV,stdout=f,stderr=subprocess.STDOUT,check=True)
 dump_json(OUT/'COMPLETE.json',{'complete':True,'jobs':len(jobs),'total_image_evaluations':sum(641*j['n_test'] for j in jobs),'manifest_sha256':digest_file(OUT/'evaluation_manifest.json')})
 print('All authorized research validation experiments complete',flush=True)
if __name__=='__main__':main()
