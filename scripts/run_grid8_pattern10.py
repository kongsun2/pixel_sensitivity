"""Ten dispersed masks: verified reuse of masks 0..2, inference of masks 3..9."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import copy,json,os,shutil,subprocess,sys
import yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from core import digest_file,digest_json,dump_json,load_jsonl,stable_seed,fault_pattern,cell_bounds
from run import versions
INVARIANTS=['source_sha256','manifest_sha256','classes_sha256','weights_sha256','weights_path','batch','threads','split','selected_ids','versions']
DATASETS=['gc10_det','neu_det','pcb_ind']
CFG=ROOT/'configs/bright25_grid8_pattern10.yaml'
OUT=ROOT/'reports/bright25_grid8_pattern10_2026-11-10'
PYTHON=str(ROOT/'.venv/bin/python')
ENV=dict(os.environ,OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',YOLO_CONFIG_DIR=str(ROOT/'.ultralytics'))


def check_reuse_contract(old,new):
 for key in INVARIANTS:
  if old[key]!=new[key]:raise ValueError(f'Prediction reuse incompatible: {key}')
 expected=copy.deepcopy(old['config']);expected['repeats']=new['config']['repeats']
 if new['config']!=expected or new['config']['repeats']<old['config']['repeats']:
  raise ValueError('Only increasing repeat count is compatible with prediction reuse')


def initialize(d,cfg):
 oldroot=ROOT/f'runs/evaluation/{d}_s11_bright25_g8';old=json.loads((oldroot/'run_meta.json').read_text())
 identity={key:copy.deepcopy(old[key]) for key in INVARIANTS}
 identity.update(config=cfg,device='0,1',worker_count=2)
 identity['source_sha256']={name:digest_file(ROOT/'src'/name) for name in ['run.py','core.py']}
 identity['versions']=versions()
 for key,path in [('manifest_sha256',ROOT/f'data/canonical/{d}/manifest.jsonl'),('classes_sha256',ROOT/f'data/canonical/{d}/classes.json'),('weights_sha256',Path(identity['weights_path']))]:identity[key]=digest_file(path)
 selected=sorted([r for r in load_jsonl(ROOT/f'data/canonical/{d}/manifest.jsonl') if r['split']=='test'],key=lambda x:x['image_id'])
 identity['selected_ids']=[r['image_id'] for r in selected];check_reuse_contract(old,identity)
 shapes=sorted({(r['height'],r['width']) for r in selected});pattern_records=[]
 for h,w in shapes:
  seen=set()
  for rep in range(10):
   seed=stable_seed('pattern-v1',d,h,w,8,8,'dispersed',1,301,rep);offsets=fault_pattern(h,w,8,8,25,'dispersed',1,seed)
   shape_hash=digest_json(offsets.tolist())
   if shape_hash in seen:raise ValueError('Repeated random placement pattern')
   seen.add(shape_hash)
   for cell in range(64):
    x0,y0,x1,y1=cell_bounds(h,w,8,8,cell)
    if len(offsets)!=25 or offsets[:,0].max()>=y1-y0 or offsets[:,1].max()>=x1-x0:raise ValueError('Invalid pattern capacity')
   pattern_records.append({'dataset':d,'pattern_id':rep+1,'repeat_index':rep,'width':w,'height':h,'seed':seed,'offsets_yx':offsets.tolist(),'offsets_sha256':shape_hash})
 root=ROOT/f'runs/evaluation/{d}_s11_bright25_g8_p10';pred=root/'predictions';pred.mkdir(parents=True,exist_ok=True);run_hash=digest_json(identity)
 if (root/'run_meta.json').exists():
  if json.loads((root/'run_meta.json').read_text())['run_hash']!=run_hash:raise ValueError('Output contains another experiment')
 else:dump_json(root/'run_meta.json',{'run_hash':run_hash,**identity,'reuse_provenance':{'run':str(oldroot),'run_hash':old['run_hash'],'patterns':[1,2,3],'validation':'same source, model, full test IDs, versions, preprocessing, injection and matching settings; only total repeats and worker assignment differ'}})
 markers=list((oldroot/'predictions').glob('*.done.json'))
 if len(markers)!=193:raise ValueError('Old 3-pattern experiment is incomplete')
 reused=0
 for marker in markers:
  mark=json.loads(marker.read_text());source=marker.with_name(marker.name.replace('.done.json','.jsonl'))
  if mark['run_hash']!=old['run_hash'] or mark['sha256']!=digest_file(source) or mark['n_images']!=len(selected):raise ValueError('Invalid original prediction shard')
  target=pred/source.name;done=pred/marker.name
  if done.exists():
   existing=json.loads(done.read_text())
   if existing['run_hash']!=run_hash or existing['sha256']!=digest_file(target) or existing['sha256']!=mark['sha256']:raise ValueError('Invalid reused cache')
  else:
   tmp=target.with_suffix('.partial');shutil.copyfile(source,tmp);tmp.replace(target)
   dump_json(done,{**mark,'run_hash':run_hash,'seconds_wall':0.0,'original_seconds_wall':mark['seconds_wall'],
    'cache_reuse':{'source_run_hash':old['run_hash'],'source_file':str(source),'source_sha256':mark['sha256']}})
  reused+=1
 print('Initialized',d,'verified cached shards',reused,flush=True)
 return identity,pattern_records


def infer(d,device,worker,identity):
 cmd=[PYTHON,'src/run.py','--manifest',f'data/canonical/{d}/manifest.jsonl','--classes',f'data/canonical/{d}/classes.json',
 '--weights',identity['weights_path'],'--config',str(CFG),'--split','test','--out',f'runs/evaluation/{d}_s11_bright25_g8_p10',
 '--device',str(device),'--batch','16','--threads','8','--devices','0,1','--worker-count','2','--worker-index',str(worker)]
 return cmd


def analyze(d):
 print('Starting analysis',d,flush=True)
 with (ROOT/f'logs/p10_{d}_analysis.log').open('w') as f:
  subprocess.run([PYTHON,'src/analyze.py','--run',f'runs/evaluation/{d}_s11_bright25_g8_p10','--classes',f'data/canonical/{d}/classes.json','--ap','--plots','--bootstrap','2000'],env=ENV,stdout=f,stderr=subprocess.STDOUT,check=True)
 print('Completed analysis',d,flush=True)


def main():
 os.chdir(ROOT);OUT.mkdir(parents=True,exist_ok=True);cfg=yaml.safe_load(CFG.read_text());identities={};patterns=[]
 for d in DATASETS:
  identities[d],pats=initialize(d,cfg);patterns+=pats
 dump_json(OUT/'patterns.json',patterns)
 dump_json(OUT/'protocol.json',{'date':'2026-11-10','config_sha256':digest_file(CFG),'source_sha256':digest_file(__file__),
 'patterns':'10 dispersed random-placement masks, not 10 defect types; P01..P03 reused after contract+SHA verification; P04..P10 newly evaluated',
 'averaging':'For each dataset/pattern average 64 independent location evaluations; overall equal mean of 10 pattern means. Do not combine predictions from different masks into a single AP evaluation.',
 'datasets':{d:{'n_test':len(x['selected_ids']),'weights_sha256':x['weights_sha256'],'manifest_sha256':x['manifest_sha256']} for d,x in identities.items()},
 'total_evaluations_including_reuse':700613,'new_evaluations':489664})
 with ThreadPoolExecutor(max_workers=3) as pool:
  analyses=[]
  for d in DATASETS:
   print('Starting inference',d,flush=True);jobs=[]
   for device in [0,1]:
    f=(ROOT/f'logs/p10_{d}_gpu{device}.log').open('w');jobs.append((subprocess.Popen(infer(d,device,device,identities[d]),env=ENV,stdout=f,stderr=subprocess.STDOUT),f))
   codes=[]
   for p,f in jobs:codes.append(p.wait());f.close()
   if any(codes):raise RuntimeError(f'Inference workers failed: {d} {codes}')
   print('Completed inference',d,flush=True);analyses.append(pool.submit(analyze,d))
  for f in analyses:f.result()
 subprocess.run([PYTHON,'scripts/report_grid8_pattern10.py'],env=ENV,check=True)
 print('All ten-pattern experiments completed',flush=True)

if __name__=='__main__':main()
