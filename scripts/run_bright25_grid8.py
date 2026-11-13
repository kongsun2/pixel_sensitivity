"""Re-evaluate fixed models on 64 locations; do not interpolate the 4x4 maps."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json,os,subprocess,sys,yaml
ROOT=Path(__file__).resolve().parents[1];os.chdir(ROOT);sys.path.insert(0,str(ROOT/'src'))
from core import digest_file,dump_json,load_jsonl,cell_bounds,fault_pattern,stable_seed
PYTHON=str(ROOT/'.venv/bin/python')
ENV=dict(os.environ,OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',YOLO_CONFIG_DIR=str(ROOT/'.ultralytics'))
DATASETS=[('gc10_det','gc10_s11_frozen'),('neu_det','neu_det_s11_bright'),('pcb_ind','pcb_ind_s11_bright')]
CFG=ROOT/'configs/bright25_grid8.yaml';config=yaml.safe_load(CFG.read_text());protocol={}
for d,old in DATASETS:
 meta=json.loads((ROOT/f'runs/evaluation/{old}/run_meta.json').read_text())
 for key,path in [('manifest_sha256',ROOT/f'data/canonical/{d}/manifest.jsonl'),('classes_sha256',ROOT/f'data/canonical/{d}/classes.json'),('weights_sha256',Path(meta['weights_path']))]:
  if digest_file(path)!=meta[key]:raise ValueError(f'4x4 data/model changed: {d} {key}')
 if any(config[k]!=meta['config'][k] for k in ['predict','evaluation','repeats','fault_seed_base']):raise ValueError('Protocol differs beyond grid')
 if config['conditions'][0]!=next(x for x in meta['config']['conditions'] if x['id']=='bright_k25'):raise ValueError('Fault definition changed')
 records=[r for r in load_jsonl(ROOT/f'data/canonical/{d}/manifest.jsonl') if r['split']=='test']
 for h,w in {(r['height'],r['width']) for r in records}:
  for rep in range(3):
   seed=stable_seed('pattern-v1',d,h,w,8,8,'dispersed',1,301,rep)
   offsets=fault_pattern(h,w,8,8,25,'dispersed',1,seed)
   for g in range(64):
    x0,y0,x1,y1=cell_bounds(h,w,8,8,g)
    if not (len(offsets)==25 and offsets[:,0].max()<y1-y0 and offsets[:,1].max()<x1-x0):raise ValueError('Invalid 8x8 pattern')
 protocol[d]={'old_run':old,'weights':meta['weights_path'],'weights_sha256':meta['weights_sha256'],'manifest_sha256':meta['manifest_sha256'],'classes_sha256':meta['classes_sha256'],'n_test':len(records)}
dump_json(ROOT/'reports/bright25_grid8_2026-11-10/protocol.json',{'date':'2026-11-10','config_sha256':digest_file(CFG),'script_sha256':digest_file(__file__),'datasets':protocol,
 'notes':'Reuse same trained models and full test sets. Grid changed from4 to8; 25 sites now confined to a quarter-sized cell, and grid-dependent RNG gives new patterns. Thus this is a new localized injection experiment, not interpolation or a pure resolution-only comparison.'})
def infer_cmd(d,device,worker=None):
 cmd=[PYTHON,'src/run.py','--manifest',f'data/canonical/{d}/manifest.jsonl','--classes',f'data/canonical/{d}/classes.json','--weights',protocol[d]['weights'],'--config',str(CFG),'--split','test','--out',f'runs/evaluation/{d}_s11_bright25_g8','--device',str(device),'--batch','16','--threads','8']
 if worker is not None:cmd+=['--devices','0,1','--worker-count','2','--worker-index',str(worker)]
 return cmd
def analyze(d):
 print('Starting analysis',d,flush=True)
 with (ROOT/f'logs/grid8_{d}_analysis.log').open('w') as f:
  subprocess.run([PYTHON,'src/analyze.py','--run',f'runs/evaluation/{d}_s11_bright25_g8','--classes',f'data/canonical/{d}/classes.json','--ap','--plots','--bootstrap','2000'],env=ENV,stdout=f,stderr=subprocess.STDOUT,check=True)
 print('Completed analysis',d,flush=True)
def single(d,device):
 print('Starting inference',d,flush=True)
 with (ROOT/f'logs/grid8_{d}_inference.log').open('w') as f:subprocess.run(infer_cmd(d,device),env=ENV,stdout=f,stderr=subprocess.STDOUT,check=True)
 print('Completed inference',d,flush=True);analyze(d)
print('All model/data/settings and 8x8 patterns verified',flush=True)
jobs=[]
for device in [0,1]:
 f=(ROOT/f'logs/grid8_gc10_gpu{device}.log').open('w');jobs.append((subprocess.Popen(infer_cmd('gc10_det',device,device),env=ENV,stdout=f,stderr=subprocess.STDOUT),f))
codes=[]
for p,f in jobs:codes.append(p.wait());f.close()
if any(codes):raise RuntimeError(f'GC10 inference worker error: {codes}')
print('Completed inference gc10_det',flush=True)
with ThreadPoolExecutor(max_workers=3) as pool:
 futures=[pool.submit(analyze,'gc10_det'),pool.submit(single,'neu_det',0),pool.submit(single,'pcb_ind',1)]
 for f in futures:f.result()
subprocess.run([PYTHON,'scripts/plot_bright25_grid8.py'],env=ENV,check=True)
print('All three 8x8 experiments complete',flush=True)
