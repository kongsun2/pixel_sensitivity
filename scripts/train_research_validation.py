"""Durable two-GPU training queue for the predeclared replication/corrected-split models."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import queue,threading,json,os,subprocess,time,sys
ROOT=Path(__file__).resolve().parents[1];os.chdir(ROOT);sys.path.insert(0,str(ROOT/'src'))
from core import dump_json,digest_file
OUT=ROOT/'reports/research_validation_2026-11-11';OUT.mkdir(parents=True,exist_ok=True)
LOG=ROOT/'logs/research_validation';LOG.mkdir(parents=True,exist_ok=True)
JOBS=[('gc10_det',22),('neu_det',22),('pcb_ind_board',11),('gc10_det',33),('neu_det',33),('pcb_ind_board',22),('pcb_ind_board',33)]
Q=queue.Queue();[Q.put(x) for x in JOBS];lock=threading.Lock()
state_path=OUT/'training_state.json';state=json.loads(state_path.read_text()) if state_path.exists() else {}
def record(key,value):
 with lock:state[key]=value;dump_json(state_path,state)
def worker(device):
 while True:
  try:d,seed=Q.get_nowait()
  except queue.Empty:return
  key=f'{d}_s{seed}';name=f'validation_20261111_{key}';run=ROOT/'runs/train'/name;data=ROOT/f'data/canonical/{d}/dataset.yaml'
  try:
   while not data.exists():
    record(key,{'status':'waiting_dataset','device':device,'dataset':d,'seed':seed});time.sleep(5)
   old=state.get(key,{})
   if old.get('status')=='complete' and (run/'weights/best.pt').exists() and digest_file(run/'weights/best.pt')==old.get('weights_sha256'):
    print('cached training',key,flush=True);continue
   if run.exists():raise RuntimeError(f'Incomplete training directory exists: {run}; inspect before resume')
   cmd=[str(ROOT/'.venv/bin/python'),'src/train.py','--data',str(data),'--name',name,'--weights',str(ROOT/'yolov8n.pt'),'--seed',str(seed),'--device',str(device),'--epochs','100','--batch','16']
   env=dict(os.environ,OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',YOLO_CONFIG_DIR=str(ROOT/'.ultralytics'))
   record(key,{'status':'running','device':device,'dataset':d,'seed':seed,'command':cmd,'train_source_sha256':digest_file(ROOT/'src/train.py'),'pretrained_sha256':digest_file(ROOT/'yolov8n.pt'),'data_yaml_sha256':digest_file(data)})
   print('Starting training',key,'GPU',device,flush=True)
   with (LOG/(key+'_train.log')).open('w') as f:subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
   log=(LOG/(key+'_train.log')).read_text(errors='replace')
   if 'epochs completed in' not in log:raise RuntimeError('Training completion marker missing')
   record(key,{**state[key],'status':'complete','weights':str(run/'weights/best.pt'),'weights_sha256':digest_file(run/'weights/best.pt')});print('Completed training',key,flush=True)
  except Exception as e:
   record(key,{**state.get(key,{}),'status':'failed','error':str(e)});raise
  finally:Q.task_done()
with ThreadPoolExecutor(max_workers=2) as pool:
 futures=[pool.submit(worker,d) for d in [0,1]]
 for f in futures:f.result()
print('All seven training jobs complete',flush=True)
