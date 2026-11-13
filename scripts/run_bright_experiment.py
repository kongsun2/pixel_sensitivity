"""Continue one dataset's fixed bright-only experiment after training completes."""
import argparse,os,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--dataset',required=True,choices=['pcb_ind','neu_det']);p.add_argument('--device',default='0');p.add_argument('--devices',help='Optional comma-separated devices for disjoint inference workers');p.add_argument('--wait-pid',type=int);a=p.parse_args()
root=Path(__file__).resolve().parents[1];os.chdir(root);d=a.dataset;py=str(root/'.venv/bin/python')
if a.wait_pid:
 while Path(f'/proc/{a.wait_pid}').exists():time.sleep(5)
log=(root/f'logs/{d}_train.log').read_text(errors='replace')
if 'epochs completed in' not in log or 'Traceback' in log:raise RuntimeError('Training did not finish cleanly')
env=dict(os.environ,OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',YOLO_CONFIG_DIR=str(root/'.ultralytics'))
out=f'runs/evaluation/{d}_s11_bright';data=f'data/canonical/{d}'
commands=[('inference',[py,'src/run.py','--manifest',data+'/manifest.jsonl','--classes',data+'/classes.json','--weights',f'runs/train/{d}_yolov8n_s11/weights/best.pt','--config','configs/bright_frozen.yaml','--split','test','--out',out,'--device',a.device,'--batch','16','--threads','8']),
 ('analysis',[py,'src/analyze.py','--run',out,'--classes',data+'/classes.json','--ap','--plots','--bootstrap','2000']),
 ('report',[py,'scripts/report_bright_replication.py','--dataset',d])]
for name,cmd in commands:
 print('Starting',d,name,flush=True)
 if name=='inference' and a.devices:
  devices=a.devices.split(',')
  if len(devices)!=len(set(devices)):raise ValueError('Duplicate devices')
  jobs=[]
  for i,device in enumerate(devices):
   args=cmd.copy();args[args.index('--device')+1]=device
   args+=['--devices',a.devices,'--worker-count',str(len(devices)),'--worker-index',str(i)]
   f=(root/f'logs/{d}_{name}_gpu{device}.log').open('w')
   jobs.append((subprocess.Popen(args,env=env,stdout=f,stderr=subprocess.STDOUT),f))
  codes=[]
  for worker,f in jobs:codes.append(worker.wait());f.close()
  if any(codes):raise RuntimeError(f'Inference worker failure: {codes}')
 else:
  with (root/f'logs/{d}_{name}.log').open('w') as f:subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
 print('Completed',d,name,flush=True)
print('Experiment complete',d,flush=True)
