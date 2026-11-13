"""Evaluate disjoint fixed shards on two GPUs, then analyze only after both succeed."""
import os,subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[1];os.chdir(root)
py=str(root/'.venv/bin/python');env=dict(os.environ,OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',YOLO_CONFIG_DIR=str(root/'.ultralytics'))
base=[py,'src/run.py','--manifest','data/canonical/gc10_det/manifest.jsonl','--classes','data/canonical/gc10_det/classes.json','--weights','runs/train/gc10_yolov8n_s11/weights/best.pt','--config','configs/gc10_frozen.yaml','--split','test','--out','runs/evaluation/gc10_s11_frozen','--batch','16','--threads','8','--devices','0,1','--worker-count','2']
jobs=[]
for i in range(2):
 f=(root/f'logs/inference_gpu{i}.log').open('w')
 jobs.append((subprocess.Popen(base+['--device',str(i),'--worker-index',str(i)],env=env,stdout=f,stderr=subprocess.STDOUT),f))
codes=[]
for p,f in jobs:codes.append(p.wait());f.close()
if any(codes):raise RuntimeError(f'Inference worker failed: {codes}')
print('Both inference workers completed',flush=True)
for name,cmd in [('analysis',[py,'src/analyze.py','--run','runs/evaluation/gc10_s11_frozen','--classes','data/canonical/gc10_det/classes.json','--ap','--plots','--bootstrap','2000']),('report',[py,'scripts/report_gc10.py'])]:
 print('Starting',name,flush=True)
 with (root/f'logs/{name}.log').open('w') as f:subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
 print('Completed',name,flush=True)
print('Experiment complete',flush=True)
