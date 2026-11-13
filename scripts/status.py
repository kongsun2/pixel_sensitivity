"""Compact local progress view; does not modify running jobs."""
from pathlib import Path
import csv,json
r=Path(__file__).resolve().parents[1]
p=r/'runs/train/gc10_yolov8n_s11/results.csv'
if p.exists():
 with p.open() as f:rows=list(csv.DictReader(f))
 if rows:
  d={k.strip():v for k,v in rows[-1].items()}
  print('training', {k:d[k] for k in ['epoch','time','metrics/precision(B)','metrics/recall(B)','metrics/mAP50(B)','metrics/mAP50-95(B)']})
p=r/'runs/evaluation/gc10_s11_frozen/predictions'
files=list(p.glob('*.done.json'));print('completed evaluation shards',len(files),'/529')
if files:
 d=json.loads(max(files,key=lambda p:p.stat().st_mtime).read_text());print('last completed',d)
p=r/'logs/train.log'
if p.exists():
 s=p.read_text(errors='replace')
 if 'Traceback' in s:print(s[s.rfind('Traceback'):][-3000:])
