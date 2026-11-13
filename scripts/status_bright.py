"""Compact progress of the two additional experiments."""
from pathlib import Path
import csv
root=Path(__file__).resolve().parents[1]
for d in ['neu_det','pcb_ind']:
 p=root/f'runs/train/{d}_yolov8n_s11/results.csv'
 if p.exists():
  rows=list(csv.DictReader(p.open()))
  if rows:print(d,'epoch',rows[-1]['epoch'],'val mAP50',rows[-1]['metrics/mAP50(B)'])
 markers=list((root/f'runs/evaluation/{d}_s11_bright/predictions').glob('*.done.json'));print(d,'completed shards',len(markers),'/241')
 p=root/f'logs/{d}_experiment.log'
 if p.exists():print(p.read_text().strip())
 for phase in ['train','inference','analysis','report']:
  p=root/f'logs/{d}_{phase}.log'
  if p.exists():
   s=p.read_text(errors='replace')
   if 'Traceback' in s:print(s[s.rfind('Traceback'):][-2500:])
