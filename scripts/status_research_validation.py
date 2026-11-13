"""Read-only compact progress of the authorized remaining study."""
from pathlib import Path
import json,csv
ROOT=Path(__file__).resolve().parents[1];out=ROOT/'reports/research_validation_2026-11-11'
p=out/'training_state.json';state=json.loads(p.read_text()) if p.exists() else {}
print('TRAINING',sum(v['status']=='complete' for v in state.values()),'/7 complete')
for key,v in state.items():
 if v['status']!='complete':
  p=ROOT/'runs/train'/('validation_20261111_'+key)/'results.csv';rows=list(csv.DictReader(p.open())) if p.exists() else []
  print(' ',key,v['status'],'epoch',rows[-1]['epoch'] if rows else '-',v.get('error',''))
p=out/'evaluation_state.json';state=json.loads(p.read_text()) if p.exists() else {}
print('EVALUATION',sum(v['status']=='complete' for v in state.values()),'/27 complete')
for key,v in state.items():
 if v['status']!='complete':
  count=len(list((Path(v['run_path'])/'predictions').glob('*.done.json'))) if 'run_path' in v else 0
  print(' ',key,v['status'],count,'/641 shards')
print('EXISTING ANALYSES', {d:(out/d/'REPORT_KO.md').exists() for d in ['distribution','positions','pcb_split']})
for phase in ['training_queue','master_compiled']:
 p=ROOT/f'logs/research_validation/{phase}.log'
 if p.exists():
  text=p.read_text(errors='replace')
  if 'Traceback' in text:print('ERROR',phase,text[text.rfind('Traceback'):][-2500:])
print('FINAL COMPLETE',(out/'COMPLETE.json').exists())
