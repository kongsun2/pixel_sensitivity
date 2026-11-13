"""Compact progress including the distinction between reused and new predictions."""
from pathlib import Path
import json,collections
ROOT=Path(__file__).resolve().parents[1]
for d in ['gc10_det','neu_det','pcb_ind']:
 files=list((ROOT/f'runs/evaluation/{d}_s11_bright25_g8_p10/predictions').glob('*.done.json'));rows=[json.loads(p.read_text()) for p in files]
 reused=sum('cache_reuse' in r for r in rows);counts=collections.Counter(r['repeat'] for r in rows if r['repeat'] is not None)
 print(d,'reused',reused,'new',len(rows)-reused,'/448; complete patterns',[f'P{r+1:02d}' for r,n in sorted(counts.items()) if n==64])
log=ROOT/'logs/p10_pipeline.log'
if log.exists():print(log.read_text()[-1800:])
