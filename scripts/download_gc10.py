"""Download pinned images plus bulk annotations, avoiding thousands of label requests."""
import json,time
from pathlib import Path
from huggingface_hub import snapshot_download
import requests
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
REPO='dronefreak/GC10-DET'
REVISION='42c086fa8dea913987febe794b1e7608b3db464d'
out=ROOT/'data/raw/gc10_det'
if len(list((out/'data/images').glob('*/*.jpg')))!=2300:
    snapshot_download(REPO,repo_type='dataset',revision=REVISION,local_dir=out,
        allow_patterns=['README.md','data/data.yaml','data/images/**/*.jpg'],max_workers=8)
checked=0
covered=set()
for split in ['train','valid','test']:
    rel=f'data/images/{split}/metadata.jsonl'
    for attempt in range(8):
        response=requests.get(f'https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{rel}',timeout=90)
        if response.status_code!=429:break
        wait=int(response.headers.get('Retry-After','60'))+2
        print(f'HF rate limit; waiting {wait}s before bulk annotation retry',flush=True)
        while wait>0:
            step=min(30,wait);time.sleep(step);wait-=step
    response.raise_for_status();(out/rel).write_text(response.text)
    labels=out/'data/labels'/split;labels.mkdir(parents=True,exist_ok=True)
    for line in response.text.splitlines():
        row=json.loads(line)
        key=(split,row['file_name'])
        if key in covered:raise ValueError('Duplicate metadata record')
        covered.add(key)
        if len(row['objects']['categories'])!=len(row['objects']['bbox']):raise ValueError('Class/box count differs')
        w,h=Image.open(out/'data/images'/split/row['file_name']).size
        lines=[]
        for cl,(x,y,bw,bh) in zip(row['objects']['categories'],row['objects']['bbox']):
            lines.append(f'{cl} {(x+bw/2)/w:.10f} {(y+bh/2)/h:.10f} {bw/w:.10f} {bh/h:.10f}')
        dest=labels/(Path(row['file_name']).stem+'.txt');value='\n'.join(lines)+('\n' if lines else '')
        if dest.exists():
            old=dest.read_text().strip();new=value.strip()
            if old or new:
                if not np.allclose(np.array([list(map(float,l.split())) for l in old.splitlines()]),
                                   np.array([list(map(float,l.split())) for l in new.splitlines()]),atol=1e-6,rtol=0):
                    raise ValueError(f'Bulk metadata differs from existing YOLO label: {dest}')
            checked+=1
        dest.write_text(value)
if len(covered)!=2300:raise ValueError(f'Incomplete metadata: {len(covered)}')
(out/'provenance.json').write_text(json.dumps({'repo':REPO,'revision':REVISION,
    'url':'https://huggingface.co/datasets/'+REPO,'retrieved_date':'2026-11-04',
    'labels':'Reconstructed losslessly from same-revision bulk COCO xywh metadata; normalized to YOLO',
    'existing_labels_cross_checked':checked},indent=2))
print('Download complete; existing labels cross-checked:',checked,flush=True)
