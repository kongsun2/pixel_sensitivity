"""Audit all original files; conservatively group filename prefixes across splits.
Prefix grouping is a proxy, not a verified product/coil identity.
Exact decoded duplicates are excluded and documented, never silently retained.
"""
import csv,hashlib,json,sys
from collections import defaultdict,Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import cv2,numpy as np,yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from prepare import load_gt,read_rgb
from core import dump_json
raw=ROOT/'data/raw/gc10_det';inputs=ROOT/'inputs';inputs.mkdir(exist_ok=True)
names=yaml.safe_load((raw/'data/data.yaml').read_text())['names'];names=[names[i] for i in sorted(names)]
files=sorted((raw/'data/images').glob('*/*.jpg'))
if len(files)!=2300:raise ValueError(f'Incomplete mirror: {len(files)} images')
cv2.setNumThreads(1)
def audit(p):
 im=read_rgb(p);h,w=im.shape[:2];label=raw/'data/labels'/p.parent.name/(p.stem+'.txt')
 gt=load_gt(label,'yolo',names,w,h,None)
 return {'image_id':p.stem,'image_path':str(p),'label_path':str(label),'source_split':p.parent.name,
         'group_id':p.stem.rsplit('_',1)[0],'decoded_sha256':hashlib.sha256(str(im.shape).encode()+im.tobytes()).hexdigest(),
         'width':w,'height':h,'gt':gt}
with ThreadPoolExecutor(max_workers=8) as pool:records=list(pool.map(audit,files))
seen={};kept=[];duplicates=[]
for r in records:
 if not r['gt']:continue  # Unannotated does not establish a verified normal product.
 if r['decoded_sha256'] in seen:
  first=seen[r['decoded_sha256']]
  if first['gt']!=r['gt']:raise ValueError('Exact duplicate has conflicting labels; explicit review needed')
  duplicates.append({'excluded':r['image_id'],'retained':first['image_id']});continue
 seen[r['decoded_sha256']]=r;kept.append(r)
groups=sorted({r['group_id'] for r in kept});order=np.random.default_rng(101).permutation(groups).tolist()
# Predefined 70/15/15 group split; do not choose split using model performance.
n1=int(len(order)*.7);n2=int(len(order)*.85)
mapping={g:('train' if i<n1 else 'val' if i<n2 else 'test') for i,g in enumerate(order)}
source_groups=defaultdict(set);class_counts={s:np.zeros(len(names),int) for s in ['train','val','test']}
for r in kept:
 source_groups[r['group_id']].add(r['source_split']);r['split']=mapping[r['group_id']]
 for g in r['gt']:class_counts[r['split']][g['class_id']]+=1
if any((v==0).any() for v in class_counts.values()):raise ValueError('Some classes absent from split')
with (inputs/'gc10_index.csv').open('w',newline='') as f:
 writer=csv.DictWriter(f,fieldnames=['image_id','image_path','label_path','split','group_id']);writer.writeheader()
 writer.writerows({k:r[k] for k in writer.fieldnames} for r in kept)
dump_json(inputs/'gc10_classes.json',names)
dump_json(inputs/'gc10_source_audit.json',{'n_source':len(records),'n_retained':len(kept),'exact_duplicates_excluded':duplicates,
 'split_counts':dict(Counter(r['split'] for r in kept)),'class_counts':{k:v.tolist() for k,v in class_counts.items()},
 'n_prefix_groups':len(groups),'mirror_prefix_groups_crossing_splits':sum(len(s)>1 for s in source_groups.values()),
 'grouping_rule':'filename excluding final underscore frame token; conservative proxy, physical capture identity unverified',
 'split_seed':101,'split_group_ratios':[.7,.15,.15],
 'unverified_empty_label_images_excluded':[r['image_id'] for r in records if not r['gt']],
 'near_duplicate_audit':'Not performed; exact decoded pixel duplicates checked. Prefix grouping reduces suspected sequence leakage but cannot establish independent coils.'})
print(json.dumps(json.loads((inputs/'gc10_source_audit.json').read_text()),indent=2))
