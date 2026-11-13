"""Audited imports for the official NEU/PCB archives; original files are preserved."""
from pathlib import Path
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor
import argparse,csv,hashlib,json,sys,xml.etree.ElementTree as ET
import cv2,numpy as np,yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from core import dump_json
from prepare import read_rgb,load_gt


def read_neu_annotation(path,names,w,h):
 out=[]
 for obj in ET.parse(path).getroot().findall('object'):
  box=[float(obj.findtext('bndbox/'+k)) for k in ['xmin','ymin','xmax','ymax']]
  box[0]-=1;box[1]-=1
  if not (0<=box[0]<box[2]<=w and 0<=box[1]<box[3]<=h):raise ValueError(f'Invalid VOC box: {path} {box}')
  out.append({'class_id':names.index(obj.findtext('name')),'bbox_xyxy':box,'difficult':obj.findtext('difficult','0')=='1'})
 return out


def read_pcb_coco(path,names):
 j=json.loads(Path(path).read_text());images={x['id']:x for x in j['images']}
 cats={x['id']:names.index(x['name']) for x in j['categories']}
 out={Path(x['file_name']).name:[] for x in j['images']}
 for a in j['annotations']:
  if a.get('iscrowd',0):raise ValueError('Crowd ignore policy not supported')
  im=images[a['image_id']];x,y,w,h=map(float,a['bbox']);x1=x+w;y1=y+h
  if not (-1e-7<=x<x1<=im['width']+1e-7 and -1e-7<=y<y1<=im['height']+1e-7):raise ValueError(f'Invalid official COCO bbox: {a}')
  out[Path(im['file_name']).name].append({'class_id':cats[a['category_id']],
   'bbox_xyxy':[max(0.,x),max(0.,y),min(float(im['width']),x1),min(float(im['height']),y1)]})
 return out


def main():
 parser=argparse.ArgumentParser();parser.add_argument('dataset',choices=['neu_det','pcb_ind']);args=parser.parse_args();dataset=args.dataset
 raw=ROOT/'data/raw'/dataset/'extracted';inputs=ROOT/'inputs';converted=ROOT/'data/imported_labels'/dataset;converted.mkdir(parents=True,exist_ok=True)
 cv2.setNumThreads(1)
 if dataset=='neu_det':
  base=raw/'NEU-DET';names=['crazing','inclusion','patches','pitted_surface','rolled-in_scale','scratches']
  paths=sorted((base/'IMAGES').glob('*.jpg'))
  if len(paths)!=1800:raise ValueError('Expected 1800 NEU images')
  def audit(p):
   im=read_rgb(p);h,w=im.shape[:2];boxes=read_neu_annotation(base/'ANNOTATIONS'/(p.stem+'.xml'),names,w,h)
   return {'image_id':p.stem,'image_path':str(p),'height':h,'width':w,'gt':boxes,'source_split':'unsplit','image_class':p.stem.rsplit('_',1)[0],
    'pixel_hash':hashlib.sha256(str(im.shape).encode()+im.tobytes()).hexdigest()}
 else:
  base=raw/'YOLO';names=yaml.safe_load((base/'data.yaml').read_text())['names']
  official=json.loads((raw/'classes.json').read_text())['classes']
  if names!=[x['name'] for x in sorted(official,key=lambda x:x['id'])]:raise ValueError('Official class mapping mismatch')
  paths=sorted((base/'images').glob('*/*.jpg'))
  if len(paths)!=4789:raise ValueError('Expected 4789 PCB-IND images')
  coco={split:read_pcb_coco(raw/'COCO/annotations'/(split+'.json'),names) for split in ['train','val','test']}
  def audit(p):
   im=read_rgb(p);h,w=im.shape[:2]
   gt=coco[p.parent.name][p.name]
   return {'image_id':p.stem,'image_path':str(p),'height':h,'width':w,'gt':gt,'source_split':p.parent.name,'split':p.parent.name,
    'pixel_hash':hashlib.sha256(str(im.shape).encode()+im.tobytes()).hexdigest()}

 with ThreadPoolExecutor(max_workers=8) as pool:records=list(pool.map(audit,paths))
 by_hash=defaultdict(list)
 for r in records:by_hash[r['pixel_hash']].append(r)
 kept=[];excluded=[];conflicts=[];no_labels=[];duplicate_annotations=[]
 difficult=sum(g.get('difficult',False) for r in records for g in r['gt'])
 for copies in by_hash.values():
  r=copies[0]
  # Compare annotation content, not source ordering or difficult metadata.
  signatures={json.dumps(sorted((g['class_id'],*g['bbox_xyxy']) for g in x['gt'])) for x in copies}
  if len(signatures)>1:
   conflicts.append([x['image_id'] for x in copies]);continue
  if not r['gt']:no_labels.append(r['image_id']);continue
  excluded.extend({'excluded':x['image_id'],'retained':r['image_id']} for x in copies[1:])
  dedup={}
  for g in r['gt']:
   key=(g['class_id'],*g['bbox_xyxy'])
   if key in dedup:duplicate_annotations.append({'image_id':r['image_id'],'box':list(key)})
   dedup[key]=g
  r['gt']=list(dedup.values());kept.append(r)
 if dataset=='neu_det':
  rng=np.random.default_rng(101)
  for cls in names:
   subset=sorted((r for r in kept if r['image_class']==cls),key=lambda r:r['image_id'])
   order=rng.permutation(len(subset));n1=int(.7*len(subset));n2=int(.85*len(subset))
   for i,j in enumerate(order):subset[j]['split']='train' if i<n1 else 'val' if i<n2 else 'test'
 counts={s:np.zeros(len(names),int) for s in ['train','val','test']}
 for r in kept:
  lines=[];h,w=r['height'],r['width']
  for g in r['gt']:
   x0,y0,x1,y1=g['bbox_xyxy'];lines.append(f"{g['class_id']} {(x0+x1)/(2*w):.10f} {(y0+y1)/(2*h):.10f} {(x1-x0)/w:.10f} {(y1-y0)/h:.10f}")
   counts[r['split']][g['class_id']]+=1
  label=converted/(r['image_id']+'.txt');label.write_text('\n'.join(lines)+'\n');load_gt(label,'yolo',names,w,h,None)
  r['label_path']=str(label);r['group_id']=r['image_id'] if dataset=='neu_det' else r['image_id'].rsplit('_',1)[0]
 numeric_groups=defaultdict(set)
 for r in kept:numeric_groups[r['image_id'].split('_')[0]].add(r['split'])
 numeric_overlap={'n_numeric_ids':len(numeric_groups),'cross_split_numeric_ids':sum(len(v)>1 for v in numeric_groups.values()),'train_test_numeric_ids':sum({'train','test'}<=v for v in numeric_groups.values()),'meaning':'Numeric board identity not confirmed; t/b sides may share numeric ID across splits'} if dataset=='pcb_ind' else None
 group_splits=defaultdict(set)
 for r in kept:group_splits[r['group_id']].add(r['split'])
 if any(len(v)>1 for v in group_splits.values()):raise ValueError('Prefix group crosses splits')
 if any((v==0).any() for v in counts.values()):raise ValueError('Missing class in split')
 with (inputs/(dataset+'_index.csv')).open('w',newline='') as f:
  wr=csv.DictWriter(f,fieldnames=['image_id','image_path','label_path','split','group_id']);wr.writeheader();wr.writerows({k:r[k] for k in wr.fieldnames} for r in kept)
 dump_json(inputs/(dataset+'_classes.json'),names)
 dump_json(inputs/(dataset+'_source_audit.json'),{'dataset':dataset,'n_source':len(records),'n_retained':len(kept),
  'split_counts':dict(Counter(r['split'] for r in kept)),'class_counts':{s:v.tolist() for s,v in counts.items()},
  'exact_duplicates_excluded':excluded,'conflicting_exact_duplicate_groups_excluded':conflicts,'empty_labels_excluded':no_labels,
  'duplicate_annotations_removed':duplicate_annotations,'difficult_source_objects':difficult,'difficult_policy':'Include all difficult objects as ordinary GT; none ignored or silently removed' if dataset=='neu_det' else 'Not applicable: official YOLO labels contain no difficult flag',
  'coordinate_convention':'VOC 1-based inclusive -> 0-based half-open (xmin/ymin minus 1)' if dataset=='neu_det' else 'Official COCO xywh -> 0-based half-open; native COCO coordinates avoid 6-decimal YOLO boundary rounding',
  'split_policy':'NEU: per-filename-class stratified 70/15/15, seed=101; no verified product/capture grouping available' if dataset=='neu_det' else 'Official v4 80/10/10 split retained; filename prefix groups checked disjoint across splits',
  'numeric_id_split_overlap':numeric_overlap,'n_groups':len(group_splits),'grouping_warning':'Image IDs are used as bootstrap groups; independent source products/coils are unverified' if dataset=='neu_det' else 'Filename prefix (excluding final patch number) used as bootstrap group; physical product identity unverified',
  'near_duplicate_audit':'Exact decoded pixels checked; no manual near-duplicate audit'})
 print(json.dumps(json.loads((inputs/(dataset+'_source_audit.json')).read_text()),indent=2),flush=True)

if __name__=='__main__':main()
