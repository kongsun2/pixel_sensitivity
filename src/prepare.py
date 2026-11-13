"""Convert a user-audited CSV index (YOLO or VOC labels) to immutable canonical data.
The CSV must contain image_id,image_path,label_path,split,group_id.
Paths in CSV are resolved relative to the CSV directory. Blank label_path is forbidden.
An empty YOLO label FILE is allowed only as an explicitly verified negative sample.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, re
from collections import defaultdict
from pathlib import Path
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import yaml
from core import digest_file, dump_json


def read_rgb(path: Path) -> np.ndarray:
    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if im is None or im.dtype != np.uint8:
        raise ValueError(f'Cannot read uint8 image: {path}')
    if im.ndim == 2:
        return np.repeat(im[...,None],3,axis=2)
    if im.ndim == 3 and im.shape[2] == 3:
        return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    raise ValueError(f'Unsupported channels: {path}; alpha/16bit handling must be specified')


def load_gt(path: Path, fmt: str, names: list[str], w: int, h: int, voc_mode: str | None) -> list[dict]:
    out=[]
    if not path.is_file():
        raise FileNotFoundError(f'Missing annotation is NOT interpreted as a normal image: {path}')
    if fmt == 'yolo':
        for ln,line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(),1):
            if not line.strip(): continue
            fields=list(map(float,line.split()))
            if len(fields)!=5 or not np.isfinite(fields).all():
                raise ValueError(f'{path}:{ln}: require five finite YOLO detection values')
            cl,xc,yc,bw,bh=fields
            if cl != int(cl): raise ValueError('Noninteger class')
            box=[(xc-bw/2)*w,(yc-bh/2)*h,(xc+bw/2)*w,(yc+bh/2)*h]
            out.append({'class_id':int(cl),'bbox_xyxy':box})
    else:
        root=ET.parse(path).getroot()
        size=root.find('size')
        if size is not None:
            if int(size.findtext('width','-1'))!=w or int(size.findtext('height','-1'))!=h:
                raise ValueError(f'XML/image dimensions differ: {path}')
        for obj in root.findall('object'):
            if int(obj.findtext('difficult','0')) != 0:
                raise ValueError('difficult objects need an explicit ignore policy; not silently discarded')
            cl=names.index(obj.findtext('name',''))
            b=obj.find('bndbox')
            if b is None: raise ValueError(f'Missing bndbox: {path}')
            x0,y0,x1,y1=[float(b.findtext(k,'nan')) for k in ['xmin','ymin','xmax','ymax']]
            if voc_mode=='1inclusive': x0-=1; y0-=1
            elif voc_mode=='0inclusive': x1+=1; y1+=1
            elif voc_mode!='0halfopen': raise ValueError('Explicit VOC coordinate convention required')
            out.append({'class_id':cl,'bbox_xyxy':[x0,y0,x1,y1]})
    seen=set()
    for g in out:
        x0,y0,x1,y1=g['bbox_xyxy']
        if not (0<=g['class_id']<len(names)) or not np.isfinite(g['bbox_xyxy']).all():
            raise ValueError(f'Invalid class/box: {path}')
        eps=1e-4
        if not (-eps<=x0<x1<=w+eps and -eps<=y0<y1<=h+eps):
            raise ValueError(f'Out-of-image or empty box: {path}: {g}')
        g['bbox_xyxy']=[max(0.,x0),max(0.,y0),min(float(w),x1),min(float(h),y1)]
        key=(g['class_id'],*g['bbox_xyxy'])
        if key in seen: raise ValueError(f'Duplicate GT annotation: {path}')
        seen.add(key)
    return out


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--index',required=True)
    p.add_argument('--classes',required=True,help='JSON list of names in class-id order')
    p.add_argument('--dataset',required=True)
    p.add_argument('--format',choices=['yolo','voc'],required=True)
    p.add_argument('--voc-mode',choices=['0halfopen','0inclusive','1inclusive'])
    p.add_argument('--out',required=True)
    p.add_argument('--split-mode',choices=['supplied','group-random'],default='supplied')
    p.add_argument('--seed',type=int,default=101)
    args=p.parse_args()
    index=Path(args.index).resolve(); dest=Path(args.out).resolve()
    if dest.exists() and any(dest.iterdir()): raise FileExistsError('Use a NEW output directory')
    names=json.loads(Path(args.classes).read_text(encoding='utf-8-sig'))
    if not isinstance(names,list) or not names or len(set(names))!=len(names) or not all(isinstance(s,str) for s in names):
        raise ValueError('classes JSON must be a nonempty unique string list')
    with index.open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
    if not rows: raise ValueError('Empty index')
    ids=set()
    for r in rows:
        if not re.fullmatch(r'[A-Za-z0-9_.-]+',r['image_id']): raise ValueError('Unsafe image_id')
        if r['image_id'] in ids: raise ValueError('Duplicate image_id')
        ids.add(r['image_id'])
        if not r['group_id']: raise ValueError('group_id required; use image_id only when real grouping unknown')
    if args.split_mode=='group-random':
        groups=sorted({r['group_id'] for r in rows})
        if len(groups)<10: raise ValueError('Too few groups for this example random split; supply planned splits')
        order=np.random.default_rng(args.seed).permutation(groups).tolist()
        n1=int(len(order)*.70); n2=int(len(order)*.85)
        mapping={g:('train' if i<n1 else 'val' if i<n2 else 'test') for i,g in enumerate(order)}
        for r in rows:r['split']=mapping[r['group_id']]
    group_splits=defaultdict(set)
    for r in rows:
        if r['split'] not in {'train','val','pilot','test'}:raise ValueError('Bad split name')
        group_splits[r['group_id']].add(r['split'])
    if any(len(s)>1 for s in group_splits.values()):
        raise ValueError('Known group crosses splits; repair the index or use group-random')
    (dest/'images').mkdir(parents=True); (dest/'labels').mkdir()
    records=[]; pixel_hashes={}; split_files=defaultdict(list); counts=defaultdict(lambda:np.zeros(len(names),int))
    for r in rows:
        ip=(index.parent/r['image_path']).resolve(); lp=(index.parent/r['label_path']).resolve()
        if not r['label_path']:raise ValueError('Blank label_path')
        im=read_rgb(ip); h,w=im.shape[:2]
        ph=hashlib.sha256(str(im.shape).encode()+im.tobytes()).hexdigest()
        if ph in pixel_hashes: raise ValueError(f'Duplicate decoded image: {r["image_id"]}, {pixel_hashes[ph]}; audit before conversion')
        pixel_hashes[ph]=r['image_id']
        gt=load_gt(lp,args.format,names,w,h,args.voc_mode)
        op=dest/'images'/f'{r["image_id"]}.png'
        if not cv2.imwrite(str(op),cv2.cvtColor(im,cv2.COLOR_RGB2BGR)):
            raise IOError(f'Cannot write: {op}')
        yp=dest/'labels'/f'{r["image_id"]}.txt'
        label_lines=[]
        for j,g in enumerate(gt):
            g['gt_id']=f'{r["image_id"]}:{j}'
            x0,y0,x1,y1=g['bbox_xyxy']
            label_lines.append(f'{g["class_id"]} {(x0+x1)/(2*w):.10f} {(y0+y1)/(2*h):.10f} {(x1-x0)/w:.10f} {(y1-y0)/h:.10f}')
            counts[r['split']][g['class_id']]+=1
        yp.write_text('\n'.join(label_lines)+('\n' if label_lines else ''),encoding='utf-8')
        records.append({'dataset_id':args.dataset,'image_id':r['image_id'],'group_id':r['group_id'],
            'split':r['split'],'image_path':str(op),'width':w,'height':h,
            'image_sha256':digest_file(op),'source_image_path':str(ip),
            'source_sha256':digest_file(ip),'gt':gt})
        split_files[r['split']].append(str(op))
    for split,paths in split_files.items():
        (dest/f'{split}.txt').write_text('\n'.join(paths)+'\n',encoding='utf-8')
    for required in ['train','val','test']:
        if required not in split_files:raise ValueError(f'Missing {required} split')
    (dest/'manifest.jsonl').write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in records)+'\n',encoding='utf-8')
    (dest/'dataset.yaml').write_text(yaml.safe_dump({'path':str(dest),'train':'train.txt','val':'val.txt','test':'test.txt',
        'names':dict(enumerate(names))},allow_unicode=True,sort_keys=False),encoding='utf-8')
    dump_json(dest/'classes.json',names)
    dump_json(dest/'audit.json',{'dataset':args.dataset,'split_mode':args.split_mode,'seed':args.seed,
        'index_sha256':digest_file(index),'n_images':dict((s,len(v)) for s,v in split_files.items()),
        'class_counts':{s:c.tolist() for s,c in counts.items()},'n_groups':len(group_splits),
        'warning':'A group_id based only on image_id does not establish independent products/lots.'})
    print(dest/'dataset.yaml')

if __name__=='__main__':main()
