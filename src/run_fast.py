"""Fast FP32 BGR/packed-transfer inference; same raw-space injection and YOLO preprocessing.
Requires a trained YOLOv8 checkpoint. Test scores must not be used to tune config.
"""
from __future__ import annotations
import argparse, importlib.metadata, json, platform, re, time
from functools import lru_cache
from pathlib import Path
import cv2, numpy as np, yaml
from core import cell_bounds,digest_file,digest_json,dump_json,fault_pattern,inject_fault,load_jsonl,stable_seed


def versions():
    out={'python':platform.python_version()}
    for p in ['ultralytics','torch','torchvision','numpy','opencv-python','PyYAML','pycocotools']:
        try:out[p]=importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:out[p]='not-installed'
    return out


def load_clean(row):
    p=Path(row['image_path'])
    if digest_file(p)!=row['image_sha256']:raise ValueError(f'Image changed after manifest freeze: {p}')
    bgr=cv2.imread(str(p),cv2.IMREAD_COLOR)
    if bgr is None:raise ValueError(f'Cannot read {p}')
    rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
    if rgb.shape[:2]!=(row['height'],row['width']):raise ValueError('Image shape changed')
    return rgb


class NmsDeadlineGuard(__import__('logging').Handler):
    """Reject incomplete detection batches rather than silently save NMS timeouts."""
    def emit(self,record):
        message=record.getMessage()
        if 'NMS time limit' in message and 'exceeded' in message:
            raise RuntimeError('Incomplete NMS batch rejected: '+message)


def pack_predictions(results):
    """One GPU->CPU transfer per batch, preserving the original result/image order."""
    import torch
    tensors=[r.boxes.data if r.boxes is not None else None for r in results]
    lengths=[0 if t is None else len(t) for t in tensors]
    if not sum(lengths):return [[] for _ in results]
    active=[t for t in tensors if t is not None and len(t)]
    if any(t.shape[1]!=6 for t in active):raise ValueError('Expected nontracking detection boxes')
    flat=torch.cat(active,dim=0).detach().cpu().numpy();out=[];start=0
    for n in lengths:
        out.append([{'class_id':int(row[5]),'score':float(row[4]),'bbox_xyxy':[float(v) for v in row[:4]]} for row in flat[start:start+n]])
        start+=n
    return out


def predict_packed(model,bgrs,pred_cfg,device):
    """Reuse the pinned Ultralytics predictor stages, omitting only outer-loop bookkeeping/profiling."""
    import torch
    from ultralytics.utils import LOGGER
    if not any(isinstance(h,NmsDeadlineGuard) for h in LOGGER.handlers):LOGGER.addHandler(NmsDeadlineGuard())
    predictor=model.predictor
    if predictor is None or not getattr(predictor,'done_warmup',False):
        results=model.predict(source=bgrs,imgsz=pred_cfg['imgsz'],conf=pred_cfg['conf_floor'],
            iou=pred_cfg['nms_iou'],max_det=pred_cfg['max_det'],rect=False,half=False,
            augment=False,agnostic_nms=False,device=device,save=False,verbose=False)
    else:
        predictor.batch=([f'array_{i}.jpg' for i in range(len(bgrs))],bgrs,['']*len(bgrs))
        with torch.inference_mode():
            tensor=predictor.preprocess(bgrs)
            raw=predictor.inference(tensor)
            if tensor.device.type=='cuda':torch.cuda.synchronize(tensor.device)
            results=predictor.postprocess(raw,tensor,bgrs)
    return pack_predictions(results)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',required=True);p.add_argument('--classes',required=True)
    p.add_argument('--weights',required=True);p.add_argument('--config',required=True)
    p.add_argument('--split',choices=['val','pilot','test'],required=True)
    p.add_argument('--out',required=True);p.add_argument('--device',default='0')
    p.add_argument('--clean-only',action='store_true');p.add_argument('--limit',type=int)
    p.add_argument('--batch',type=int,default=16);p.add_argument('--threads',type=int,default=8)
    p.add_argument('--pattern-key',default=None,help='Stable dataset namespace for mask seeds')
    p.add_argument('--devices',default=None,help='Full device list used by disjoint shard workers')
    p.add_argument('--worker-index',type=int,default=0);p.add_argument('--worker-count',type=int,default=1)
    a=p.parse_args()
    import torch
    torch.set_num_threads(a.threads);cv2.setNumThreads(1)
    if a.batch < 1:raise ValueError('batch must be positive')
    if not 0<=a.worker_index<a.worker_count:raise ValueError('Invalid worker assignment')
    if a.worker_count>1 and not a.devices:raise ValueError('Distributed workers require shared --devices identity')
    cfg=yaml.safe_load(Path(a.config).read_text(encoding='utf-8'))
    rows=sorted((r for r in load_jsonl(a.manifest) if r['split']==a.split),key=lambda r:r['image_id'])
    if a.limit is not None:rows=rows[:a.limit]
    if not rows:raise ValueError('No records for split')
    datasets={r['dataset_id'] for r in rows}
    if len(datasets)!=1:raise ValueError('One dataset per run')
    names=json.loads(Path(a.classes).read_text(encoding='utf-8'))
    weights=Path(a.weights).resolve()
    if not weights.is_file():raise FileNotFoundError('Provide an existing trained best.pt')
    for item in cfg['conditions']:
        if item.get('channels','all')!='all':raise ValueError('Fast BGR backend supports all-channel scalar faults only')
        if not re.fullmatch(r'[A-Za-z0-9_.-]+',item['id']):raise ValueError('Unsafe condition id')
        if item['count']<1:raise ValueError('Use the clean shard for zero damage')
    if len({x['id'] for x in cfg['conditions']})!=len(cfg['conditions']):raise ValueError('Duplicate conditions')
    if not (0<=cfg['predict']['conf_floor']<=cfg['evaluation']['conf_operating']<=1):
        raise ValueError('conf_floor must not exceed operating threshold')
    out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=True)
    identity={'source_sha256':{name:digest_file(Path(__file__).parent/name) for name in ['run_fast.py','run.py','core.py']},
        'backend':'bgr-packed-direct-predictor-fp32-batch16-v3','pattern_key':a.pattern_key or next(iter(datasets)),
        'manifest_sha256':digest_file(a.manifest),'classes_sha256':digest_file(a.classes),
        'weights_sha256':digest_file(weights),'weights_path':str(weights),'config':cfg,
        'batch':a.batch,'threads':a.threads,'split':a.split,'selected_ids':[r['image_id'] for r in rows],'device':a.devices or a.device,'worker_count':a.worker_count,'versions':versions()}
    run_hash=digest_json(identity)
    if a.worker_index>0:
        for _ in range(120):
            if (out/'run_meta.json').exists():break
            time.sleep(.5)
        else:raise RuntimeError('Worker 0 did not initialize run metadata')
    if (out/'run_meta.json').exists():
        existing=json.loads((out/'run_meta.json').read_text())
        if existing['run_hash']!=run_hash:raise ValueError('Output directory has a different config/model/data/environment; use a new directory')
    else:dump_json(out/'run_meta.json',{'run_hash':run_hash,**identity})
    from ultralytics import YOLO
    model=YOLO(str(weights))
    model_names=[model.names[i] for i in range(len(model.names))]
    if model_names!=names:raise ValueError(f'Checkpoint class order mismatch: {model_names} != {names}')
    pred_cfg=cfg['predict'];gr,gc=cfg['grid']['rows'],cfg['grid']['cols']

    # Decode and verify each source once; faults always operate on copies.
    images={r['image_id']:np.ascontiguousarray(load_clean(r)[...,::-1]) for r in rows}  # BGR once; all-channel injection commutes with channel reversal
    pattern=lru_cache(maxsize=1024)(fault_pattern)
    def infer_batch(bgrs):
        return predict_packed(model,bgrs,pred_cfg,a.device)
    infer_batch([images[rows[0]['image_id']]])
    expected=[r['image_id'] for r in rows]
    def write_shard(name,condition=None,rep=None,cell=None):
        path=out/'predictions'/f'{name}.jsonl';done=path.with_suffix('.done.json')
        path.parent.mkdir(parents=True,exist_ok=True)
        if done.exists() and path.exists():
            mark=json.loads(done.read_text())
            if mark['run_hash']!=run_hash or mark['sha256']!=digest_file(path):raise ValueError('Corrupted cached shard')
            if [r['image_id'] for r in load_jsonl(path)]!=expected:raise ValueError('Cached image set mismatch')
            print('cached',name,flush=True);return
        tmp=path.with_suffix('.partial');wall_start=time.perf_counter()
        with tmp.open('w',encoding='utf-8') as f:
            for start in range(0,len(rows),a.batch):
                chunk=rows[start:start+a.batch];rgbs=[];metas=[]
                for row in chunk:
                    clean=images[row['image_id']];rgb=clean;fault_meta=None
                    if condition is not None:
                        h,w=clean.shape[:2]
                        seed=stable_seed('pattern-v1',a.pattern_key or row['dataset_id'],h,w,gr,gc,
                            condition['layout'],condition.get('side',1),cfg['fault_seed_base'],rep)
                        offsets=pattern(h,w,gr,gc,condition['count'],condition['layout'],condition.get('side',1),seed)
                        bounds=cell_bounds(h,w,gr,gc,cell)
                        rgb,fault_meta=inject_fault(clean,offsets,bounds,condition['target'],condition['alpha'],condition.get('channels','all'))
                        fault_meta.update({'bounds_xyxy':list(bounds),'pattern_seed':seed,'cell':cell,'repeat':rep})
                    rgbs.append(rgb);metas.append(fault_meta)
                predictions=infer_batch(rgbs)
                if len(predictions)!=len(chunk):raise ValueError('Inference batch count mismatch')
                for row,pred,fault_meta in zip(chunk,predictions,metas):
                    record={'image_id':row['image_id'],'group_id':row['group_id'],
                        'height':row['height'],'width':row['width'],'gt':row['gt'],'pred':pred,'fault':fault_meta}
                    f.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
            f.flush()
        tmp.replace(path)
        dump_json(done,{'run_hash':run_hash,'sha256':digest_file(path),'n_images':len(rows),
            'seconds_wall':time.perf_counter()-wall_start,'condition_id':condition['id'] if condition else 'clean',
            'repeat':rep,'cell':cell,'worker_device':a.device})
        print('done',name,flush=True)
    if a.worker_index==0:write_shard('clean')
    if not a.clean_only:
        task_index=0
        for cond in cfg['conditions']:
            for rep in range(cfg['repeats']):
                for cell in range(gr*gc):
                    if task_index % a.worker_count==a.worker_index:
                        write_shard(f'{cond["id"]}__r{rep:03d}__g{cell:03d}',cond,rep,cell)
                    task_index+=1
    print('Finished:',out)

if __name__=='__main__':main()
