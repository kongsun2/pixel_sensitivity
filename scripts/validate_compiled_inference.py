"""Validate execution-only changes on fixed validation controls, never choose for accuracy."""
from pathlib import Path
from functools import lru_cache
import argparse,sys,json,time,os
import cv2,numpy as np,yaml,torch
from ultralytics import YOLO
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from run import load_clean
from run_compiled import pack_predictions,predict_packed
from safe_compiled_nms import reference_nms,guarded_nms,get_stats
from core import load_jsonl,stable_seed,fault_pattern,cell_bounds,inject_fault,operational_metrics,match_predictions,dump_json,digest_file
from coco_metrics import coco_ap
p=argparse.ArgumentParser();p.add_argument('--device',default='0');a=p.parse_args();torch.set_num_threads(4);cv2.setNumThreads(1)
OUT=ROOT/'reports/research_validation_2026-11-11';OUT.mkdir(exist_ok=True,parents=True)
DATA=[('gc10_det','gc10_s11_frozen'),('neu_det','neu_det_s11_bright'),('pcb_ind','pcb_ind_s11_bright')]
CFG=yaml.safe_load((ROOT/'configs/research_validation/all_conditions.yaml').read_text());pattern=lru_cache(maxsize=200)(fault_pattern)
results={};accepted=[]
for batch in [16]:
 all_pass=True;checks=[]
 for d,old in DATA:
  meta=json.loads((ROOT/f'runs/evaluation/{old}/run_meta.json').read_text());model=YOLO(meta['weights_path']);names=json.loads((ROOT/f'data/canonical/{d}/classes.json').read_text())
  rows=sorted([r for r in load_jsonl(ROOT/f'data/canonical/{d}/manifest.jsonl') if r['split']=='val'],key=lambda r:r['image_id'])
  # Fixed sample includes a partial final128batch; selection is independent of outcomes.
  indices=np.random.default_rng(20261111).permutation(len(rows))[:130];rows=[rows[i] for i in sorted(indices)];images=[load_clean(r) for r in rows]
  def infer(arrays,bs,fast):
   pred=[];start=time.perf_counter()
   for offset in range(0,len(arrays),bs):
    selected=arrays[offset:offset+bs];bgr=[np.ascontiguousarray(x[...,::-1]) for x in selected]
    if fast:
     with guarded_nms():pred.extend(predict_packed(model,bgr,{'imgsz':640,'conf_floor':.001,'nms_iou':.7,'max_det':300},a.device))
    else:
     with reference_nms():rr=model.predict(source=bgr,imgsz=640,conf=.001,iou=.7,max_det=300,rect=False,half=False,augment=False,agnostic_nms=False,device=a.device,save=False,verbose=False)
     for r in rr:
      ps=[]
      if r.boxes is not None:
       for box,score,cl in zip(r.boxes.xyxy.cpu().numpy(),r.boxes.conf.cpu().numpy(),r.boxes.cls.cpu().numpy()):ps.append({'class_id':int(cl),'score':float(score),'bbox_xyxy':[float(v) for v in box]})
      pred.append(ps)
   return pred,time.perf_counter()-start
  for c in [None]+CFG['conditions']:
   imgs=[]
   for row,im in zip(rows,images):
    if c is None:imgs.append(im);continue
    h,w=im.shape[:2];seed=stable_seed('pattern-v1',d,h,w,8,8,c['layout'],c['side'],301,0);off=pattern(h,w,8,8,c['count'],c['layout'],c['side'],seed)
    rgb,rm=inject_fault(im,off,cell_bounds(h,w,8,8,36),c['target'],c['alpha'],'all')
    bgr,bm=inject_fault(np.ascontiguousarray(im[...,::-1]),off,cell_bounds(h,w,8,8,36),c['target'],c['alpha'],'all')
    if not np.array_equal(rgb[...,::-1],bgr) or rm!=bm:raise ValueError('BGR injection equivalence failed')
    imgs.append(rgb)
   key=(d,'clean' if c is None else c['id'])
   # Reference tensors use original batch16 and original CPU extraction.
   if key not in results:results[key]=infer(imgs,16,False)
   ref,refsecs=results[key];got,secs=infer(imgs,batch,True)
   rrec=[{**row,'pred':pred} for row,pred in zip(rows,ref)];grec=[{**row,'pred':pred} for row,pred in zip(rows,got)]
   rm=operational_metrics(rrec,.25,.5);gm=operational_metrics(grec,.25,.5);ra=coco_ap(rrec,names);ga=coco_ap(grec,names)
   operational_equal=all(rm[k]==gm[k] for k in ['tp','fp','fn']);apdiff=max(abs(ra[k]-ga[k]) for k in ['coco_map50','coco_map50_95'])
   per_image_equal=all(all(match_predictions(r['gt'],r['pred'],.25,.5)[k]==match_predictions(g['gt'],g['pred'],.25,.5)[k] for k in ['tp','fp','fn','gt_matched']) for r,g in zip(rrec,grec))
   exact_predictions=(ref==got)
   ok=operational_equal and per_image_equal and exact_predictions and apdiff<=1e-6;all_pass&=ok
   checks.append({'dataset':d,'condition':key[1],'n_images':len(rows),'batch':batch,'operational_equal':operational_equal,'per_image_matches_equal':per_image_equal,'exact_predictions':exact_predictions,'ap_max_abs_difference':apdiff,'accepted':ok,'reference_seconds':refsecs,'candidate_seconds':secs})
   print(json.dumps(checks[-1]),flush=True)
  del model;torch.cuda.empty_cache()
  if not all_pass:break
 dump_json(OUT/f'compiled_validation_batch{batch}.json',{'criteria':'Exact TP/FP/FN and COCO AP absolute difference <=1e-6 on fixed validation controls; RGB/BGR injected pixels+metadata exact','device':a.device,'checks':checks,'all_pass':all_pass,'source_sha256':digest_file(__file__),'runner_sha256':digest_file(ROOT/'src/run_compiled.py'),'nms_sha256':digest_file(ROOT/'src/safe_compiled_nms.py'),'reference_fast_runner_sha256':digest_file(ROOT/'src/run_fast.py'),'nms_stats':get_stats()})
 if all_pass:
  accepted.append(batch);dump_json(OUT/'compiled_backend_validation.json',{'selected_batch':batch,'all_pass':True,'checks':checks,'criteria':'Exact TP/FP/FN, abs AP diff<=1e-6; validation controls only; original batch16 preserved; all prediction dicts and per-image decisions exact on fixed controls; no model accuracy selection','runner_sha256':digest_file(ROOT/'src/run_compiled.py'),'nms_sha256':digest_file(ROOT/'src/safe_compiled_nms.py'),'reference_fast_runner_sha256':digest_file(ROOT/'src/run_fast.py'),'nms_stats':get_stats(),'reference_runner_sha256':digest_file(ROOT/'src/run.py')});break
if not accepted:raise RuntimeError('No validated execution batch')
print('Validated execution batch',accepted[0],flush=True)
