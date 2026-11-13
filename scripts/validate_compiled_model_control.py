"""Check direct FP32 execution on each trained checkpoint before study inference."""
from pathlib import Path
import argparse,sys,json
import numpy as np,cv2,torch,yaml
from ultralytics import YOLO
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from core import load_jsonl,stable_seed,fault_pattern,cell_bounds,inject_fault,dump_json,digest_file
from run import load_clean
from run_compiled import predict_packed
from safe_compiled_nms import reference_nms,guarded_nms,get_stats
p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--weights',required=True);p.add_argument('--pattern-key',required=True);p.add_argument('--device',default='0');p.add_argument('--out',required=True);a=p.parse_args()
torch.set_num_threads(2);cv2.setNumThreads(1);rows=sorted([r for r in load_jsonl(ROOT/f'data/canonical/{a.dataset}/manifest.jsonl') if r['split']=='val'],key=lambda r:r['image_id']);ix=np.random.default_rng(20261111).permutation(len(rows))[:17];rows=[rows[i] for i in sorted(ix)];images=[load_clean(r) for r in rows];cfg=yaml.safe_load((ROOT/'configs/research_validation/all_conditions.yaml').read_text());model=YOLO(a.weights);pcfg=cfg['predict'];checks=[]
for c in [None]+cfg['conditions']:
 bgrs=[]
 for row,im in zip(rows,images):
  rgb=im
  if c is not None:
   h,w=im.shape[:2];seed=stable_seed('pattern-v1',a.pattern_key,h,w,8,8,c['layout'],c['side'],301,0);off=fault_pattern(h,w,8,8,c['count'],c['layout'],c['side'],seed);rgb,_=inject_fault(im,off,cell_bounds(h,w,8,8,36),c['target'],c['alpha'],'all')
  bgrs.append(np.ascontiguousarray(rgb[...,::-1]))
 ref=[];fast=[]
 for start in range(0,len(bgrs),16):
  batch=bgrs[start:start+16]
  with reference_nms():rr=model.predict(source=batch,imgsz=640,conf=.001,iou=.7,max_det=300,rect=False,half=False,augment=False,agnostic_nms=False,device=a.device,save=False,verbose=False)
  for r in rr:
   ps=[]
   if r.boxes is not None:
    for box,sc,cl in zip(r.boxes.xyxy.cpu().numpy(),r.boxes.conf.cpu().numpy(),r.boxes.cls.cpu().numpy()):ps.append({'class_id':int(cl),'score':float(sc),'bbox_xyxy':[float(v) for v in box]})
   ref.append(ps)
  with guarded_nms():fast+=predict_packed(model,batch,pcfg,a.device)
 if ref!=fast:raise ValueError(f'Checkpoint control predictions differ: {a.dataset} {c}')
 checks.append({'condition':'clean' if c is None else c['id'],'n_images':len(rows),'exact_predictions':True})
fingerprints={'dataset':a.dataset,'pattern_key':a.pattern_key,'weights_sha256':digest_file(a.weights),'manifest_sha256':digest_file(ROOT/f'data/canonical/{a.dataset}/manifest.jsonl'),'classes_sha256':digest_file(ROOT/f'data/canonical/{a.dataset}/classes.json'),'condition_config_sha256':digest_file(ROOT/'configs/research_validation/all_conditions.yaml'),'control_script_sha256':digest_file(__file__),'source_sha256':{name:digest_file(ROOT/'src'/name) for name in ['run_compiled.py','safe_compiled_nms.py','run_fast.py','run.py','core.py']}}
dump_json(a.out,{'fingerprints':fingerprints,'all_pass':True,'dataset':a.dataset,'weights_sha256':digest_file(a.weights),'runner_sha256':digest_file(ROOT/'src/run_compiled.py'),'nms_sha256':digest_file(ROOT/'src/safe_compiled_nms.py'),'nms_stats':get_stats(),'selected_validation_ids':[r['image_id'] for r in rows],'checks':checks})
print('Checkpoint controls exact',a.dataset,a.weights,flush=True)
