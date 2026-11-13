"""Read-only evidence preparation for the requested DOCX rewrite."""
from pathlib import Path
from zipfile import ZipFile
from collections import defaultdict,Counter
import json,sys,contextlib,io
import numpy as np,pandas as pd
from lxml import etree as ET
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from core import digest_file,dump_json,load_jsonl,operational_metrics,match_predictions
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
OUT=ROOT/'reports/progress_report_reorganized_2026-11-12';OUT.mkdir(parents=True,exist_ok=True)
SOURCE=Path('/home/hyochang/pilot_progress_report.docx');ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
with ZipFile(SOURCE) as z:
 body=ET.fromstring(z.read('word/document.xml')).find('w:body',ns);blocks=[]
 for node in body:
  tag=ET.QName(node).localname
  if tag=='p':
   text=''.join(node.xpath('.//w:t/text()',namespaces=ns));style=node.xpath('./w:pPr/w:pStyle/@w:val',namespaces=ns)
   if text:blocks.append({'kind':'paragraph','style':style,'text':text})
  elif tag=='tbl':blocks.append({'kind':'table','rows':[[''.join(c.xpath('.//w:t/text()',namespaces=ns)) for c in row.findall('w:tc',ns)] for row in node.findall('w:tr',ns)]})
dump_json(OUT/'reference_document_extract.json',{'source':str(SOURCE),'sha256':digest_file(SOURCE),'blocks':blocks})
# Audit the stricter middle filename component B identified by the reference document.
records=load_jsonl(ROOT/'data/canonical/gc10_det/manifest.jsonl');groups=defaultdict(set);views=defaultdict(set);bc=defaultdict(set);ab=defaultdict(set)
for r in records:
 parts=r['image_id'].split('_');a,b,c=parts[1:4];groups[b].add(r['split']);views[b].add(a);bc[b,c].add(a);ab[a,b].add(r['split'])
audit={'n_images':len(records),'n_B_groups':len(groups),'B_cross_split_groups':sum(len(x)>1 for x in groups.values()),'B_train_test_overlap':sum({'train','test'}<=x for x in groups.values()),'B_multi_camera_groups':sum(len(x)>1 for x in views.values()),'BC_multi_camera_pairs':sum(len(x)>1 for x in bc.values()),'AB_cross_split_groups':sum(len(x)>1 for x in ab.values()),'B_physical_meaning':'unconfirmed proxy, interpreted as coil-like only in reference document','manifest_sha256':digest_file(ROOT/'data/canonical/gc10_det/manifest.jsonl')}
dump_json(OUT/'gc10_B_group_audit.json',audit)
pd.DataFrame([{'B':b,'splits':','.join(sorted(ss)),'n_cameras':len(views[b]),'train_test_overlap':{'train','test'}<=ss} for b,ss in sorted(groups.items())]).to_csv(OUT/'gc10_B_groups.csv',index=False)
# Check the actual format discrepancy reported for this PCB image.
p=ROOT/'data/raw/pcb_ind/extracted';raw_label=(p/'YOLO/labels/val/5799_t_038.txt').read_text();j=json.loads((p/'COCO/annotations/val.json').read_text());im=next(x for x in j['images'] if Path(x['file_name']).stem=='5799_t_038');ann=[x for x in j['annotations'] if x['image_id']==im['id']]
dump_json(OUT/'pcb_annotation_format_audit.json',{'image_id':'5799_t_038','raw_yolo_lines':raw_label.splitlines(),'raw_yolo_n':len(raw_label.splitlines()),'raw_yolo_zero_height_n':sum(float(x.split()[4])==0 for x in raw_label.splitlines()),'coco_valid_annotations':ann,'coco_n':len(ann),'current_canonical_policy':'uses official COCO, four positive-area boxes, no zero-height box'})
# Recompute only CLEAN per-class metrics (no GPU); exact same COCO settings as existing runs.
rows=[]
for d,label in [('gc10_det','GC10-DET'),('neu_det','NEU-DET'),('pcb_ind_board','PCB-IND 새 분할')]:
 run=ROOT/f'runs/research_validation_compiled_20261111/{d}_s11__bright_k25';clean=load_jsonl(run/'predictions/clean.jsonl');names=json.loads((ROOT/f'data/canonical/{d}/classes.json').read_text());base=json.loads((run/'analysis/clean_metrics.json').read_text());images=[];anns=[];dets=[];aid=1
 for iid,r in enumerate(clean,1):
  images.append({'id':iid,'width':r['width'],'height':r['height'],'file_name':r['image_id']})
  for g in r['gt']:
   x0,y0,x1,y1=g['bbox_xyxy'];anns.append({'id':aid,'image_id':iid,'category_id':g['class_id']+1,'bbox':[x0,y0,x1-x0,y1-y0],'area':(x1-x0)*(y1-y0),'iscrowd':0});aid+=1
  for pr in r['pred']:
   x0,y0,x1,y1=pr['bbox_xyxy'];dets.append({'image_id':iid,'category_id':pr['class_id']+1,'bbox':[x0,y0,max(0,x1-x0),max(0,y1-y0)],'score':pr['score']})
 with contextlib.redirect_stdout(io.StringIO()):
  gt=COCO();gt.dataset={'images':images,'annotations':anns,'categories':[{'id':i+1,'name':n} for i,n in enumerate(names)],'info':{}};gt.createIndex();dt=gt.loadRes(dets);ev=COCOeval(gt,dt,'bbox');ev.params.imgIds=[i['id'] for i in images];ev.params.maxDets=[1,10,100];ev.evaluate();ev.accumulate();ev.summarize()
 assert np.isclose(ev.stats[0],base['coco_map50_95'],atol=1e-12) and np.isclose(ev.stats[1],base['coco_map50'],atol=1e-12)
 pp=ev.eval['precision']
 for k,name in enumerate(names):
  filtered=[{'gt':[g for g in r['gt'] if g['class_id']==k],'pred':[p for p in r['pred'] if p['class_id']==k]} for r in clean];m=operational_metrics(filtered,.25,.5)
  a=pp[:,:,k,0,2];a50=pp[0,:,k,0,2];ap=float(a[a>=0].mean());ap50=float(a50[a50>=0].mean())
  rows.append({'dataset':d,'dataset_label':label,'class_id':k,'class_name':name,'n_gt':m['n_gt'],'tp':m['tp'],'fp':m['fp'],'fn':m['fn'],'recall_pct':100*m['recall'],'precision_pct':100*m['precision'] if m['precision'] is not None else None,'coco_map50_pct':100*ap50,'coco_map50_95_pct':100*ap})
pd.DataFrame(rows).to_csv(OUT/'clean_per_class_seed11.csv',index=False)
# Bridge conditional loss vs net Recall drop using already validated object-event aggregates.
old=pd.read_csv(ROOT/'reports/research_validation_2026-11-11/distribution/dataset_summary.csv');new=pd.read_csv(ROOT/'reports/research_validation_2026-11-11/corrected_pcb_distribution/dataset_summary.csv');bridge=pd.concat([old[old.dataset.isin(['gc10_det','neu_det'])],new],ignore_index=True)
bridge['clean_tp_observation_denominator']=bridge.n_clean_tp*640;bridge['miss_transition_pct']=100*bridge.overall_lost_object_observations/bridge.clean_tp_observation_denominator
bridge['net_recall_drop_check_pp']=100*(bridge.overall_lost_object_observations-bridge.overall_recovered_object_observations)/bridge.n_object_observations
assert np.allclose(bridge.mean_recall_drop_pp,bridge.net_recall_drop_check_pp,atol=1e-10)
bridge.to_csv(OUT/'metric_definition_bridge.csv',index=False)
# Use the SAME dispersion definition for raw/residual comparisons.
res=[]
for d,folder in [('gc10_det','distribution'),('neu_det','distribution'),('pcb_ind_board','corrected_pcb_distribution')]:
 c=pd.read_csv(ROOT/f'reports/research_validation_2026-11-11/{folder}/{d}_cell_exposure.csv')
 res.append({'dataset':d,'raw_spatial_sd_pp':float(c.recall_drop_pp.std(ddof=0)),'residual_spatial_sd_pp':float(c.descriptive_residual_drop_pp.std(ddof=0)),'residual_spatial_rms_pp':float(np.sqrt((c.descriptive_residual_drop_pp**2).mean()))})
pd.DataFrame(res).to_csv(OUT/'descriptive_spatial_dispersion.csv',index=False)
dump_json(OUT/'evidence_preparation.json',{'reference_docx_sha256':digest_file(SOURCE),'gc10_B_audit':audit,'current_series_root':str(ROOT),'new_inference_or_training':False,'perclass_ap_clean_only_recomputed_and_matches_saved_overall':True})
print('Evidence and new read-only audits saved:',OUT)
