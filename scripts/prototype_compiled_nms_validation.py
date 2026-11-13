"""One-process isolated compiled-NMS probe; production imports and sources unchanged."""
from pathlib import Path
from contextlib import contextmanager
from collections import Counter
import os,json,sys,time,statistics,traceback
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('MKL_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import cv2,numpy as np,torch,torchvision
from ultralytics import YOLO
from ultralytics.utils.nms import TorchNMS
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from run import load_clean
from run_fast import predict_packed,pack_predictions,NmsDeadlineGuard
from core import load_jsonl,stable_seed,fault_pattern,cell_bounds,inject_fault,digest_file,match_predictions,dump_json
OUT=ROOT/'reports/research_validation_2026-11-11/compiled_nms_prototype';OUT.mkdir(parents=True,exist_ok=True)
ORIGINAL_COMPILED=torchvision.ops.nms;COUNTS=Counter();CANDIDATE_N=[]
PRED={'imgsz':640,'conf_floor':.001,'nms_iou':.7,'max_det':300};DOC='https://docs.pytorch.org/vision/0.24/generated/torchvision.ops.nms.html'
summary={'status':'starting','production_integration':False,'source_sha256':digest_file(__file__),'runner_sha256':digest_file(ROOT/'src/run_fast.py'),'official_docs':DOC,'gpu':0,'allocator_cap_gib':3,'shared_gpu_caveat':'Production contexts remain active; alternating warmed timings are indicative, not an isolated speed guarantee.'}


@contextmanager
def override_nms(fn):
 previous=torchvision.ops.nms;torchvision.ops.nms=fn
 try:yield
 finally:torchvision.ops.nms=previous


def rank_only(boxes,scores,threshold):
 if not boxes.numel():return TorchNMS.nms(boxes,scores,threshold)
 order=scores.argsort(0,descending=True)
 rank=torch.arange(len(order),0,-1,device=scores.device,dtype=scores.dtype)
 keep=ORIGINAL_COMPILED(boxes[order],rank,threshold)
 return order[keep]


def guarded(boxes,scores,threshold):
 n=len(boxes);CANDIDATE_N.append(n)
 if n==0:COUNTS['empty']+=1;return TorchNMS.nms(boxes,scores,threshold)
 if n>2048 or boxes.dtype!=torch.float32 or scores.dtype!=torch.float32:COUNTS['large_or_dtype']+=1;return TorchNMS.nms(boxes,scores,threshold)
 wh=boxes[:,2:]-boxes[:,:2];area=wh[:,0]*wh[:,1]
 if not bool(torch.isfinite(boxes).all() and torch.isfinite(scores).all() and torch.isfinite(area).all() and (wh>0).all()):
  COUNTS['invalid_or_degenerate']+=1;return TorchNMS.nms(boxes,scores,threshold)
 lt=torch.maximum(boxes[:,None,:2],boxes[None,:,:2]);rb=torch.minimum(boxes[:,None,2:],boxes[None,:,2:]);inter=(rb-lt).clamp_(min=0).prod(dim=2)
 union=area[:,None]+area[None,:]-inter;iou=inter/union
 if not bool(torch.isfinite(iou).all()):COUNTS['nonfinite_iou']+=1;return TorchNMS.nms(boxes,scores,threshold)
 near=torch.triu((iou-float(threshold)).abs()<=1e-5,diagonal=1)
 if bool(near.any()):COUNTS['near_threshold']+=1;return TorchNMS.nms(boxes,scores,threshold)
 COUNTS['compiled_rank_preserved']+=1;return rank_only(boxes,scores,threshold)


def main():
 torch.set_num_threads(2);cv2.setNumThreads(1);torch.cuda.set_device(0)
 total=torch.cuda.get_device_properties(0).total_memory;torch.cuda.set_per_process_memory_fraction(3*1024**3/total,device=0)
 # Include tied scores, exact-threshold overlap, degenerate boxes and translation.
 synthetic=[]
 cases=[('ties',[[0,0,10,10],[1,1,11,11],[20,20,25,25],[0,0,10,10]],[.5,.5,.5,.3],.7),
 ('near_threshold',[[0,0,10,10],[0,0,7,10]],[.8,.7],.7),('degenerate',[[0,0,0,4],[1,1,3,3]],[.8,.7],.7),
 ('negative_coordinates',[[-20,-20,-10,-10],[-19,-19,-9,-9],[20,20,25,25]],[.5,.5,.4],.7)]
 for name,b,s,t in cases:
  boxes=torch.tensor(b,device='cuda:0',dtype=torch.float32);scores=torch.tensor(s,device='cuda:0',dtype=torch.float32);bc=boxes.clone();sc=scores.clone()
  ref=TorchNMS.nms(boxes,scores,t);got=guarded(boxes,scores,t)
  if not torch.equal(ref,got) or not torch.equal(bc,boxes) or not torch.equal(sc,scores):raise RuntimeError('Synthetic equivalence/input mutation failure: '+name)
  synthetic.append({'name':name,'exact_indices':True,'kept':ref.cpu().tolist()})
 model=YOLO(json.loads((ROOT/'runs/evaluation/gc10_s11_frozen/run_meta.json').read_text())['weights_path'])
 rows=sorted([r for r in load_jsonl(ROOT/'data/canonical/gc10_det/manifest.jsonl') if r['split']=='val'],key=lambda r:r['image_id'])
 ix=sorted(np.random.default_rng(20261111).permutation(len(rows))[:32]);rows=[rows[i] for i in ix];clean=[np.ascontiguousarray(load_clean(r)[...,::-1]) for r in rows];controls={'clean':clean}
 for name,target,layout,side in [('hot25',255,'dispersed',1),('dark25',0,'dispersed',1),('cluster25',255,'cluster',5)]:
  arrays=[]
  for row,im in zip(rows,clean):
   h,w=im.shape[:2];seed=stable_seed('pattern-v1','gc10_det',h,w,8,8,layout,side,301,0);off=fault_pattern(h,w,8,8,25,layout,side,seed)
   bad,_=inject_fault(im,off,cell_bounds(h,w,8,8,36),target,1.,'all');arrays.append(bad)
  controls[name]=arrays
 modes={'reference_pure_torch':TorchNMS.nms,'guarded_rank_compiled':guarded,'rank_compiled_diagnostic':rank_only}
 def infer(arrays,mode):
  predictions=[]
  with override_nms(modes[mode]):
   for offset in range(0,len(arrays),16):predictions.extend(predict_packed(model,arrays[offset:offset+16],PRED,'0'))
  if len(predictions)!=len(arrays):raise RuntimeError('Prediction image count changed')
  return predictions
 checks=[]
 with torch.inference_mode():
  infer(clean[:16],'reference_pure_torch')
  sequence=[('clean',32),('hot25',32),('dark25',32),('cluster25',32),('clean',17),('hot25',32),('dark25',17),('cluster25',32),('clean',32)]
  for name,n in sequence:
   arrays=controls[name][:n];ref=infer(arrays,'reference_pure_torch')
   for mode in ['guarded_rank_compiled','rank_compiled_diagnostic']:
    got=infer(arrays,mode);exact=ref==got;per_image=all(all(match_predictions(row['gt'],r,.25,.5)[k]==match_predictions(row['gt'],g,.25,.5)[k] for k in ['tp','fp','fn','gt_matched']) for row,r,g in zip(rows[:n],ref,got))
    record={'condition':name,'n_images':n,'partial_last_batch':n%16,'mode':mode,'exact_prediction_dicts':exact,'per_image_decisions_equal':per_image,'reference_boxes':sum(map(len,ref)),'candidate_boxes':sum(map(len,got))};checks.append(record);print(json.dumps(record),flush=True)
  predictor=model.predictor
  def stages(arrays,mode):
   predictor.batch=([f'proto_{i}.jpg' for i in range(len(arrays))],arrays,['']*len(arrays));torch.cuda.synchronize(0);start=time.perf_counter()
   tensor=predictor.preprocess(arrays);torch.cuda.synchronize(0);after_pre=time.perf_counter();raw=predictor.inference(tensor);torch.cuda.synchronize(0);after_forward=time.perf_counter()
   with override_nms(modes[mode]):out=pack_predictions(predictor.postprocess(raw,tensor,arrays))
   torch.cuda.synchronize(0);end=time.perf_counter()
   return {'whole_batch_ms':(end-start)*1000,'preprocess_ms':(after_pre-start)*1000,'forward_ms':(after_forward-after_pre)*1000,'postprocess_pack_ms':(end-after_forward)*1000}
  # Warm each execution path, then alternate order across fixed changing controls.
  for mode in modes:infer(clean[:16],mode)
  timing=[];order=list(modes)
  for trial in range(8):
   name=list(controls)[trial%4];arrays=controls[name][:16];rotation=trial%3;trial_order=order[rotation:]+order[:rotation]
   for mode in trial_order:timing.append({'trial':trial,'condition':name,'mode':mode,**stages(arrays,mode)})
 summary.update(status='complete',synthetic=synthetic,checks=checks,guard_counts=dict(COUNTS),candidate_n_summary={'n_calls':len(CANDIDATE_N),'min':min(CANDIDATE_N),'median':statistics.median(CANDIDATE_N),'max':max(CANDIDATE_N)},
  guarded_exact=all(c['exact_prediction_dicts'] for c in checks if c['mode']=='guarded_rank_compiled'),diagnostic_exact=all(c['exact_prediction_dicts'] for c in checks if c['mode']=='rank_compiled_diagnostic'),timings=timing,
  peak_allocated_gib=torch.cuda.max_memory_allocated(0)/1024**3,peak_reserved_gib=torch.cuda.max_memory_reserved(0)/1024**3)
 summary['median_ms']={mode:{key:statistics.median(t[key] for t in timing if t['mode']==mode) for key in ['whole_batch_ms','preprocess_ms','forward_ms','postprocess_pack_ms']} for mode in modes}
 dump_json(OUT/'result.json',summary)
 med=summary['median_ms'];table='| 경로 | 전체 batch | 전처리 | forward | 후처리+pack |\n|---|---:|---:|---:|---:|\n'
 for mode,v in med.items():table+='| '+mode+' | '+' | '.join(f'{v[k]:.3f}' for k in ['whole_batch_ms','preprocess_ms','forward_ms','postprocess_pack_ms'])+' |\n'
 (OUT/'REPORT_KO.md').write_text(f'''# Compiled NMS 격리 프로토타입\n\n운영 코드와 프로세스는 변경하지 않았다. 별도GPU0프로세스에서3GiB allocator상한으로 검사했다. torchvision이 import된 경우에만 compiled경로를 타는 고정Ultralytics분기를 이용했으며, 기준 경로는 torchvision.ops.nms를 원래TorchNMS.nms로 명시적으로 치환했다.\n\n기존 scores.argsort내림차순 순서를 먼저 고정하고 compiled NMS에는 엄격히 감소하는 정수 rank score만 전달한 후 원래인덱스로 복원한다. 원본 confidence값은 변경하지 않는다. 후보는 N>2048·비정상/퇴화상자·비유한값 또는 임의 쌍의IoU가 기준±1e-5이면 원래TorchNMS로 돌아간다.\n\n- 보수적 후보 전체 예측사전 완전일치: {summary['guarded_exact']}\n- near-IoU보호가 없는 진단 경로 완전일치: {summary['diagnostic_exact']}\n- guard분기 집계: {summary['guard_counts']}\n- 입력 후보수: {summary['candidate_n_summary']}\n- Peak allocated/reserved: {summary['peak_allocated_gib']:.3f}/{summary['peak_reserved_gib']:.3f}GiB\n\n고정32검증영상의clean/hot25/dark25/cluster25와17장부분배치, 부분배치후재실행을 검사했다. 동점점수·정확한IoU경계·퇴화·음수좌표의 합성 인덱스 검사도 통과했다.\n\n단위ms, 워밍업 후 입력조건과 경로순서를 교대한8회 중앙값:\n\n{table}\n\n현재GPU는 운영작업과 공유하므로 격리속도비나 모든 데이터/가중치에 대한 정확성 보장이 아니다. [torchvision0.24공식문서]({DOC})는 score동점의선택비보장과IoU기준을 설명한다. rank처리와경계fallback은 그차이를 보수적으로 다루기위한 제한된 프로토타입이며 운영에 통합하지 않았다.\n''')
 print(json.dumps({k:v for k,v in summary.items() if k not in ['checks','timings']},indent=2),flush=True)

if __name__=='__main__':
 try:main()
 except Exception as exc:
  summary.update(status='error_or_unsupported',error=str(exc),traceback=traceback.format_exc());dump_json(OUT/'result.json',summary);(OUT/'REPORT_KO.md').write_text('# Compiled NMS 격리 프로토타입\n\n오류로중단했다. 운영코드는 변경하지 않았다.\n\n```text\n'+summary['traceback']+'\n```\n');print(summary['traceback'],file=sys.stderr);raise SystemExit(1)
