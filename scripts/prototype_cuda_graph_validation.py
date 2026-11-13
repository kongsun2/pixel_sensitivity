"""Isolated CUDA graph feasibility probe; never imported by production runners."""
from pathlib import Path
from functools import lru_cache
import hashlib,json,os,statistics,sys,time,traceback,logging
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('MKL_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import cv2,numpy as np,torch
from ultralytics import YOLO
from ultralytics.utils import LOGGER
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from run import load_clean
from run_fast import predict_packed,pack_predictions
from core import load_jsonl,stable_seed,fault_pattern,cell_bounds,inject_fault,digest_file,match_predictions,dump_json
OUT=ROOT/'reports/research_validation_2026-11-11/graph_prototype';OUT.mkdir(parents=True,exist_ok=True)
DOC='https://docs.pytorch.org/docs/2.9/notes/cuda.html#cuda-graphs'
PRED={'imgsz':640,'conf_floor':.001,'nms_iou':.7,'max_det':300}
summary={'status':'starting','production_integration':False,'source_sha256':digest_file(__file__),'reference_source_sha256':digest_file(ROOT/'src/run_fast.py'),'official_docs':DOC,'gpu':0,'batch':16,'allocator_cap_gib':3,'shared_gpu_caveat':'Production inference contexts run simultaneously; timings are warmed alternating observations, not an isolated speed benchmark.'}
class NmsWarnings(logging.Handler):
 def emit(self,r):
  if 'NMS time limit' in r.getMessage():raise RuntimeError(r.getMessage())
LOGGER.addHandler(NmsWarnings())


def raw_main(raw):return raw[0] if isinstance(raw,(list,tuple)) else raw


def nms_copy(raw):
 if isinstance(raw,tuple):return (raw[0].clone(),*raw[1:])
 if isinstance(raw,list):return [raw[0].clone(),*raw[1:]]
 return raw.clone()


def anchor_digest(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def main():
 torch.set_num_threads(2);cv2.setNumThreads(1);torch.cuda.set_device(0)
 total=torch.cuda.get_device_properties(0).total_memory;torch.cuda.set_per_process_memory_fraction(3*1024**3/total,device=0)
 meta=json.loads((ROOT/'runs/evaluation/gc10_s11_frozen/run_meta.json').read_text());weights=meta['weights_path'];model=YOLO(weights)
 allrows=sorted([r for r in load_jsonl(ROOT/'data/canonical/gc10_det/manifest.jsonl') if r['split']=='val'],key=lambda r:r['image_id'])
 ix=sorted(np.random.default_rng(20261111).permutation(len(allrows))[:32]);rows=[allrows[i] for i in ix];clean=[np.ascontiguousarray(load_clean(r)[...,::-1]) for r in rows]
 controls={'clean':clean}
 for name,target,layout,side in [('hot25',255,'dispersed',1),('dark25',0,'dispersed',1),('cluster25',255,'cluster',5)]:
  arrays=[]
  for row,im in zip(rows,clean):
   h,w=im.shape[:2];seed=stable_seed('pattern-v1','gc10_det',h,w,8,8,layout,side,301,0)
   off=fault_pattern(h,w,8,8,25,layout,side,seed);bad,_=inject_fault(im,off,cell_bounds(h,w,8,8,36),target,1.,'all');arrays.append(bad)
  controls[name]=arrays
 # Pinned predictor owns all preprocessing and NMS. Capture only its fixed-shape forward.
 with torch.inference_mode():
  predict_packed(model,clean[:16],PRED,'0');predictor=model.predictor
  static_input=predictor.preprocess(clean[:16]);side_stream=torch.cuda.Stream(device=0);side_stream.wait_stream(torch.cuda.current_stream(0))
  with torch.cuda.stream(side_stream):
   for _ in range(3):warm_output=predictor.inference(static_input)
  torch.cuda.current_stream(0).wait_stream(side_stream);torch.cuda.synchronize(0)
  anchor_refs=[]
  for module in predictor.model.modules():
   for attr in ['anchors','strides']:
    t=getattr(module,attr,None)
    if isinstance(t,torch.Tensor) and t.is_cuda and t.numel():anchor_refs.append((type(module).__name__,attr,t,anchor_digest(t)))
  graph=torch.cuda.CUDAGraph()
  with torch.cuda.graph(graph):captured_output=predictor.inference(static_input)
  torch.cuda.synchronize(0)
  if torch.cuda.max_memory_reserved(0)>3*1024**3:raise RuntimeError('Allocator cap exceeded')
  def eager(arrays):
   result=[]
   for i in range(0,len(arrays),16):result.extend(predict_packed(model,arrays[i:i+16],PRED,'0'))
   return result
  def graphed(arrays):
   result=[]
   for i in range(0,len(arrays),16):
    part=arrays[i:i+16]
    if len(part)!=16:
     result.extend(predict_packed(model,part,PRED,'0'));continue
    predictor.batch=([f'prototype_{j}.jpg' for j in range(16)],part,['']*16)
    tensor=predictor.preprocess(part);static_input.copy_(tensor);graph.replay();torch.cuda.synchronize(0)
    # NMS may modify xywh predictions in-place: never let it edit retained captured output.
    result.extend(pack_predictions(predictor.postprocess(nms_copy(captured_output),tensor,part)))
   return result
  checks=[]
  sequence=[('clean',32),('hot25',32),('dark25',32),('cluster25',32),('clean',17),('hot25',32),('dark25',17),('cluster25',32),('clean',32)]
  for name,n in sequence:
   arrays=controls[name][:n];reference=eager(arrays);candidate=graphed(arrays)
   if len(reference)!=n or len(candidate)!=n:raise RuntimeError('Missing prediction images')
   exact=reference==candidate;per_image=True
   for row,r,g in zip(rows[:n],reference,candidate):
    rm=match_predictions(row['gt'],r,.25,.5);gm=match_predictions(row['gt'],g,.25,.5)
    per_image &= all(rm[k]==gm[k] for k in ['tp','fp','fn','gt_matched'])
   checks.append({'condition':name,'n_images':n,'partial_eager_fallback':n%16,'exact_prediction_dicts':exact,'per_image_operational_equal':per_image,'n_reference_boxes':sum(map(len,reference)),'n_candidate_boxes':sum(map(len,candidate))})
   print(json.dumps(checks[-1]),flush=True)
  # Full-sized graph constants must remain alive even after eager partial-batch anchor recaching.
  anchors_unchanged=all(anchor_digest(t)==h for _,_,t,h in anchor_refs)
  timings=[]
  names=list(controls)
  for trial in range(4):
   name=names[trial%4];arrays=controls[name][:16]
   for mode in (['eager','graph'] if trial%2==0 else ['graph','eager']):
    torch.cuda.synchronize(0);start=time.perf_counter();eager(arrays) if mode=='eager' else graphed(arrays);torch.cuda.synchronize(0)
    timings.append({'scope':'whole_batch','trial':trial,'condition':name,'mode':mode,'wall_ms':(time.perf_counter()-start)*1000})
  # Forward-only includes mandatory static-input copy for graph mode.
  tensor=predictor.preprocess(clean[:16]);predictor.batch=([f'prototype_{i}.jpg' for i in range(16)],clean[:16],['']*16);predictor.inference(tensor);torch.cuda.synchronize(0)
  for trial in range(8):
   for mode in (['eager','graph'] if trial%2==0 else ['graph','eager']):
    start_event=torch.cuda.Event(enable_timing=True);end_event=torch.cuda.Event(enable_timing=True);torch.cuda.synchronize(0);start=time.perf_counter();start_event.record()
    if mode=='eager':raw=predictor.inference(tensor)
    else:static_input.copy_(tensor);graph.replay()
    end_event.record();end_event.synchronize()
    timings.append({'scope':'forward_with_input_copy_for_graph','trial':trial,'condition':'clean','mode':mode,'wall_ms':(time.perf_counter()-start)*1000,'cuda_event_ms':start_event.elapsed_time(end_event)})
 summary.update(status='complete',weights_sha256=digest_file(weights),n_validation_images=32,checks=checks,all_predictions_exact=all(c['exact_prediction_dicts'] for c in checks),all_per_image_decisions_equal=all(c['per_image_operational_equal'] for c in checks),
  partial_batch_policy='Eager fallback for any size other than16; no dynamic-shape graph capture',anchor_reference_count=len(anchor_refs),captured_anchor_tensors_unchanged=anchors_unchanged,nms_policy='Clone captured main prediction tensor before original pinned postprocess; CUDA sync before NMS; NMS deadline warning aborts prototype',
  peak_allocated_gib=torch.cuda.max_memory_allocated(0)/1024**3,peak_reserved_gib=torch.cuda.max_memory_reserved(0)/1024**3,timings=timings)
 for scope in ['whole_batch','forward_with_input_copy_for_graph']:
  values={mode:statistics.median(t['wall_ms'] for t in timings if t['scope']==scope and t['mode']==mode) for mode in ['eager','graph']}
  summary[scope+'_median_wall_ms']=values;summary[scope+'_observed_ratio_eager_over_graph']=values['eager']/values['graph']
 dump_json(OUT/'result.json',summary)
 (OUT/'REPORT_KO.md').write_text(f'''# CUDA graph 격리 실행 프로토타입\n\n상태: {summary['status']}. 운영 실행 경로는 변경하지 않았다. 기존 GC10 체크포인트와 고정 검증영상32장을 사용했다. clean→hot25→dark25→cluster25와17장 부분배치 및 부분배치 후 다시 전체배치를 비교했다.\n\n- 전체 예측 사전 완전 일치: {summary['all_predictions_exact']}\n- 이미지별 검출 판정 일치: {summary['all_per_image_decisions_equal']}\n- 보존한 anchor 상수 불변: {anchors_unchanged}\n- Peak allocated/reserved: {summary['peak_allocated_gib']:.3f}/{summary['peak_reserved_gib']:.3f}GiB\n- 전체 배치 중앙 wall ms: {summary['whole_batch_median_wall_ms']}\n- forward+graph입력복사 중앙 wall ms: {summary['forward_with_input_copy_for_graph_median_wall_ms']}\n\n동일GPU에서 운영 작업이 함께 실행되므로 시간은 격리 속도 향상의 증거가 아니다. 교대로 워밍된 반복을 기록했지만 표본 수도 작다. 입력전처리와후처리는 고정Ultralytics predictor를 사용했고 CUDA graph는 batch16 forward만 캡처했다. 부분배치는 eager로 처리했다. 모델Detect의 anchor참조, static입력과출력을 보존했고 NMS에는 캡처출력의 복사본을 넘겼다.\n\n[PyTorch2.9공식문서]({DOC})의 side-stream워밍업과 같은 메모리 주소·고정 shape·장기참조 규칙을 따랐다. 이 제한된 결과만으로 운영 적용 또는 모든 새 체크포인트/조건의 동등성을 보장하지 않는다.\n''')
 print(json.dumps({k:v for k,v in summary.items() if k not in ['timings','checks']},indent=2),flush=True)

if __name__=='__main__':
 try:main()
 except Exception as exc:
  summary.update(status='error_or_unsupported',error=str(exc),traceback=traceback.format_exc())
  dump_json(OUT/'result.json',summary);(OUT/'REPORT_KO.md').write_text('# CUDA graph 격리 프로토타입\n\n실행 불가 또는 오류로 중단했다. 운영 실행은 변경하지 않았다.\n\n```text\n'+summary['traceback']+'\n```\n')
  print(summary['traceback'],file=sys.stderr);raise SystemExit(1)
