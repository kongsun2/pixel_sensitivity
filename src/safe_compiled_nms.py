"""Conservative compiled NMS with the saved TorchNMS ordering and boundary fallback.

Import is lazy with respect to torchvision and does not patch an operator. Install
explicitly in a dedicated inference process, or use the scoped context managers.
Because torchvision presence changes Ultralytics' branch, reference validation must
use reference_nms(), not an unguarded model call after invoking either context.
Contexts mutate a process-global operator and are not thread-safe for simultaneous
mixed-mode inference. Current evidence is finite validation, not a universal proof
of floating-point equivalence for arbitrary boxes, devices or software versions.
"""
from __future__ import annotations
from collections import Counter
from contextlib import contextmanager
import math
import torch
from ultralytics.utils.nms import TorchNMS

_REFERENCE_NMS=TorchNMS.nms
_COMPILED_NMS=None
_COUNTS=Counter()
_CANDIDATE_HIST=Counter()
MAX_COMPILED_N=2048
NEAR_IOU_TOLERANCE=1e-5


def _torchvision():
 global _COMPILED_NMS
 import torchvision
 if _COMPILED_NMS is None:_COMPILED_NMS=torchvision.ops.nms
 return torchvision


def _reference(boxes,scores,threshold,reason):
 _COUNTS[reason]+=1
 return _REFERENCE_NMS(boxes,scores,threshold)


def _guarded_impl(boxes,scores,iou_threshold):
 if boxes.ndim!=2 or boxes.shape[1]!=4 or scores.ndim!=1 or len(scores)!=len(boxes):raise ValueError('NMS expects boxes[N,4] and scores[N]')
 if boxes.device!=scores.device:raise ValueError('Boxes and scores must share a device')
 threshold=float(iou_threshold)
 if not math.isfinite(threshold) or not 0<=threshold<=1:raise ValueError('NMS IoU threshold must be finite and between0and1')
 n=len(boxes);_COUNTS['calls']+=1;_CANDIDATE_HIST[n]+=1
 if n==0:return _reference(boxes,scores,threshold,'empty')
 if n>MAX_COMPILED_N:return _reference(boxes,scores,threshold,'large_n')
 if boxes.dtype!=torch.float32 or scores.dtype!=torch.float32:return _reference(boxes,scores,threshold,'unsupported_dtype')
 wh=boxes[:,2:]-boxes[:,:2];area=wh[:,0]*wh[:,1]
 valid=(torch.isfinite(boxes).all() & torch.isfinite(scores).all() & torch.isfinite(area).all() & (wh>0).all() & (area>0).all())
 if not bool(valid):return _reference(boxes,scores,threshold,'invalid_or_degenerate')
 # Use the original float32 area/intersection arithmetic before trusting compiled decisions.
 lt=torch.maximum(boxes[:,None,:2],boxes[None,:,:2]);rb=torch.minimum(boxes[:,None,2:],boxes[None,:,2:])
 inter=(rb-lt).clamp_(min=0).prod(dim=2);union=area[:,None]+area[None,:]-inter;iou=inter/union
 if not bool(torch.isfinite(iou).all()):return _reference(boxes,scores,threshold,'nonfinite_iou')
 if bool(torch.triu((iou-threshold).abs()<=NEAR_IOU_TOLERANCE,diagonal=1).any()):return _reference(boxes,scores,threshold,'near_threshold')
 # Preserve even tied-score argsort order. Ordinal scores are never returned to users.
 order=scores.argsort(0,descending=True)
 ranks=torch.arange(n,0,-1,device=scores.device,dtype=scores.dtype)
 _torchvision()
 keep=_COMPILED_NMS(boxes[order],ranks,threshold)
 _COUNTS['compiled_rank_preserved']+=1
 return order[keep]


@contextmanager
def reference_nms():
 """Force the original pure TorchNMS even after torchvision is imported."""
 tv=_torchvision();previous=tv.ops.nms;tv.ops.nms=_REFERENCE_NMS
 try:yield
 finally:tv.ops.nms=previous


@contextmanager
def guarded_nms():
 """Temporarily choose the conservative compiled operator, restoring prior state."""
 tv=_torchvision();previous=tv.ops.nms;tv.ops.nms=_guarded_impl
 try:yield
 finally:tv.ops.nms=previous


def install_guarded_nms():
 """Explicitly install in this process; call only in the dedicated inference runner."""
 tv=_torchvision();tv.ops.nms=_guarded_impl
 return _guarded_impl


def get_stats(reset=False):
 """Return compact per-process counters without retaining per-image candidate arrays."""
 n=sum(_CANDIDATE_HIST.values());hist=sorted(_CANDIDATE_HIST.items());median=None
 if n:
  targets=[(n-1)//2,n//2];seen=0;values=[]
  for value,count in hist:
   for target in targets:
    if seen<=target<seen+count:values.append(value)
   seen+=count
  median=sum(values)/len(values)
 result={'counts':dict(_COUNTS),'candidate_n':{'n_calls':n,'min':hist[0][0] if n else None,'median':median,'max':hist[-1][0] if n else None,
  'mean':sum(k*v for k,v in hist)/n if n else None},'settings':{'max_compiled_n':MAX_COMPILED_N,'near_iou_tolerance':NEAR_IOU_TOLERANCE,'compiled_dtype':'float32'}}
 if reset:_COUNTS.clear();_CANDIDATE_HIST.clear()
 return result
