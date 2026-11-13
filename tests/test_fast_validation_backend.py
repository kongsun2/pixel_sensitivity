import sys
from pathlib import Path
import numpy as np,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from run_fast import pack_predictions
from core import inject_fault

def test_packed_predictions_keep_image_order_empty_images_and_scores():
 class Boxes:
  def __init__(self,x):self.data=torch.tensor(x,dtype=torch.float32).reshape(-1,6)
 class Result:
  def __init__(self,x):self.boxes=Boxes(x)
 out=pack_predictions([Result([[1,2,3,4,.5,2]]),Result([]),Result([[5,6,7,8,.25,1]])])
 assert out==[[{'class_id':2,'score':.5,'bbox_xyxy':[1.,2.,3.,4.]}],[],[{'class_id':1,'score':.25,'bbox_xyxy':[5.,6.,7.,8.]}]]

def test_all_channel_injection_commutes_with_rgb_bgr_conversion():
 im=np.random.default_rng(1).integers(0,256,(20,24,3),dtype=np.uint8);offset=np.array([[0,0],[1,3],[8,6]])
 for target in [0,128,255]:
  for alpha in [.5,1.]:
   rgb,rm=inject_fault(im,offset,(3,4,13,15),target,alpha,'all')
   bgr,bm=inject_fault(np.ascontiguousarray(im[...,::-1]),offset,(3,4,13,15),target,alpha,'all')
   assert np.array_equal(rgb[...,::-1],bgr)
   assert rm==bm

def test_nms_time_limit_guard_rejects_truncated_output():
 import logging,pytest
 from run_fast import NmsDeadlineGuard
 guard=NmsDeadlineGuard()
 with pytest.raises(RuntimeError):guard.emit(logging.LogRecord('ultralytics',logging.WARNING,'',0,'NMS time limit 2.800s exceeded',(),None))
 guard.emit(logging.LogRecord('ultralytics',logging.WARNING,'',0,'ordinary diagnostic',(),None))
