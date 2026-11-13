"""CPU checks for tie order, guards and scoped operator replacement."""
import sys
from pathlib import Path
import torch,pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import safe_compiled_nms as safe
from ultralytics.utils.nms import TorchNMS


@pytest.fixture(autouse=True)
def stats():
 safe.get_stats(reset=True)
 yield
 safe.get_stats(reset=True)


def candidate(boxes,scores,threshold=.7):
 with safe.guarded_nms():
  import torchvision
  return torchvision.ops.nms(boxes,scores,threshold)


def test_tied_overlapping_scores_keep_reference_sort_selection_and_original_inputs():
 boxes=torch.tensor([[0,0,10,10],[.5,.5,10.5,10.5],[20,20,25,25],[0,0,10,10]],dtype=torch.float32)
 scores=torch.tensor([.5,.5,.5,.3]);old_boxes=boxes.clone();old_scores=scores.clone()
 expected=TorchNMS.nms(boxes,scores,.7);actual=candidate(boxes,scores)
 assert torch.equal(actual,expected)
 assert torch.equal(boxes,old_boxes) and torch.equal(scores,old_scores)
 assert safe.get_stats()['counts']['compiled_rank_preserved']==1


def test_near_threshold_and_degenerate_use_original_reference():
 boxes=torch.tensor([[0,0,10,10],[0,0,7,10]],dtype=torch.float32);scores=torch.tensor([.8,.7])
 assert torch.equal(candidate(boxes,scores),TorchNMS.nms(boxes,scores,.7))
 assert safe.get_stats()['counts']['near_threshold']==1
 boxes=torch.tensor([[0,0,0,4],[1,1,3,3]],dtype=torch.float32)
 assert torch.equal(candidate(boxes,scores),TorchNMS.nms(boxes,scores,.7))
 assert safe.get_stats()['counts']['invalid_or_degenerate']==1


def test_large_dtype_nonfinite_and_empty_guards_call_saved_reference(monkeypatch):
 calls=[]
 def fake(boxes,scores,threshold):calls.append(len(boxes));return torch.tensor([123],dtype=torch.long)
 monkeypatch.setattr(safe,'_REFERENCE_NMS',fake)
 for boxes,scores in [(torch.zeros(2049,4),torch.ones(2049)),(torch.zeros(2,4,dtype=torch.float64),torch.ones(2,dtype=torch.float64)),(torch.tensor([[0.,0.,float('inf'),3.]]),torch.ones(1)),(torch.empty(0,4),torch.empty(0))]:
  assert candidate(boxes,scores).tolist()==[123]
 assert calls==[2049,2,1,0]


def test_contexts_restore_operators_and_reference_forces_pure_torch():
 import torchvision
 before=torchvision.ops.nms
 with safe.reference_nms():
  assert torchvision.ops.nms is safe._REFERENCE_NMS
  with safe.guarded_nms():assert torchvision.ops.nms is not safe._REFERENCE_NMS
  assert torchvision.ops.nms is safe._REFERENCE_NMS
 assert torchvision.ops.nms is before
 installed=safe.install_guarded_nms()
 try:
  assert torchvision.ops.nms is installed
  with safe.reference_nms():assert torchvision.ops.nms is safe._REFERENCE_NMS
  assert torchvision.ops.nms is installed
 finally:torchvision.ops.nms=before


def test_random_tied_scores_match_saved_reference_on_cpu():
 for seed in range(8):
  g=torch.Generator().manual_seed(seed);xy=torch.rand(64,2,generator=g)*30;wh=torch.rand(64,2,generator=g)*10+.1
  boxes=torch.cat([xy,xy+wh],dim=1);scores=(torch.rand(64,generator=g)*10).round()/10
  assert torch.equal(candidate(boxes,scores),TorchNMS.nms(boxes,scores,.7))
