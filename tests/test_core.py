import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pytest
from core import cell_bounds,fault_pattern,inject_fault,iou_matrix,match_predictions,operational_metrics,stable_seed
from analyze import paired_ci
from prepare import load_gt


def g(box=(0,0,10,10),cl=0):return {'bbox_xyxy':list(box),'class_id':cl}
def p(score=.9,box=(0,0,10,10),cl=0):return {**g(box,cl),'score':score}

def test_grid_partitions_exactly():
    im=np.zeros((201,303),int)
    for b in range(16):
        x0,y0,x1,y1=cell_bounds(201,303,4,4,b);im[y0:y1,x0:x1]+=1
    assert np.all(im==1)

def test_same_pattern_translated():
    im=np.full((200,200,3),128,np.uint8);off=fault_pattern(200,200,4,4,9,'dispersed',1,13)
    a,ma=inject_fault(im,off,cell_bounds(200,200,4,4,0),0)
    b,mb=inject_fault(im,off,cell_bounds(200,200,4,4,1),0)
    assert ma['count_changed']==mb['count_changed']==9
    assert np.array_equal(a[:50,:50],b[:50,50:100]);assert np.all(im==128)

def test_nested_patterns():
    a=fault_pattern(200,200,4,4,9,'dispersed',1,13)
    b=fault_pattern(200,200,4,4,25,'dispersed',1,13)
    assert np.array_equal(a,b[:9])

def test_disperse_spacing():
    off=fault_pattern(200,200,4,4,25,'dispersed',1,13)
    d=np.abs(off[:,None]-off[None,:]).max(axis=2);np.fill_diagonal(d,10000)
    assert d.min()>1

def test_cluster_count():
    off=fault_pattern(200,200,4,4,9,'cluster',3,13)
    assert len(off)==len(np.unique(off,axis=0))==9
    assert np.ptp(off[:,0])==2 and np.ptp(off[:,1])==2

def test_wrong_cluster_budget_fails():
    with pytest.raises(ValueError):fault_pattern(200,200,4,4,10,'cluster',3,13)

def test_oversize_fails():
    with pytest.raises(ValueError):fault_pattern(20,20,4,4,30,'dispersed',1,13)

def test_outside_grid_fails():
    with pytest.raises(ValueError):inject_fault(np.zeros((20,20,3),np.uint8),np.array([[10,1]]),(0,0,5,5),255)

def test_duplicates_fail():
    with pytest.raises(ValueError):inject_fault(np.zeros((10,10,3),np.uint8),np.array([[1,1],[1,1]]),(0,0,10,10),255)

def test_noop_explicit():
    im=np.zeros((10,10,3),np.uint8);out,m=inject_fault(im,np.array([[1,1]]),(0,0,10,10),0)
    assert m['count_commanded']==1 and m['count_changed']==0;assert np.array_equal(im,out)

def test_zero_alpha():
    im=np.full((10,10,3),30,np.uint8);out,m=inject_fault(im,np.array([[1,1]]),(0,0,10,10),255,0.)
    assert np.array_equal(im,out) and m['count_changed']==0

def test_single_channel():
    im=np.full((10,10,3),128,np.uint8);out,m=inject_fault(im,np.array([[1,1]]),(0,0,10,10),255,1.,'r')
    assert out[1,1].tolist()==[255,128,128];assert m['count_changed_channels']==1

def test_iou_halfopen():
    assert iou_matrix([[0,0,10,10]],[[0,0,10,10]])[0,0]==1
    assert iou_matrix([[0,0,10,10]],[[10,0,20,10]])[0,0]==0

def test_one_gt_duplicate_predictions():
    m=match_predictions([g()],[p(.9),p(.8)],.25,.5)
    assert (m['tp'],m['fp'],m['fn'])==(1,1,0)

def test_wrong_class():
    m=match_predictions([g()],[p(cl=1)],.25,.5)
    assert (m['tp'],m['fp'],m['fn'])==(0,1,1)

def test_operating_vs_low_floor():
    gt=[g()];pred=[p(.1)]
    assert match_predictions(gt,pred,.25,.5)['tp']==0
    assert match_predictions(gt,pred,.001,.5)['gt_scores']==[.1]

def test_no_pred_or_gt():
    assert match_predictions([g()],[],.25,.5)['fn']==1
    assert match_predictions([],[p()],.25,.5)['fp']==1
    m=operational_metrics([{'gt':[],'pred':[]}],.25,.5)
    assert m['recall'] is None and m['precision'] is None

def test_recall_is_pooled_not_image_mean():
    r=[{'gt':[g()],'pred':[p()]},{'gt':[g() for _ in range(9)],'pred':[]}]
    m=operational_metrics(r,.25,.5)
    assert m['recall']==.1 and m['miss_rate']==.9

def test_identical_bootstrap_zero():
    x=np.array([1,2,0,1]);f=np.tile(x,(3,1));den=np.array([1,2,1,2])
    assert paired_ci(x,f,den,['a','b','c','d'],100,5)==[0.,0.]

def test_stable_seed():
    assert stable_seed('x',1)==stable_seed('x',1)
    assert stable_seed('x',1)!=stable_seed('x',2)

def test_yolo_box_roundtrip(tmp_path):
    f=tmp_path/'x.txt';f.write_text('0 0.5 0.5 1 1\n')
    assert load_gt(f,'yolo',['x'],20,10,None)[0]['bbox_xyxy']==[0.,0.,20.,10.]

def test_voc_explicit_convention(tmp_path):
    f=tmp_path/'x.xml';f.write_text('<annotation><object><name>x</name><bndbox><xmin>1</xmin><ymin>1</ymin><xmax>20</xmax><ymax>10</ymax></bndbox></object></annotation>')
    assert load_gt(f,'voc',['x'],20,10,'1inclusive')[0]['bbox_xyxy']==[0.,0.,20.,10.]

def test_missing_annotation_not_normal(tmp_path):
    with pytest.raises(FileNotFoundError):load_gt(tmp_path/'missing','yolo',['x'],20,10,None)

@pytest.mark.parametrize('channels',['all','r','g','b'])
@pytest.mark.parametrize('alpha',[0.,.5,1.])
def test_sparse_fault_metrics_equal_full_image_difference(channels,alpha):
    rng=np.random.default_rng(71)
    im=rng.integers(0,256,(31,39,3),dtype=np.uint8)
    off=np.array([[0,0],[2,3],[7,8]])
    out,meta=inject_fault(im,off,(4,5,20,20),128,alpha,channels)
    delta=out.astype(np.int16)-im.astype(np.int16)
    assert meta['count_changed']==int(np.any(delta!=0,axis=2).sum())
    assert meta['count_changed_channels']==int(np.count_nonzero(delta))
    assert meta['l1_rgb_sum']==int(np.abs(delta).sum())
    assert meta['linf_rgb']==int(np.abs(delta).max())
