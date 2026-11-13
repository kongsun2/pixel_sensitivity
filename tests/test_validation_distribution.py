import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from analyze_validation_distribution import count_sites_in_box, object_events, make_crossfit_expectations

def test_halfopen_sites_and_empty():
 assert count_sites_in_box([[1,1],[4,4],[5,3],[3,5]], [1,1,5,5])==2
 assert count_sites_in_box([], [1,1,5,5])==0

def test_classaware_clean_loss_and_recovery():
 gt=[{'class_id':0,'bbox_xyxy':[0,0,10,10]},{'class_id':1,'bbox_xyxy':[20,20,30,30]}]
 clean={'gt':gt,'pred':[{'class_id':0,'bbox_xyxy':[0,0,10,10],'score':.9}]}
 fault={'gt':gt,'pred':[{'class_id':1,'bbox_xyxy':[0,0,10,10],'score':.99},{'class_id':1,'bbox_xyxy':[20,20,30,30],'score':.8}], 'fault':{'coordinates_xy':[[2,2],[29,29],[30,30]]}}
 e=object_events(clean,fault)
 assert e[0]['clean_tp'] and e[0]['lost'] and not e[0]['recovered'] and e[0]['n_sites']==1
 assert not e[1]['clean_tp'] and e[1]['recovered'] and not e[1]['lost'] and e[1]['n_sites']==1

def test_crossfit_excludes_outcome_of_heldout_fold():
 # tuple: fold, cell, clean-status, class, size, hit-bin -> [n, event_count]
 rows={(0,0,1,0,0,'none'):[10,10],(1,0,1,0,0,'none'):[10,0]}
 expected=make_crossfit_expectations(rows)
 assert expected[(0,0,1,0,0,'none')]['expected_events']==0
 assert expected[(1,0,1,0,0,'none')]['expected_events']==10

def test_numeric_split_check_combines_both_sides():
 from analyze_validation_distribution import numeric_split_check
 result=numeric_split_check([{'image_id':'4_t_001','split':'train'},{'image_id':'4_b_002','split':'test'}])
 assert result['numeric_groups_crossing_splits']==1
 assert result['numeric_groups_in_train_and_test']==1

def test_numeric_split_check_confirms_disjoint_ids():
 from analyze_validation_distribution import numeric_split_check
 result=numeric_split_check([{'image_id':'4_t_001','split':'train'},{'image_id':'5_b_002','split':'test'}])
 assert result['numeric_groups_crossing_splits']==0
 assert result['numeric_groups_in_train_and_test']==0
