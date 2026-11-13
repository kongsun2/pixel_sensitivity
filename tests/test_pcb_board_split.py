"""Numeric-ID split must keep both sides and patches together without editing inputs."""
import copy,sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_pcb_board_split import assign_numeric_group_split


def samples():
 return [{'image_id':f'{i}_'+side+f'_{patch:03d}','image_path':f'/x/{i}_{side}_{patch}.jpg','label_path':f'/y/{i}_{side}_{patch}.txt','split':'train' if side=='t' else 'test','group_id':f'{i}_{side}'} for i in range(100,120) for side in ['t','b'] for patch in [1,2]]


def test_both_sides_and_patches_share_numeric_group_and_split():
 rows=samples();original=copy.deepcopy(rows);out=assign_numeric_group_split(rows,101)
 assert rows==original
 groups={}
 for r in out:
  assert r['group_id']==r['image_id'].split('_')[0]
  groups.setdefault(r['group_id'],set()).add(r['split'])
  assert r['image_path']==next(x['image_path'] for x in original if x['image_id']==r['image_id'])
 assert all(len(s)==1 for s in groups.values())
 assert {s:sum(next(iter(v))==s for v in groups.values()) for s in ['train','val','test']}=={'train':14,'val':3,'test':3}


def test_assignment_is_deterministic_under_source_row_order():
 a=assign_numeric_group_split(samples(),101);b=assign_numeric_group_split(list(reversed(samples())),101)
 assert {r['image_id']:r['split'] for r in a}=={r['image_id']:r['split'] for r in b}


def test_reject_non_numeric_proxy_id():
 rows=samples();rows[0]['image_id']='unknown_t_001'
 with pytest.raises(ValueError):assign_numeric_group_split(rows,101)
