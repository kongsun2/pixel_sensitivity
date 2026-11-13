"""Prediction reuse must reject changes to the model/data/injection contract."""
import copy,sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_grid8_pattern10 import check_reuse_contract

def reference():
 return {'source_sha256':{'run.py':'same','core.py':'same'},'manifest_sha256':'data','classes_sha256':'classes','weights_sha256':'weights','weights_path':'/model.pt','batch':16,'threads':8,'split':'test','selected_ids':['a'],'versions':{'numpy':'same'},'config':{'grid':{'rows':8,'cols':8},'repeats':3,'fault_seed_base':301,'predict':{'conf_floor':.001},'evaluation':{'conf_operating':.25},'conditions':[{'id':'bright_k25','count':25,'target':255}]},'device':'0','worker_count':1}

def test_reuse_allows_more_patterns_and_worker_assignment():
 old=reference();new=copy.deepcopy(old);new['config']['repeats']=10;new['device']='0,1';new['worker_count']=2
 check_reuse_contract(old,new)

@pytest.mark.parametrize('field',['fault_seed_base','grid','conditions'])
def test_reuse_rejects_injection_changes(field):
 old=reference();new=copy.deepcopy(old);new['config']['repeats']=10;new['config'][field]='changed'
 with pytest.raises(ValueError):check_reuse_contract(old,new)

@pytest.mark.parametrize('field',['weights_sha256','manifest_sha256','classes_sha256','source_sha256','selected_ids','versions','batch'])
def test_reuse_rejects_data_model_or_code_changes(field):
 old=reference();new=copy.deepcopy(old);new['config']['repeats']=10;new[field]='changed'
 with pytest.raises(ValueError):check_reuse_contract(old,new)
