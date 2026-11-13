"""Joint resampling must preserve the same image/mask draw at every location."""
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from analyze_validation_positions import joint_position_bootstrap


def test_identical_positions_have_exact_zero_deviation_band():
    rng=np.random.default_rng(4)
    one=rng.integers(-1,2,(4,1,8))
    loss=np.repeat(one,5,axis=1)
    result=joint_position_bootstrap(loss,np.full(8,2),[str(i) for i in range(8)],n_boot=200,seed=7)
    assert np.allclose(result['deviation'],0)
    assert result['deviation_critical_radius']==pytest.approx(0)
    assert np.std(result['draws_sensitivity'][:,0])>0


def test_constant_location_effect_is_detected_without_independent_cell_resampling():
    loss=np.zeros((4,5,10));loss[:,0,:]=1
    result=joint_position_bootstrap(loss,np.full(10,2),[str(i) for i in range(10)],n_boot=200,seed=7)
    assert result['sensitivity'][0]==pytest.approx(.5)
    assert result['deviation'][0]==pytest.approx(.4)
    assert result['deviation_critical_radius']==pytest.approx(0)
    assert result['deviation_simultaneous_low'][0]>0
    assert np.all(result['deviation_simultaneous_high'][1:]<0)


def test_duplicate_rows_with_same_group_preserve_the_joint_bootstrap():
    rng=np.random.default_rng(3);loss=rng.integers(-1,2,(3,4,6));gt=np.full(6,2);groups=list('abcdef')
    a=joint_position_bootstrap(loss,gt,groups,n_boot=200,seed=5)
    b=joint_position_bootstrap(np.repeat(loss,2,axis=2),np.repeat(gt,2),np.repeat(groups,2),n_boot=200,seed=5)
    assert np.allclose(a['draws_sensitivity'],b['draws_sensitivity'])
    assert np.allclose(a['deviation'],b['deviation'])


def test_grand_mean_pairing_keeps_all_deviation_draws_centered():
    rng=np.random.default_rng(1);loss=rng.integers(-1,2,(3,4,6));gt=np.full(6,2)
    a=joint_position_bootstrap(loss,gt,list('abcdef'),n_boot=200,seed=9)
    d=a['draws_sensitivity']-a['draws_sensitivity'].mean(axis=1,keepdims=True)
    assert np.allclose(d.sum(axis=1),0)
    again=joint_position_bootstrap(loss,gt,list('abcdef'),n_boot=200,seed=9)
    assert np.array_equal(a['draws_sensitivity'],again['draws_sensitivity'])


def test_no_ground_truth_is_not_a_zero_effect_result():
    with pytest.raises(ValueError):joint_position_bootstrap(np.zeros((2,3,4)),np.zeros(4),list('abcd'),n_boot=20)


def test_recall_uses_pooled_gt_not_image_mean():
    loss=np.zeros((3,2,2));loss[:,0,0]=1
    result=joint_position_bootstrap(loss,np.array([1,9]),['a','b'],n_boot=100,seed=7)
    assert result['sensitivity'][0]==pytest.approx(.1)
    assert result['sensitivity'][1]==0
