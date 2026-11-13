"""Full-report guards and mean arithmetic must reject partial or mixed experiments."""
import sys,copy
from pathlib import Path
import numpy as np,pandas as pd,pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from report_research_validation import validate_manifest,validate_grid_rows,summarize_metrics


def jobs():
 conditions=['bright_k25','dark_k25','stuck_k25','bright_k1','bright_k9','bright_partial25','bright_cluster25']
 return [{'key':f'{d}_{seed}_{c}','dataset':d,'seed':seed,'condition':c} for d in ['gc10_det','neu_det','pcb_ind_board'] for seed in [11,22,33] for c in (conditions if seed==11 else ['bright_k25'])]


def test_manifest_requires_every_expected_job_and_complete_state():
 j=jobs();state={x['key']:{'status':'complete'} for x in j};validate_manifest({'jobs':j},state)
 bad=copy.deepcopy(state);bad[j[0]['key']]['status']='running'
 with pytest.raises(ValueError):validate_manifest({'jobs':j},bad)
 with pytest.raises(ValueError):validate_manifest({'jobs':j[:-1]},state)


def frame():
 return pd.DataFrame([{'condition_id':'bright_k25','repeat':r,'cell':b,'recall':.8-r*.001-b*.0001,'precision':.9,'coco_map50':.7,'coco_map50_95':.4,'n_images':5,'n_gt':10,'fp_per_image':.2} for r in range(10) for b in range(64)])


def test_grid_complete_unique_and_metrics_mean_in_percentage_points():
 m=frame();validate_grid_rows(m,'bright_k25',5,10)
 base={'recall':.85,'precision':.9,'coco_map50':.75,'coco_map50_95':.45}
 result,patterns=summarize_metrics(m,base)
 assert len(patterns)==10
 assert result['mean_recall_pct']==pytest.approx(m.recall.mean()*100)
 assert result['recall_drop_pp']==pytest.approx((.85-m.recall.mean())*100)
 assert result['pattern_sd_recall_drop_pp']==pytest.approx(np.std([(.85-g.recall.mean())*100 for _,g in m.groupby('repeat')],ddof=1))


def test_grid_rejects_duplicate_missing_nonfinite_or_changed_gt():
 m=frame()
 for broken in [m.iloc[:-1],pd.concat([m.iloc[:-1],m.iloc[:1]])]:
  with pytest.raises(ValueError):validate_grid_rows(broken,'bright_k25',5,10)
 m.loc[0,'recall']=np.nan
 with pytest.raises(ValueError):validate_grid_rows(m,'bright_k25',5,10)
 m=frame();m.loc[0,'n_gt']=11
 with pytest.raises(ValueError):validate_grid_rows(m,'bright_k25',5,10)


def test_compiled_certificate_matches_all_sources_and_eager_reference_chain():
 from report_research_validation import validate_backend_certificate
 meta={'backend':'bgr-packed-safe-compiled-nms-fp32-bs16-v4','source_sha256':{'run_compiled.py':'compiled','safe_compiled_nms.py':'nms','run_fast.py':'fast','run.py':'high_api'}}
 cert={'selected_batch':16,'all_pass':True,'checks':[{'accepted':True,'exact_predictions':True,'per_image_matches_equal':True,'ap_max_abs_difference':0.}],
 'runner_sha256':'compiled','nms_sha256':'nms','reference_runner_sha256':'high_api','reference_fast_runner_sha256':'fast'}
 assert validate_backend_certificate(meta,cert)=='compiled'
 cert['nms_sha256']='wrong'
 with pytest.raises(ValueError):validate_backend_certificate(meta,cert)


def test_fast_certificate_is_still_supported_and_requires_reference_source():
 from report_research_validation import validate_backend_certificate
 meta={'backend':'bgr-packed-direct-predictor-fp32-batch16-v3','source_sha256':{'run_fast.py':'fast','run.py':'high_api'}}
 cert={'selected_batch':16,'all_pass':True,'checks':[{'accepted':True,'exact_predictions':True,'per_image_matches_equal':True,'ap_max_abs_difference':0.}],
 'runner_sha256':'fast','reference_runner_sha256':'high_api'}
 assert validate_backend_certificate(meta,cert)=='fast'
 cert['reference_runner_sha256']='wrong'
 with pytest.raises(ValueError):validate_backend_certificate(meta,cert)


def test_checkpoint_certificate_is_bound_to_model_split_source_and_validation_ids():
 from report_research_validation import validate_checkpoint_control,CONDITIONS
 source={'run_compiled.py':'runner','safe_compiled_nms.py':'nms','run_fast.py':'fast','run.py':'old','core.py':'core'}
 meta={'source_sha256':source,'pattern_key':'pcb_ind','weights_sha256':'weight','manifest_sha256':'data','classes_sha256':'classes'};job={'dataset':'pcb_ind_board'}
 cert={'all_pass':True,'runner_sha256':'runner','weights_sha256':'weight','nms_sha256':'nms','selected_validation_ids':[f'v{i}' for i in range(17)],
 'checks':[{'condition':c,'n_images':17,'exact_predictions':True} for c in ['clean',*CONDITIONS]],
 'fingerprints':{'dataset':'pcb_ind_board','pattern_key':'pcb_ind','weights_sha256':'weight','manifest_sha256':'data','classes_sha256':'classes','condition_config_sha256':'config','control_script_sha256':'script','source_sha256':source}}
 ids=set(cert['selected_validation_ids']);validate_checkpoint_control(meta,job,cert,'script','config',ids)
 bad=copy.deepcopy(cert);bad['selected_validation_ids'][0]='test_image'
 with pytest.raises(ValueError):validate_checkpoint_control(meta,job,bad,'script','config',ids)
 bad=copy.deepcopy(cert);bad['fingerprints']['weights_sha256']='other'
 with pytest.raises(ValueError):validate_checkpoint_control(meta,job,bad,'script','config',ids)
 bad=copy.deepcopy(cert);bad['checks']=bad['checks'][:-1]
 with pytest.raises(ValueError):validate_checkpoint_control(meta,job,bad,'script','config',ids)
