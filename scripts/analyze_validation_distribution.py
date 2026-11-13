"""Existing-cache object exposure analysis; descriptive, not causal adjustment.

Default sources are the three completed ten-pattern bright25 8x8 runs.
Optional --run DATASET=PATH and --classes DATASET=PATH support future runs.
No GPU/model inference is performed. Original source files are never modified.
"""
from __future__ import annotations
import argparse,collections,csv,json,os,sys
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('OPENBLAS_NUM_THREADS','4');os.environ.setdefault('MKL_NUM_THREADS','4')
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from core import digest_file,dump_json,load_jsonl,match_predictions
DEFAULT_OUT=ROOT/'reports/research_validation_2026-11-11/distribution'
TITLES={'gc10_det':'GC10-DET','neu_det':'NEU-DET','pcb_ind':'PCB-IND','pcb_ind_board':'PCB-IND (numeric-ID split)'}
FIELDS=['n_gt','n_clean_tp','n_fault_tp','n_lost','n_recovered','n_gt_hit','n_clean_tp_hit','n_lost_hit','n_lost_unhit','n_sites_sum']


def count_sites_in_box(coordinates_xy,box):
 xy=np.asarray(coordinates_xy,dtype=float).reshape(-1,2)
 if not len(xy):return 0
 x0,y0,x1,y1=box
 return int(((xy[:,0]>=x0)&(xy[:,0]<x1)&(xy[:,1]>=y0)&(xy[:,1]<y1)).sum())


def object_events(clean,fault,threshold=.25,match_iou=.5,clean_match=None):
 if clean['gt']!=fault['gt']:raise ValueError('Ground truth changed after injection')
 cm=clean_match if clean_match is not None else match_predictions(clean['gt'],clean['pred'],threshold,match_iou)
 fm=match_predictions(fault['gt'],fault['pred'],threshold,match_iou)
 out=[]
 for i,g in enumerate(clean['gt']):
  c=bool(cm['gt_matched'][i]);f=bool(fm['gt_matched'][i]);sites=count_sites_in_box(fault['fault']['coordinates_xy'],g['bbox_xyxy'])
  out.append({'class_id':g['class_id'],'clean_tp':c,'fault_tp':f,'lost':c and not f,'recovered':not c and f,'n_sites':sites})
 return out


def hit_bin(n):return 'none' if n==0 else 'one' if n==1 else 'two_to_four' if n<=4 else 'five_or_more'


def make_crossfit_expectations(rows):
 """Out-of-fold event rates from class x size x hit bins, never using cell.
 rows: (fold,cell,clean_status,class,size,hit_bin) -> [observations,event_count].
 Training rates exclude every observation of the held-out image group fold.
 Fallback levels progressively drop class, then size, then hit.
 """
 def keys(status,cl,size,hit):return [(status,cl,size,hit),(status,size,hit),(status,hit),(status,)]
 total=[collections.defaultdict(lambda:np.zeros(2,float)) for _ in range(4)]
 folds=[collections.defaultdict(lambda:np.zeros(2,float)) for _ in range(4)]
 for (fold,cell,status,cl,size,hit),val in rows.items():
  for level,key in enumerate(keys(status,cl,size,hit)):
   total[level][key]+=val;folds[level][(fold,key)]+=val
 out={}
 for full,val in rows.items():
  fold,cell,status,cl,size,hit=full
  for level,key in enumerate(keys(status,cl,size,hit)):
   other=total[level][key]-folds[level][(fold,key)]
   if other[0]>0:
    out[full]={'expected_events':float(val[0]*other[1]/other[0]),'risk':float(other[1]/other[0]),'fallback_level':level};break
  else:out[full]={'expected_events':None,'risk':None,'fallback_level':4}
 return out


def add_events(count,event):
 c=event['clean_tp'];f=event['fault_tp'];lost=event['lost'];rec=event['recovered'];hit=event['n_sites']>0
 for key,val in zip(FIELDS,[1,c,f,lost,rec,hit,c and hit,lost and hit,lost and not hit,event['n_sites']]):count[key]+=int(val)


def rates(count):
 def safe(num,den):return float(num/den) if den else None
 return {**count,'loss_risk_clean_tp':safe(count['n_lost'],count['n_clean_tp']),
 'recovery_risk_clean_fn':safe(count['n_recovered'],count['n_gt']-count['n_clean_tp']),
 'clean_tp_exposure_fraction':safe(count['n_clean_tp_hit'],count['n_clean_tp']),
 'recall_drop_pp':safe(100*(count['n_lost']-count['n_recovered']),count['n_gt'])}


def summarize_dataset(d,run,classes_path,out):
 meta=json.loads((run/'run_meta.json').read_text());cfg=meta['config'];t=cfg['evaluation']['conf_operating'];iou=cfg['evaluation']['match_iou'];names=json.loads(classes_path.read_text())
 if digest_file(classes_path)!=meta['classes_sha256']:raise ValueError('Class map SHA mismatch')
 if cfg['grid']!={'rows':8,'cols':8} or cfg['repeats']!=10:raise ValueError('Expected complete 8x8 ten-pattern run')
 hashes={}
 def read_shard(name):
  p=run/'predictions'/f'{name}.jsonl';mark=json.loads(p.with_suffix('.done.json').read_text())
  h=digest_file(p)
  if mark['run_hash']!=meta['run_hash'] or mark['sha256']!=h:raise ValueError('Invalid prediction shard: '+name)
  if mark.get('n_images')!=len(meta['selected_ids']):raise ValueError('Unexpected shard image count')
  hashes[name]=h
  return p
 clean=load_jsonl(read_shard('clean'));by_id={r['image_id']:r for r in clean};ids=[r['image_id'] for r in clean]
 matches={r['image_id']:match_predictions(r['gt'],r['pred'],t,iou) for r in clean}
 def group(r):return r['image_id'].split('_')[0] if d.startswith('pcb') else r['group_id']
 groups=sorted({group(r) for r in clean});order=np.random.default_rng(8811).permutation(groups).tolist();fold_map={g:i%5 for i,g in enumerate(order)}
 objmeta={};areas=[];normalized=[];centers=[];base_class=collections.defaultdict(lambda:collections.Counter())
 for r in clean:
  for i,g in enumerate(r['gt']):
   x0,y0,x1,y1=g['bbox_xyxy'];area=(x1-x0)*(y1-y0);rel=area/(r['height']*r['width']);cx=(x0+x1)/(2*r['width']);cy=(y0+y1)/(2*r['height']);areas.append(area);normalized.append(rel);centers.append((cx,cy))
   base_class[g['class_id']]['n_gt']+=1;base_class[g['class_id']]['n_clean_tp']+=int(matches[r['image_id']]['gt_matched'][i])
   objmeta[r['image_id'],i]={'relative_area':rel,'area_px':area}
 bounds=np.quantile(normalized,[1/3,2/3]);size_names=['small_tertile','medium_tertile','large_tertile']
 for v in objmeta.values():v['size_bin']=int(np.searchsorted(bounds,v['relative_area'],side='right'))
 cells=collections.defaultdict(collections.Counter);strata=collections.defaultdict(collections.Counter);by_pattern=collections.defaultdict(collections.Counter);cross=collections.defaultdict(lambda:np.zeros(2,float))
 for rep in range(10):
  for cell in range(64):
   p=read_shard(f'bright_k25__r{rep:03d}__g{cell:03d}')
   with p.open() as f:
    n_seen=0
    for expected,line in zip(ids,f):
     n_seen+=1
     fault=json.loads(line)
     if fault['image_id']!=expected:raise ValueError('Image sequence mismatch')
     cr=by_id[expected];events=object_events(cr,fault,t,iou,matches[expected])
     for i,event in enumerate(events):
      add_events(cells[cell],event);add_events(by_pattern[rep,cell],event)
      cl=event['class_id'];size=objmeta[expected,i]['size_bin'];hb=hit_bin(event['n_sites']);add_events(strata[cl,size,hb],event)
      status=int(event['clean_tp']);event_count=event['lost'] if status else event['recovered'];key=(fold_map[group(cr)],cell,status,cl,size,hb)
      cross[key]+=[1,event_count]
    if n_seen!=len(ids) or f.readline():raise ValueError('Fault record count differs')
  print(d,'processed pattern',rep+1,flush=True)
 expectations=make_crossfit_expectations(cross);pred_lost=collections.Counter();pred_rec=collections.Counter();fallback=collections.Counter()
 for key,pred in expectations.items():
  fold,cell,status,cl,size,hb=key;n=cross[key][0];fallback[pred['fallback_level']]+=int(n)
  if pred['expected_events'] is None:raise ValueError('No out-of-fold status support')
  (pred_lost if status else pred_rec)[cell]+=pred['expected_events']
 saved=pd.read_csv(run/'analysis/grid_summary.csv').sort_values('cell');occupancy=pd.read_csv(run/'analysis/occupancy.csv').set_index('cell')
 cell_rows=[]
 for cell in range(64):
  value=rates(cells[cell]);observed=value['recall_drop_pp'];expected=100*(pred_lost[cell]-pred_rec[cell])/value['n_gt'];saved_value=float(saved.loc[saved.cell==cell,'sensitivity_pp'].iloc[0])
  if not np.isclose(observed,saved_value,atol=1e-10):raise ValueError('Object-event net drop differs from existing evaluation')
  cell_rows.append({'dataset':d,'cell':cell,'grid_label':f'G{cell+1}',**value,'box_area_occupancy':float(occupancy.loc[cell,'box_area_occupancy']),
   'crossfit_expected_lost':pred_lost[cell],'crossfit_expected_recovered':pred_rec[cell],'crossfit_expected_drop_pp':expected,'descriptive_residual_drop_pp':observed-expected})
 pd.DataFrame(cell_rows).to_csv(out/f'{d}_cell_exposure.csv',index=False)
 strata_rows=[{'dataset':d,'class_id':cl,'class_name':names[cl],'size_tertile':size_names[size],'hit_bin':hit,**rates(count)} for (cl,size,hit),count in sorted(strata.items())]
 pd.DataFrame(strata_rows).to_csv(out/f'{d}_class_size_hit_strata.csv',index=False)
 pattern_rows=[{'dataset':d,'pattern_id':rep+1,'cell':cell,**rates(count)} for (rep,cell),count in sorted(by_pattern.items())]
 pd.DataFrame(pattern_rows).to_csv(out/f'{d}_pattern_cell_events.csv',index=False)
 class_rows=[{'dataset':d,'class_id':cl,'class_name':names[cl],'n_gt':count['n_gt'],'n_clean_tp':count['n_clean_tp'],'clean_recall':count['n_clean_tp']/count['n_gt']} for cl,count in sorted(base_class.items())]
 pd.DataFrame(class_rows).to_csv(out/f'{d}_clean_class_distribution.csv',index=False)
 cc=pd.DataFrame(cell_rows);allcount=collections.Counter()
 for c in cells.values():allcount.update(c)
 overall=rates(allcount);hit_sum=collections.Counter();unhit_sum=collections.Counter()
 for (cl,size,hb),count in strata.items():(unhit_sum if hb=='none' else hit_sum).update(count)
 hrate=rates(hit_sum);urate=rates(unhit_sum);maxrow=cc.loc[cc.recall_drop_pp.idxmax()]
 centers=np.array(centers);central=((centers[:,0]>=.375)&(centers[:,0]<.625)&(centers[:,1]>=.375)&(centers[:,1]<.625)).sum()
 result={'dataset':d,'title':TITLES.get(d,d),'n_images':len(clean),'n_gt':len(areas),'n_clean_tp':sum(x['tp'] for x in matches.values()),'n_crossfit_proxy_groups':len(groups),
 'fold_grouping':'numeric filename prefix' if d.startswith('pcb') else 'stored group_id','central_width_height_fraction':.25,'central_area_fraction':.0625,'central_center_count':int(central),'central_center_fraction':float(central/len(areas)),
 'median_gt_area_px':float(np.median(areas)),'median_gt_area_fraction':float(np.median(normalized)),'area_tertile_edges_fraction':bounds.tolist(),
 'mean_recall_drop_pp':float(cc.recall_drop_pp.mean()),'peak_cell':maxrow.grid_label,'peak_drop_pp':float(maxrow.recall_drop_pp),
 'peak_clean_tp_exposure_fraction':float(maxrow.clean_tp_exposure_fraction),'spearman_drop_box_occupancy':float(spearmanr(cc.recall_drop_pp,cc.box_area_occupancy).statistic),
 'spearman_drop_clean_tp_exposure':float(spearmanr(cc.recall_drop_pp,cc.clean_tp_exposure_fraction).statistic),
 'raw_cell_drop_spatial_sd_pp':float(cc.recall_drop_pp.std(ddof=0)),'crossfit_residual_spatial_rms_pp':float(np.sqrt(np.mean(cc.descriptive_residual_drop_pp**2))),
 'overall_loss_risk_clean_tp':overall['loss_risk_clean_tp'],'loss_risk_when_hit':hrate['loss_risk_clean_tp'],'loss_risk_when_unhit':urate['loss_risk_clean_tp'],
 'lost_objects_hit_fraction':float(overall['n_lost_hit']/overall['n_lost']) if overall['n_lost'] else None,'overall_lost_object_observations':overall['n_lost'],'overall_recovered_object_observations':overall['n_recovered'],
 'n_object_observations':overall['n_gt'],'crossfit_fallback_observation_counts':dict(fallback),'peak_crossfit_expected_drop_pp':float(maxrow.crossfit_expected_drop_pp),'peak_residual_drop_pp':float(maxrow.descriptive_residual_drop_pp)}
 dump_json(out/f'{d}_summary.json',result);dump_json(out/f'{d}_sources.json',{'run':str(run),'run_hash':meta['run_hash'],'classes_sha256':digest_file(classes_path),'shards_sha256':hashes,'script_sha256':digest_file(__file__)})
 return result,cc


def numeric_split_check(rows):
 groups=collections.defaultdict(set);splits=collections.defaultdict(set)
 for r in rows:
  numeric=r['image_id'].split('_')[0];groups[numeric].add(r['split']);splits[r['split']].add(numeric)
 return {'numeric_groups_crossing_splits':sum(len(v)>1 for v in groups.values()),
  'numeric_groups_in_train_and_test':len(splits['train']&splits['test']),
  'n_numeric_groups_per_split':{k:len(v) for k,v in splits.items()}}


def make_split_scope_notes(runs):
 notes=[]
 for d,run in runs.items():
  if not d.startswith('pcb'):continue
  manifest=ROOT/f'data/canonical/{d}/manifest.jsonl';meta=json.loads((run/'run_meta.json').read_text())
  if not manifest.is_file() or digest_file(manifest)!=meta['manifest_sha256']:
   notes.append(f'{d}: 대응 canonical manifest를 검증하지 못했으므로 숫자 ID 분할 독립성을 추가 확인했다고 주장하지 않는다.');continue
  check=numeric_split_check(load_jsonl(manifest));cross=check['numeric_groups_crossing_splits'];shared=check['numeric_groups_in_train_and_test']
  if cross:
   notes.append(f'{d}: 기존 공식 분할에서는 숫자 ID {cross}개가 여러 분할에 걸치며 학습–시험 중복은 {shared}개다. 본 교차적합 분석은 모델 학습 시의 이런 중복을 고치지 않으므로 독립 숫자 ID 시험으로 해석하지 않는다.')
  else:
   counts=check['n_numeric_groups_per_split']
   notes.append(f'{d}: 대응 canonical manifest의 해시를 확인하고 숫자 ID를 직접 점검했다. 여러 분할에 걸치는 ID는 {cross}개, 학습–시험 중복은 {shared}개로, 학습/검증/시험 숫자 그룹 수는 {counts}다. 숫자 ID 단위 분리는 확인됐지만 실제 물리 보드 또는 코일의 정체성이 확인된 것은 아니다. 기존 공식 분할과 시험 영상·정답·학습 모델이 달라진 별도 코호트이므로 두 결과의 차이를 분할 중복 해소의 인과 효과로 단정하지 않는다.')
 return '\n\n'.join(notes)


def main():
 parser=argparse.ArgumentParser();parser.add_argument('--run',action='append',help='DATASET=run directory');parser.add_argument('--classes',action='append',help='DATASET=classes JSON');parser.add_argument('--out',type=Path,default=DEFAULT_OUT);args=parser.parse_args();out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
 runs={d:ROOT/f'runs/evaluation/{d}_s11_bright25_g8_p10' for d in ['gc10_det','neu_det','pcb_ind']} if not args.run else {s.split('=',1)[0]:Path(s.split('=',1)[1]).resolve() for s in args.run}
 classes={s.split('=',1)[0]:Path(s.split('=',1)[1]).resolve() for s in args.classes or []};results=[];tables=[]
 for d,run in runs.items():
  result,table=summarize_dataset(d,run,classes.get(d,ROOT/f'data/canonical/{d}/classes.json'),out);results.append(result);tables.append(table)
 pd.DataFrame(results).drop(columns=['crossfit_fallback_observation_counts']).to_csv(out/'dataset_summary.csv',index=False)
 n=len(results);fig,axes=plt.subplots(2,n,figsize=(6*n,10),squeeze=False,layout='constrained');lim=max(abs(t[['recall_drop_pp','descriptive_residual_drop_pp']].to_numpy()).max() for t in tables)
 for j,(result,t) in enumerate(zip(results,tables)):
  for i,key in enumerate(['recall_drop_pp','descriptive_residual_drop_pp']):
   im=axes[i,j].imshow(t[key].to_numpy().reshape(8,8),cmap='RdBu_r',vmin=-lim,vmax=lim)
   axes[i,j].set_title(result['title']+(' | observed' if i==0 else ' | observed - crossfit expectation'));axes[i,j].set_xticks([]);axes[i,j].set_yticks([])
 fig.colorbar(im,ax=list(axes.flat),label='Recall drop / residual (percentage points)');fig.savefig(out/'raw_and_descriptive_residual_maps.png',dpi=170);plt.close(fig)
 fig,axes=plt.subplots(1,n,figsize=(6*n,5),squeeze=False,layout='constrained')
 for ax,result,t in zip(axes.flat,results,tables):
  ax.scatter(100*t.clean_tp_exposure_fraction,t.recall_drop_pp,c=t.box_area_occupancy,cmap='viridis');ax.set_title(result['title']);ax.set_xlabel('Clean-TP boxes touched by commanded fault sites (%)');ax.set_ylabel('Recall drop (pp)')
 fig.savefig(out/'exposure_vs_recall_drop.png',dpi=170);plt.close(fig)
 table='| 데이터셋 | 중앙 결함 중심 비율 | 정답 상자 면적 중앙값 | 최대 감소 위치 | 최대 위치 clean TP 겹침 | 감소–노출 순위상관 |\n|---|---:|---:|---|---:|---:|\n'
 for r in results:table+=f"| {r['title']} | {100*r['central_center_fraction']:.2f}% | {r['median_gt_area_px']:.1f}px² | {r['peak_cell']} ({r['peak_drop_pp']:.3f}%p) | {100*r['peak_clean_tp_exposure_fraction']:.2f}% | {r['spearman_drop_clean_tp_exposure']:.3f} |\n"
 risk='| 데이터셋 | 겹친 clean TP의 손실률 | 비겹침 clean TP의 손실률 | 손실 객체 중 겹침 비율 | 원래 위치차 표준편차 | 교차적합 잔차 RMS |\n|---|---:|---:|---:|---:|---:|\n'
 for r in results:risk+=f"| {r['title']} | {100*r['loss_risk_when_hit']:.3f}% | {100*r['loss_risk_when_unhit']:.3f}% | {100*r['lost_objects_hit_fraction']:.2f}% | {r['raw_cell_drop_spatial_sd_pp']:.3f}%p | {r['crossfit_residual_spatial_rms_pp']:.3f}%p |\n"
 split_scope=make_split_scope_notes(runs)
 report=f'''# 기존 캐시의 제품 결함 분포·겹침 진단

## 자료와 정의

밝은 화소25개, 8×8위치, 분산형 패턴10개의 저장 예측을 사용했다. 모델 또는 입력 영상을 새로 추론하지 않았다. confidence와 matching IoU는 각 run의 기존 설정을 유지했다. 원본과 손상 예측은 같은 클래스·IoU의 일대일 대응으로 각 GT 객체의 검출 여부를 계산했다.

- **손실:** clean에서는 검출됐지만 fault에서는 검출되지 않은 GT.
- **회복:** clean에서는 미검출이었지만 fault에서는 검출된 GT.
- **겹침:** 주입 대상으로 지정한 좌표 중 하나 이상이 GT 상자 내부에 있음. 상자는 0-based half-open으로 계산했다. 원래255였던 좌표도 주입 대상에 포함하므로 실제 변경 화소와 일치하지 않을 수 있다.
- **중앙:** 가로·세로의 중앙25% 구간(.375~.625), 즉 영상 면적6.25% 영역에 GT 상자 중심이 있는 경우.
- **크기:** 데이터셋별 상대 GT 상자 면적을 3분위로 분류했다. 원본 데이터 객체를 기준으로 경계를 정하며 반복 관측으로 다시 정하지 않는다.

## 결함 분포와 노출

{table}

![노출과 성능 변화](exposure_vs_recall_drop.png)

## 객체별 손실·회복

{risk}

이 비율의 분모는 반복된 **객체 관측 수**다. 같은 원본 객체가640번 재사용되므로 독립 객체 수로 해석하거나 단순 이항 신뢰구간을 붙이지 않았다. 클래스·크기·겹침 화소수별 모든 손실/회복 분모와 분자는 `*_class_size_hit_strata.csv`에 있다. 각 칸과 패턴별 집계는 `*_pattern_cell_events.csv`에 보존했다.

## cell을 사용하지 않는 기술적 기준모형

클래스×크기3분위×상자 안 주입 화소수(none/one/2~4/5+)로 원래 검출 상태에 따른 손실/회복률을 학습했다. 위치 번호(cell)는 예측변수로 사용하지 않았다. 같은 이미지 그룹이 학습·평가 fold에 함께 들어가지 않도록 그룹5-fold 교차적합(seed8811)을 사용했다. PCB는 t/b를 합친 숫자 접두어를 fold 그룹으로 사용했고 다른 데이터는 저장된group_id를 사용했다. 숫자 접두어의 실제 제품 정체성이 확인된 것은 아니다.

평가 fold의 표본이 없는 세부 층은 클래스 제거→크기 제거→겹침 제거 순서로 더 넓은 학습 층의 비율을 사용했다. 모든 fallback 관측 수를 summary JSON에 기록했다. 이 모형이 각 위치의 객체 노출 구성에서 예측한 순손실(예상손실−예상회복)을 기존 Recall 감소에서 뺀 값을 **기술적 잔차**로 표시한다.

![관찰 지도와 기술적 잔차](raw_and_descriptive_residual_maps.png)

겹침·클래스·크기 구성만으로 원래 공간 패턴이 얼마나 재현되는지 보는 진단이며 **인과적 위치 효과를 분리했다고 주장하지 않는다.** 상자 안에서도 실제 결함 픽셀·세부 구조·손상 대비·confidence·NMS·문맥이 다를 수 있다. 잔차가 남아도 센서 위치 고유 취약성의 증거가 아니며, 잔차가 줄어도 원인을 모두 설명했다는 뜻은 아니다. 순위상관도 교란 제거 또는 독립성 검정이 아니다.

## 실험 범위

단일 학습 시드와 해당 시험 분할에 조건부인 탐색 분석이다.

{split_scope}

데이터셋별 해상도와 resize 배율 차이, 미확인 실제 코일/보드 독립성, GT 상자의 느슨함, 25개 주입 대상 중 실제 무변화 좌표도 해석에 남는다.

입력 run hash·예측 파일 SHA256·분석 코드 해시는 `*_sources.json`에 기록했다. 객체 집계에서 재계산한 모든64개 위치 Recall 감소가 기존 `grid_summary.csv`와1e-10 이내로 일치함을 검사했다.
'''
 (out/'REPORT_KO.md').write_text(report);dump_json(out/'analysis_meta.json',{'script_sha256':digest_file(__file__),'datasets':list(runs),'n_folds':5,'fold_seed':8811,'model':'out-of-fold stratified descriptive event rates, no cell predictor','causal_control':False})
 print(pd.DataFrame(results).drop(columns=['crossfit_fallback_observation_counts']).to_string(index=False));print(out/'REPORT_KO.md')

if __name__=='__main__':main()
