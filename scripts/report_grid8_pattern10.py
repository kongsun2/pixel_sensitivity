"""Per-pattern means across all64locations, plus equal-weight ten-pattern means."""
from pathlib import Path
import json,sys
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from core import digest_file,dump_json
OUT=ROOT/'reports/bright25_grid8_pattern10_2026-11-10';OUT.mkdir(parents=True,exist_ok=True)
FONT=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc');plt.rcParams['axes.unicode_minus']=False
DATASETS=[('gc10_det','GC10-DET'),('neu_det','NEU-DET'),('pcb_ind','PCB-IND')]
METRICS={'recall':'recall','precision':'precision','map50':'coco_map50','map50_95':'coco_map50_95','f1':'f1'}
pattern_rows=[];totals=[];baselines=[];input_hashes={}
for d,title in DATASETS:
 root=ROOT/f'runs/evaluation/{d}_s11_bright25_g8_p10';a=root/'analysis';meta=json.loads((root/'run_meta.json').read_text());base=json.loads((a/'clean_metrics.json').read_text());m=pd.read_csv(a/'condition_metrics.csv');s=pd.read_csv(a/'grid_summary.csv');pairs=pd.read_csv(a/'paired_images.csv')
 if meta['config']['repeats']!=10 or len(m)!=640 or len(s)!=64 or len(pairs)!=640*base['n_images']:raise ValueError('Incomplete ten-pattern data')
 if len(list((root/'predictions').glob('*.done.json')))!=641:raise ValueError('Inference shard count differs')
 if not pairs.count_commanded.eq(25).all():raise ValueError('Fault budget changed')
 input_hashes[d]={'run_hash':meta['run_hash'],'condition_metrics_sha256':digest_file(a/'condition_metrics.csv'),'paired_images_sha256':digest_file(a/'paired_images.csv')}
 baselines.append({'dataset':title,'n_test':base['n_images'],'n_gt':base['n_gt'],**{name+'_pct':100*base[key] for name,key in METRICS.items()}})
 these=[]
 for rep in range(10):
  group=m[m['repeat']==rep].sort_values('cell')
  if group.cell.tolist()!=list(range(64)):raise ValueError('Each pattern needs 64 locations')
  pp=pairs[pairs['repeat']==rep]
  r={'dataset':title,'pattern_id':rep+1,'pattern':f'P{rep+1:02d}','repeat_index':rep,'prediction_origin':'verified reuse of prior8x8' if rep<3 else 'new inference',
   'n_locations':64,'n_test_images_per_location':base['n_images'],'n_gt_per_location':base['n_gt']}
  for name,key in METRICS.items():
   r['clean_'+name+'_pct']=100*base[key];r[name+'_pct']=100*group[key].mean();r[name+'_drop_pp']=100*(base[key]-group[key].mean())
  r['fp_per_image']=group.fp_per_image.mean();r['actual_changed_pixels_mean']=pp.count_changed.mean()
  r['confidence_drop_clean_tp']=pp.clean_tp_conf_drop_sum.sum()/pp.n_clean_tp.sum() if pp.n_clean_tp.sum() else None
  r['min_location_recall_drop_pp']=100*group.sensitivity_recall.min();r['max_location_recall_drop_pp']=100*group.sensitivity_recall.max()
  these.append(r);pattern_rows.append(r)
 frame=pd.DataFrame(these);r={'dataset':title,'n_patterns':10,'n_locations_per_pattern':64,'n_test':base['n_images'],'n_gt':base['n_gt'],'n_evaluations_including_clean':641*base['n_images'],'n_new_evaluations':448*base['n_images']}
 for name,key in METRICS.items():
  r['clean_'+name+'_pct']=100*base[key];r['mean_'+name+'_pct']=frame[name+'_pct'].mean();r['mean_'+name+'_drop_pp']=frame[name+'_drop_pp'].mean()
  if not np.isclose(r['mean_'+name+'_pct'],100*m[key].mean(),atol=1e-10):raise ValueError('Pattern mean does not equal all-location mean')
 r['pattern_sd_recall_drop_pp']=frame.recall_drop_pp.std(ddof=1);r['min_pattern_mean_recall_drop_pp']=frame.recall_drop_pp.min();r['max_pattern_mean_recall_drop_pp']=frame.recall_drop_pp.max()
 peak=s.loc[s.sensitivity_pp.idxmax()];r['max_location_ten_pattern_mean_drop_pp']=peak.sensitivity_pp;r['max_location']=peak.grid_label
 r['max_location_ci_low_pp']=100*peak.pointwise_ci_low;r['max_location_ci_high_pp']=100*peak.pointwise_ci_high
 r['mean_actual_changed_pixels']=frame.actual_changed_pixels_mean.mean();totals.append(r)
patterns=pd.DataFrame(pattern_rows);overall=pd.DataFrame(totals);clean=pd.DataFrame(baselines)
patterns.to_csv(OUT/'per_pattern_dataset_means.csv',index=False);overall.to_csv(OUT/'dataset_overall_means.csv',index=False);clean.to_csv(OUT/'clean_baselines.csv',index=False)
wide_recall=patterns.pivot(index='pattern_id',columns='dataset',values='recall_pct').reindex(columns=[t for _,t in DATASETS]);wide_drop=patterns.pivot(index='pattern_id',columns='dataset',values='recall_drop_pp').reindex(columns=[t for _,t in DATASETS]);wide_ap=patterns.pivot(index='pattern_id',columns='dataset',values='map50_pct').reindex(columns=[t for _,t in DATASETS])
wide_precision=patterns.pivot(index='pattern_id',columns='dataset',values='precision_pct').reindex(columns=[t for _,t in DATASETS]);wide_ap95=patterns.pivot(index='pattern_id',columns='dataset',values='map50_95_pct').reindex(columns=[t for _,t in DATASETS])
for name,frame in [('recall_by_pattern',wide_recall),('recall_drop_by_pattern',wide_drop),('map50_by_pattern',wide_ap),('precision_by_pattern',wide_precision),('map50_95_by_pattern',wide_ap95)]:frame.to_csv(OUT/(name+'.csv'))
fig,ax=plt.subplots(figsize=(16,7));x=np.arange(10);colors=['#4677a9','#ef9b39','#bd4855'];width=.25
for i,(d,title) in enumerate(DATASETS):
 vals=patterns[patterns.dataset==title].sort_values('pattern_id').recall_drop_pp.to_numpy();bars=ax.bar(x+(i-1)*width,vals,width=width,label=title,color=colors[i])
 ax.bar_label(bars,labels=[f'{v:.3f}' for v in vals],padding=3,fontsize=8)
ax.axhline(0,color='#58636e',lw=.8);ax.set_xticks(x,[f'P{i+1:02d}' for i in range(10)]);ax.set_ylabel('평균 Recall 감소 (%p)',fontproperties=FONT,fontsize=13);ax.set_xlabel('분산형 무작위 배치 패턴',fontproperties=FONT,fontsize=13)
ax.set_title('패턴별·데이터셋별 평균 성능 변화\n밝은 화소25개 · 8×8의 64개 위치 평균 · 고정 학습 모델',fontproperties=FONT,fontsize=18,pad=16);ax.legend();ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
fig.text(.5,.015,'P01~P03: 검증한 기존 결과 재사용 | P04~P10: 신규 추론 | 양수는 감소, 음수는 개선',fontproperties=FONT,fontsize=11,ha='center');fig.tight_layout(rect=(0,.05,1,1));fig.savefig(OUT/'pattern10_recall_drop_comparison.png',dpi=220);fig.savefig(OUT/'pattern10_recall_drop_comparison.pdf');plt.close(fig)
# Show the exact ten input masks independently of their measured effect.
allp=json.loads((OUT/'patterns.json').read_text())
for d,title in DATASETS:
 pats=sorted([p for p in allp if p['dataset']==d],key=lambda p:p['pattern_id']);fig,axes=plt.subplots(2,5,figsize=(16,7))
 for ax,p in zip(axes.flat,pats):
  off=np.asarray(p['offsets_yx']);w=p['width']//8;h=p['height']//8;ax.scatter(off[:,1],off[:,0],s=15,c='#d92332');ax.set_xlim(-.5,w-.5);ax.set_ylim(h-.5,-.5);ax.set_aspect('equal');ax.set_title(f"P{p['pattern_id']:02d}");ax.set_xticks([]);ax.set_yticks([])
 fig.suptitle(f'{title} | 10개 분산형 배치 패턴 | 칸당25개',fontproperties=FONT,fontsize=20);fig.text(.5,.015,'표시한 칸 내부 패턴을 64개 위치로 각각 이동합니다. 점은 가시성을 위해 확대 표시했습니다.',fontproperties=FONT,fontsize=11,ha='center');fig.tight_layout(rect=(0,.05,1,.92),h_pad=3.0,w_pad=1.2);fig.savefig(OUT/f'{d}_ten_pattern_positions.png',dpi=180);plt.close(fig)

def wide_table(frame,label,avg_col):
 text='| 패턴 | GC10-DET | NEU-DET | PCB-IND |\n|---|---:|---:|---:|\n'
 for rep,row in frame.iterrows():text+=f'| P{rep:02d} | '+ ' | '.join(f'{row[t]:.3f}' for _,t in DATASETS)+' |\n'
 text+='| **10개 전체 평균** | '+' | '.join(f"**{overall.loc[overall.dataset==t,avg_col].iloc[0]:.3f}**" for _,t in DATASETS)+' |\n'
 return text
base_table='| 데이터셋 | 시험 영상 / GT | 원본 Recall | 원본 Precision | 원본 mAP50 | 원본 mAP50:95 |\n|---|---:|---:|---:|---:|---:|\n'
for r in baselines:base_table+=f"| {r['dataset']} | {r['n_test']} / {r['n_gt']} | {r['recall_pct']:.3f}% | {r['precision_pct']:.3f}% | {r['map50_pct']:.3f}% | {r['map50_95_pct']:.3f}% |\n"
total_table='| 데이터셋 | 주입 후 평균 Recall | 평균 Recall 감소 | 주입 후 평균 Precision | 주입 후 평균 mAP50 | 주입 후 평균 mAP50:95 |\n|---|---:|---:|---:|---:|---:|\n'
for r in totals:total_table+=f"| {r['dataset']} | {r['mean_recall_pct']:.3f}% | {r['mean_recall_drop_pp']:.3f}%p | {r['mean_precision_pct']:.3f}% | {r['mean_map50_pct']:.3f}% | {r['mean_map50_95_pct']:.3f}% |\n"
sd_table='| 데이터셋 | 패턴별 평균 감소 최솟값 | 최댓값 | 패턴 간 표준편차 |\n|---|---:|---:|---:|\n'
for r in totals:sd_table+=f"| {r['dataset']} | {r['min_pattern_mean_recall_drop_pp']:.3f}%p | {r['max_pattern_mean_recall_drop_pp']:.3f}%p | {r['pattern_sd_recall_drop_pp']:.3f}%p |\n"
report=f'''# 핫픽셀 10개 배치 패턴 — 데이터셋별 평균 결과

## 실험 정의

기존 학습된 YOLOv8n과 시험자료를 그대로 사용하고, RGB=255 고립 화소25개를 8×8 격자 안에 주입했다. **서로 다른 분산형 무작위 배치10개**를 비교한 실험이다. 군집·선형 등 서로 다른 손상 형태10종을 비교한 것은 아니다. 같은 패턴 안에서는 배치를 유지하고 64개 위치로 옮긴다. P번호는 각 데이터셋 내 패턴 번호이며, 데이터셋·해상도가 난수 seed에 포함되므로 같은 P번호라도 데이터셋 간 배치 형상이 같다는 뜻은 아니다.

패턴 P01~P03은 직전 8×8 실험의 결과다. 입력·모델·소스·환경·모든 주입/평가 설정이 동일한지 확인하고 예측 SHA256과 출처를 기록해 재사용했다. P04~P10은 새로 추론했다. 난수 seed와 정확한 칸 내부 좌표는 [patterns.json](patterns.json)에 저장했다. 패턴은 성능을 보고 선택하지 않았다.

설정: 화소25개, alpha=1, 모든 RGB채널255, grid8×8, seed base301, 패턴 인덱스0~9. imgsz640, conf operating0.25, matching IoU0.5, prediction floor0.001, NMS IoU0.7, max_det300, 같은 공식 COCO evaluator(maxDets100). 재학습하지 않았다.

## 평균 계산

- **패턴별 평균:** 해당 데이터셋·패턴의 64개 위치별 평가값을 동일 가중치로 평균.
- **데이터셋 총평균:** 패턴10개 평균을 동일 가중치로 평균. 즉64×10=640개 손상 평가값의 평균과 같다.
- 각 위치의 Recall은 그 데이터셋 시험자료 전체의 TP/GT로 계산한다. 이미지별 Recall을 먼저 평균하지 않는다.
- AP는 위치·패턴별 전체 시험자료를 공식 COCO evaluator로 평가한 뒤 평균한다. 서로 다른 패턴의 예측을 한 영상에 합치지 않는다.
- 감소량은 원본 − 주입 후이며 양수는 악화, 음수는 개선이다.

## 원본 기준 성능

{base_table}

## 패턴별 평균 Recall (%)

{wide_table(wide_recall,'Recall','mean_recall_pct')}

## 패턴별 평균 Recall 감소 (%p)

{wide_table(wide_drop,'Recall 감소','mean_recall_drop_pp')}

![패턴별 평균 성능 감소](pattern10_recall_drop_comparison.png)

## 패턴별 평균 mAP50 (%)

{wide_table(wide_ap,'mAP50','mean_map50_pct')}

## 패턴별 평균 Precision (%)

{wide_table(wide_precision,'Precision','mean_precision_pct')}

## 패턴별 평균 mAP50:95 (%)

{wide_table(wide_ap95,'mAP50:95','mean_map50_95_pct')}

## 데이터셋별 총평균

{total_table}

## 배치 패턴에 따른 변동

{sd_table}

표준편차는 패턴별 64위치 평균10개의 표본 표준편차이며 학습 시드 불확실성 또는 신뢰구간을 뜻하지 않는다. 모든 위치와 모든 패턴을 포함했고 음수 변화도 유지했다.

## 입력 패턴 시각화

![GC10 패턴](gc10_det_ten_pattern_positions.png)
![NEU 패턴](neu_det_ten_pattern_positions.png)
![PCB 패턴](pcb_ind_ten_pattern_positions.png)

## 실행 증거

데이터셋마다 clean1+64×10=641개 완료 예측 묶음이다. 총700,613회분의 이미지 평가 결과를 사용했다. 그중210,949회분은 검증한 기존 예측이고489,664회는 신규 추론이다. 새 결과 폴더의 완료 표식에 기존 run hash·원본 파일·SHA256을 명시했다.

분석은 데이터셋마다 위치별 요약64행, 위치/패턴별 지표640행, 전체 합계699,520행의 쌍체 이미지 기록을 생성한다. 각 위치의 평균지도와2,000회 pointwise bootstrap 구간은 `runs/evaluation/{{dataset}}_s11_bright25_g8_p10/analysis/`에 저장했다. 해당 구간은 지도 전체 다중 비교 보정이나 패턴 평균 간 유의성 검정이 아니다.

## 해석 범위

모든 결과는 동일한 학습 시드의 고정 모델에 조건부다. 원본 해상도(GC10 2048×1000, NEU200×200, PCB300×300)와 resize 배율이 달라 동일25개가 동일 손상 비율 또는 입력 면적을 뜻하지 않는다. PCB는 중앙 ROI 분포와 공식 분할 내 숫자 ID의 양면 중복 가능성이 있으며, NEU도 실제 제품/코일 그룹이 확인되지 않았다. 실제 RAW/ISP/DPC 또는 출하 불량률은 검증하지 않았다.

## 수치 파일

- [패턴별·데이터셋별 모든 평균 지표](per_pattern_dataset_means.csv)
- [데이터셋별 총평균](dataset_overall_means.csv)
- [패턴별 Recall 표](recall_by_pattern.csv)
- [패턴별 감소량 표](recall_drop_by_pattern.csv)
- [패턴별 mAP50 표](map50_by_pattern.csv)
- [패턴별 Precision 표](precision_by_pattern.csv)
- [패턴별 mAP50:95 표](map50_95_by_pattern.csv)
'''
(OUT/'REPORT_KO.md').write_text(report);dump_json(OUT/'report_meta.json',{'source_sha256':digest_file(__file__),'inputs':input_hashes,'n_pattern_dataset_rows':len(patterns),'n_overall_rows':len(overall)})
print('PER-PATTERN RECALL (%)');print(wide_recall.to_string());print('OVERALL');print(overall.to_string(index=False));print(OUT/'REPORT_KO.md')
