"""Generate report and shareable figures from complete prediction caches."""
import json,sys,csv
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from core import load_jsonl,match_predictions,dump_json,inject_fault
from run import load_clean
run=ROOT/'runs/evaluation/gc10_s11_frozen';out=run/'analysis'
meta=json.loads((run/'run_meta.json').read_text());cfg=meta['config']
base=json.loads((out/'clean_metrics.json').read_text());src=json.loads((ROOT/'inputs/gc10_source_audit.json').read_text())
summary=pd.read_csv(out/'grid_summary.csv');metrics=pd.read_csv(out/'condition_metrics.csv');occupancy=pd.read_csv(out/'occupancy.csv')
condition=[]
for name,g in summary.groupby('condition_id',sort=False):
 m=metrics[metrics.condition_id==name];worst=g.loc[g.sensitivity_pp.idxmax()]
 condition.append({'condition':name,'mean_recall_drop_pp':g.sensitivity_pp.mean(),'min_cell_drop_pp':g.sensitivity_pp.min(),
  'max_cell_drop_pp':g.sensitivity_pp.max(),'max_cell':worst.grid_label,
  'mean_map50_drop_pp':100*(base['coco_map50']-m.coco_map50.mean()),
  'mean_map50_95_drop_pp':100*(base['coco_map50_95']-m.coco_map50_95.mean()),
  'mean_confidence_drop':g.confidence_drop_clean_tp.mean(),'actual_changed_pixels':g.mean_actual_changed_pixels.mean()})
ct=pd.DataFrame(condition);ct.to_csv(out/'condition_summary.csv',index=False)
fig,axes=plt.subplots(3,4,figsize=(17,12),layout='constrained')
lim=max(float(summary.sensitivity_pp.abs().max()),.01)
for ax,(name,g) in zip(axes.flat,summary.groupby('condition_id',sort=False)):
 ar=g.sort_values('cell').sensitivity_pp.to_numpy().reshape(4,4)
 im=ax.imshow(ar,cmap='RdBu_r',vmin=-lim,vmax=lim)
 for y in range(4):
  for x in range(4):ax.text(x,y,f'G{y*4+x+1}\n{ar[y,x]:.2f}',ha='center',va='center',fontsize=8)
 ax.set_title(name);ax.set_xticks(range(4));ax.set_yticks(range(4))
for ax in list(axes.flat)[len(ct):]:ax.axis('off')
fig.colorbar(im,ax=list(axes.flat),label='Recall drop (percentage points)',shrink=.8)
fig.suptitle('GC10-DET | YOLOv8n | 3 pattern repetitions | fixed 0.25 confidence')
fig.savefig(out/'all_sensitivity_maps.png',dpi=170);plt.close(fig)
fig,ax=plt.subplots(figsize=(6,5));ar=occupancy.box_area_occupancy.to_numpy().reshape(4,4)
im=ax.imshow(ar,cmap='YlOrRd');fig.colorbar(im,ax=ax,label='Bounding-box area occupancy')
for y in range(4):
 for x in range(4):ax.text(x,y,f'{ar[y,x]:.3f}',ha='center',va='center')
ax.set_title('GT box occupancy (not a segmentation mask)');fig.tight_layout();fig.savefig(out/'occupancy_map.png',dpi=170);plt.close(fig)
fig,axes=plt.subplots(3,4,figsize=(16,11),layout='constrained');correlations=[]
for ax,(name,g) in zip(axes.flat,summary.groupby('condition_id',sort=False)):
 g=g.sort_values('cell');x=occupancy.box_area_occupancy.to_numpy();y=g.sensitivity_pp.to_numpy()
 rho=float(spearmanr(x,y).statistic) if np.ptp(y)>0 else None
 correlations.append({'condition':name,'spearman_rho':rho,'n_grid_cells':16})
 ax.scatter(x,y);ax.axhline(0,color='grey',lw=.5);ax.set_title(name+f' | rho={rho:.2f}' if rho is not None else name+' | constant')
 ax.set_xlabel('GT box occupancy');ax.set_ylabel('Recall drop (pp)')
for ax in list(axes.flat)[len(ct):]:ax.axis('off')
fig.savefig(out/'occupancy_vs_sensitivity.png',dpi=150);plt.close(fig);dump_json(out/'occupancy_correlations.json',correlations)
# Per-class clean recall with the same operational matching used in the main table.
clean=load_jsonl(run/'predictions/clean.jsonl');names=json.loads((ROOT/'inputs/gc10_classes.json').read_text());class_rows=[]
for cl,name in enumerate(names):
 tp=fp=fn=0
 for r in clean:
  m=match_predictions([g for g in r['gt'] if g['class_id']==cl],[p for p in r['pred'] if p['class_id']==cl],.25,.5)
  tp+=m['tp'];fp+=m['fp'];fn+=m['fn']
 class_rows.append({'class':name,'n_gt':tp+fn,'tp':tp,'fp':fp,'fn':fn,'recall':tp/(tp+fn) if tp+fn else None})
pd.DataFrame(class_rows).to_csv(out/'clean_per_class.csv',index=False)
# Show precisely the first declared pattern: never pick injection coordinates using GT.
first=load_jsonl(ROOT/'data/canonical/gc10_det/manifest.jsonl');first=next(r for r in first if r['image_id']==clean[0]['image_id'])
rgb=load_clean(first);fault=load_jsonl(run/'predictions/bright_cluster9__r000__g000.jsonl')[0]['fault']
x,y=fault['coordinates_xy'][0];offsets=np.array([[yy-fault['bounds_xyxy'][1],xx-fault['bounds_xyxy'][0]] for xx,yy in fault['coordinates_xy']])
bad,info=inject_fault(rgb,offsets,tuple(fault['bounds_xyxy']),255)
x0=max(0,x-24);y0=max(0,y-24);x1=min(rgb.shape[1],x+25);y1=min(rgb.shape[0],y+25)
fig,axes=plt.subplots(1,3,figsize=(13,5));axes[0].imshow(rgb);axes[0].plot(x,y,'r+',ms=10);axes[0].set_title(first['image_id']+' | fault site')
axes[1].imshow(rgb[y0:y1,x0:x1],interpolation='nearest');axes[1].set_title('Clean crop')
axes[2].imshow(bad[y0:y1,x0:x1],interpolation='nearest');axes[2].set_title('3x3 bright cluster crop')
for ax in axes:ax.axis('off')
fig.tight_layout();fig.savefig(out/'fault_example.png',dpi=180);plt.close(fig)
# Inference execution total excludes training and analysis.
shards=list((run/'predictions').glob('*.done.json'))
wall=sum(json.loads(p.read_text())['seconds_wall'] for p in shards)
parallel_wall=max(p.stat().st_mtime for p in shards)-(run/'run_meta.json').stat().st_mtime
train=pd.read_csv(ROOT/'runs/train/gc10_yolov8n_s11/results.csv');train.columns=train.columns.str.strip()
table='| 조건 | 평균 Recall 감소 (%p) | 위치별 최소~최대 (%p) | 최대 위치 | 평균 mAP50 감소 (%p) |\n|---|---:|---:|---|---:|\n'
for r in condition:table+=f"| {r['condition']} | {r['mean_recall_drop_pp']:.4f} | {r['min_cell_drop_pp']:.4f} ~ {r['max_cell_drop_pp']:.4f} | {r['max_cell']} | {r['mean_map50_drop_pp']:.4f} |\n"
best=ct.loc[ct.max_cell_drop_pp.idxmax()]
report=f'''# GC10-DET 화소 손상 위치별 민감도 실험 결과

## 실행 범위

실제 GC10 영상에서 YOLOv8n을 학습하고 동일한 모델로 원본/손상 추론 및 평가를 완료했다. 단일 데이터셋과 학습 시드 11에 대한 결과이며 반복 학습 시드와 다른 제조 데이터셋 재현성은 아직 검증하지 않았다.

## 데이터와 분할

- 출처: [dronefreak/GC10-DET](https://huggingface.co/datasets/dronefreak/GC10-DET), revision `42c086fa8dea913987febe794b1e7608b3db464d`. 원저자 직접 배포본이 아닌 Dataset Ninja 기반 비공식 재배포본이다.
- 원본 {src['n_source']}장; 사용 {src['n_retained']}장. 주석 없는 영상 {len(src['unverified_empty_label_images_excluded'])}장은 정상 여부가 확인되지 않아 제외했다. 정확히 동일한 decoded 영상의 제외 수: {len(src['exact_duplicates_excluded'])}.
- 실제 분할: {src['split_counts']}. 파일명의 마지막 프레임 번호를 제외한 접두어를 그룹으로 간주하여 seed=101, 그룹 수 기준 70/15/15 분할했다.
- 기본 배포 분할에서 접두어 그룹 {src['mirror_prefix_groups_crossing_splits']}개가 분할 경계를 넘었다. 재분할 후 동일 접두어는 하나의 분할에만 존재한다. 접두어는 촬영 시퀀스의 대리값이며 실제 코일/제품 독립성을 보장하지 않는다. 근접 중복의 시각 검수는 수행하지 않았다.

## 사전에 고정한 설정

Ultralytics 8.3.221, COCO 사전학습 `yolov8n.pt`, imgsz=640, 최대 100 epoch, patience=20, AdamW lr=0.001, batch=16. 검증 데이터 기준 best.pt 사용. 실제 완료 epoch: {len(train)}. 합성 손상 없이 학습했으며 공간/색상 증강은 0으로 설정했다.

손상은 원본 해상도 RGB에 주입한 뒤 YOLO 전처리를 통과한다. 4×4 격자, 조건 11개, 패턴 3회 반복. 고립 화소 1/9/25개, 검정/밝은 고정값, 128 고정값, 3×3 군집, alpha=0.5 부분 변화. `hot_like`는 255 치환이며 온도/노출에 따른 dark-current 물리 모형은 아니다. 손상량과 세부 패턴은 `configs/gc10_frozen.yaml`에 있으며 시험 결과를 보기 전에 고정했다.

같은 해상도에서는 동일 반복의 패턴을 이미지마다 유지하고 격자 위치만 이동한다. 클래스, 조건, 위치에 따라 평가 기준을 재조정하지 않는다. conf=0.25, class-aware score-first matching, IoU=0.5. AP는 COCO 공식 평가기, conf floor=0.001, maxDets=100을 사용한다.

## 원본 시험 성능

| 지표 | 값 |
|---|---:|
| 시험 이미지 | {base['n_images']} |
| GT 제품 결함 | {base['n_gt']} |
| TP / FP / FN | {base['tp']} / {base['fp']} / {base['fn']} |
| Recall | {100*base['recall']:.3f}% |
| Precision | {100*base['precision']:.3f}% |
| COCO mAP50 | {100*base['coco_map50']:.3f}% |
| COCO mAP50:95 | {100*base['coco_map50_95']:.3f}% |

클래스별 원본 성능은 `clean_per_class.csv`에 저장했다. 이 원본 기준도 미검출을 포함하므로 높은 신뢰성의 산업 배포 성능으로 해석하지 않는다.

## 위치별 변화

아래 평균은 16개 위치와 3개 패턴 반복의 평균이다. 양수는 원본보다 성능 감소, 음수는 증가다. 최대 위치는 사후 기술통계이며 유의한 취약 위치로 확정하지 않는다.

{table}

![위치별 Recall 감소](all_sensitivity_maps.png)

관찰한 가장 큰 위치별 평균 Recall 감소는 `{best['condition']}`의 {best['max_cell']}에서 {best['max_cell_drop_pp']:.4f}%p였다. 단일 GT의 검출 여부도 작은 시험집합에서는 지도 값에 영향을 준다. 작은 손상은 resize로 약화될 수 있으며, 효과가 작거나 개선된 조건도 모두 유지했다.

`grid_summary.csv`의 95% 구간은 이미지 접두어 그룹과 패턴 반복을 함께 재표집한 **위치별 쌍체 bootstrap 구간**이다. 지도 전체 동시 구간·다중 비교 보정 또는 학습 시드 불확실성을 포함하지 않는다. 각 위치의 구간이 0을 벗어났다는 이유만으로 위치 간 차이를 검증했다고 주장할 수 없다.

![GT 위치 분포](occupancy_map.png)
![GT 분포와 민감도](occupancy_vs_sensitivity.png)

GT 분포는 bounding box 점유율이며 실제 결함 픽셀 마스크가 아니다. 격자 16칸의 순위 상관은 기술적 보조 분석으로 `occupancy_correlations.json`에 저장했다. 상관만으로 위치 효과의 교란을 통제한 것은 아니다.

![손상 예시](fault_example.png)

## 실행 증거와 재현

완료된 추론 shard {len(shards)}개, 합계 {len(shards)*base['n_images']:,}장 평가, 두 GPU 평가 경과시간 약 {parallel_wall/60:.1f}분, 개별 추론 shard 시간 합계 {wall/60:.1f}분(병렬 작업의 합계). 모델/데이터/조건/코드/패키지 버전 해시는 `run_meta.json`과 `analysis_meta.json`에 기록했다. 원본·조건별 예측 JSONL과 완료 SHA256을 보존했다. 기능 테스트와 COCO 평가기는 실제 환경에서 실행했다. 정확한 재실행 순서는 프로젝트 루트 `README.md`를 따른다.

## 해석의 범위

이 결과는 공개 이미지 좌표에서 합성 화소 이상을 넣었을 때 고정된 검출 모델의 출력 변화를 측정한다. 실제 센서 RAW, demosaic, DPC, 센서 수명, 실제 제품 출하 불량률은 검증하지 않았다. 추가 시드/데이터셋, 물리적 센서 결함, 제품별 독립성 확인을 거쳐야 일반화할 수 있다.
'''
(out/'REPORT_KO.md').write_text(report)
print(report)
