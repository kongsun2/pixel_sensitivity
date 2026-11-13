"""Report each completed replication and compare unchanged bright conditions to GC10."""
import argparse,json,sys
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from core import load_jsonl,dump_json,digest_file
p=argparse.ArgumentParser();p.add_argument('--dataset',required=True,choices=['neu_det','pcb_ind']);a=p.parse_args();d=a.dataset
run=ROOT/f'runs/evaluation/{d}_s11_bright';out=run/'analysis';meta=json.loads((run/'run_meta.json').read_text());cfg=meta['config']
base=json.loads((out/'clean_metrics.json').read_text());audit=json.loads((ROOT/f'inputs/{d}_source_audit.json').read_text())
s=pd.read_csv(out/'grid_summary.csv');metrics=pd.read_csv(out/'condition_metrics.csv');occ=pd.read_csv(out/'occupancy.csv')
assert len(s)==80 and len(metrics)==240
rows=[]
for name,g in s.groupby('condition_id',sort=False):
 m=metrics[metrics.condition_id==name];worst=g.loc[g.sensitivity_pp.idxmax()]
 rows.append({'dataset':d,'condition':name,'clean_recall':base['recall'],'clean_map50':base['coco_map50'],
  'mean_recall_drop_pp':g.sensitivity_pp.mean(),'min_cell_drop_pp':g.sensitivity_pp.min(),'max_cell_drop_pp':g.sensitivity_pp.max(),
  'max_cell':worst.grid_label,'max_cell_ci_low_pp':100*worst.pointwise_ci_low,'max_cell_ci_high_pp':100*worst.pointwise_ci_high,
  'mean_map50_drop_pp':100*(base['coco_map50']-m.coco_map50.mean()),'mean_map50_95_drop_pp':100*(base['coco_map50_95']-m.coco_map50_95.mean()),
  'actual_changed_pixels':g.mean_actual_changed_pixels.mean()})
summary=pd.DataFrame(rows);summary.to_csv(out/'condition_summary.csv',index=False)
lim=max(s.sensitivity_pp.abs().max(),.01)
fig,axes=plt.subplots(2,3,figsize=(14,9),layout='constrained')
for ax,(name,g) in zip(axes.flat,s.groupby('condition_id',sort=False)):
 ar=g.sort_values('cell').sensitivity_pp.to_numpy().reshape(4,4);im=ax.imshow(ar,cmap='RdBu_r',vmin=-lim,vmax=lim)
 for y in range(4):
  for x in range(4):ax.text(x,y,f'G{y*4+x+1}\n{ar[y,x]:.2f}',ha='center',va='center',fontsize=8,color='white' if abs(ar[y,x])>lim*.55 else 'black')
 ax.set_title(name)
axes.flat[-1].axis('off');fig.colorbar(im,ax=list(axes.flat),label='Recall drop (percentage points)',shrink=.8)
fig.suptitle(d+' | YOLOv8n seed 11 | 3 pattern repetitions');fig.savefig(out/'bright_sensitivity_maps.png',dpi=170);plt.close(fig)
fig,ax=plt.subplots(figsize=(6,5));im=ax.imshow(occ.box_area_occupancy.to_numpy().reshape(4,4),cmap='YlOrRd');fig.colorbar(im,ax=ax,label='GT box area occupancy');ax.set_title(d+' | GT box occupancy');fig.tight_layout();fig.savefig(out/'occupancy_map.png',dpi=170);plt.close(fig)
manifest=load_jsonl(ROOT/f'data/canonical/{d}/manifest.jsonl');test=[r for r in manifest if r['split']=='test'];areas=np.array([r['height']*r['width'] for r in test]);dimensions=sorted({(r['width'],r['height']) for r in test})
train=pd.read_csv(ROOT/f'runs/train/{d}_yolov8n_s11/results.csv');worst=s.loc[s.sensitivity_pp.idxmax()]
markers=list((run/'predictions').glob('*.done.json'));assert len(markers)==241
table='| 조건 | 평균 Recall 감소 (%p) | 위치별 최소~최대 (%p) | 최대 위치 | 평균 mAP50 감소 (%p) |\n|---|---:|---:|---|---:|\n'
for r in rows:table+=f"| {r['condition']} | {r['mean_recall_drop_pp']:.4f} | {r['min_cell_drop_pp']:.4f} ~ {r['max_cell_drop_pp']:.4f} | {r['max_cell']} | {r['mean_map50_drop_pp']:.4f} |\n"
ci_includes_zero=worst.pointwise_ci_low<=0<=worst.pointwise_ci_high
board_note=('PCB 숫자 접두어 683개 중 124개가 여러 분할에 걸치며, 학습–시험 겹침은 63개다. `5810_t`와 `5810_b`는 다른 면 접두어로 묶인다. 공식 분할에서 면 접두어는 분리되지만 숫자가 같은 실제 보드의 양면일 가능성이 있어, 보드 단위 독립 시험으로 해석하지 않는다. Bootstrap도 면 접두어 단위여서 같은 보드 양면의 상관을 완전히 반영하지 못할 수 있다.' if d=='pcb_ind' else '')
text=f'''# {d} 밝은 화소 손상 위치별 민감도 실험

GC10과 동일한 YOLOv8n 학습 설정과 밝은 화소 손상 조건을 적용했다. 데이터셋마다 동일한 원본 COCO 사전학습 체크포인트에서 독립 학습했다. 단일 학습 시드(11)의 실험이다.

## 데이터 준비

원본 {audit['n_source']}장 중 {audit['n_retained']}장 사용. 분할: {audit['split_counts']}. 데이터 출처와 압축 파일 해시는 `data/raw/{d}/provenance.json`에, 중복·주석·분할 기록은 `inputs/{d}_source_audit.json`에 보존했다.

분할 규칙: {audit['split_policy']}.
좌표 규칙: {audit['coordinate_convention']}.
Difficult 처리: {audit['difficult_policy']}. 원본 difficult 객체 수: {audit['difficult_source_objects']}.
정확히 동일한 영상 중 주석이 충돌하는 그룹 {len(audit['conflicting_exact_duplicate_groups_excluded'])}개를 전체 제외했고, 같은 내용의 중복 영상 {len(audit['exact_duplicates_excluded'])}장과 중복 상자 {len(audit['duplicate_annotations_removed'])}개를 정리했다. 빈 라벨 제외 수: {len(audit['empty_labels_excluded'])}.

{audit['grouping_warning']}. {board_note}

근접 중복은 수동 검수하지 않았다. 공식 분할이 있는 PCB-IND는 원래 분할을 유지했고, NEU-DET는 이미지 클래스별 70/15/15(seed=101)로 분할했다.

## 고정 실험 조건

최대 100 epoch, patience=20, AdamW lr=0.001, batch=16, imgsz=640, seed=11. 실제 학습 {len(train)} epoch; 검증 최적 best.pt를 고정했다. 합성 화소 손상 없이 학습했고 공간·색상 증강은 GC10과 동일하게 0이다.

`configs/bright_frozen.yaml`의 4×4 격자 × 5개 조건 × 3개 패턴 반복을 사용했다. 고립 밝은 화소 1/9/25개와 밝은 3×3 군집은 RGB를 255로 치환하고, 부분 강도 조건은 원본과 255를 alpha=0.5로 혼합한다. 같은 해상도·반복에서 동일 패턴을 위치만 옮기며 원본 제품 결함 주석은 바꾸지 않는다. 센서 RAW의 물리적 hot pixel 모델이 아닌 영상 공간의 밝은 화소 이상이다.

원본 해상도에서 주입 후 YOLO 전처리한다. 운영 confidence=0.25, class-aware matching IoU=0.5. 예측 보존 floor=0.001, NMS IoU=0.7, max_det=300; AP는 같은 공식 COCO evaluator(maxDets=100)로 계산한다.

시험 영상 해상도 종류 수: {len(dimensions)}. 예: {dimensions[:10]}. 원본 25개 손상 비율은 영상별 평균 {np.mean(25/areas)*100:.6f}%, 최소 {np.min(25/areas)*100:.6f}%, 최대 {np.max(25/areas)*100:.6f}%다. 데이터셋마다 원본 해상도와 resize 비율이 달라 동일 화소 수가 동일 손상 비율/모델 입력 크기를 뜻하지 않는다.

## 원본 시험 성능

| 지표 | 값 |
|---|---:|
| 시험 이미지 / GT 결함 | {base['n_images']} / {base['n_gt']} |
| TP / FP / FN | {base['tp']} / {base['fp']} / {base['fn']} |
| Recall | {100*base['recall']:.3f}% |
| Precision | {100*base['precision']:.3f}% |
| COCO mAP50 | {100*base['coco_map50']:.3f}% |
| COCO mAP50:95 | {100*base['coco_map50_95']:.3f}% |

## 위치별 변화

{table}

![밝은 화소 민감도](bright_sensitivity_maps.png)
![GT 분포](occupancy_map.png)

관찰된 최대 위치별 패턴 평균 Recall 감소는 `{worst.condition_id}`의 {worst.grid_label}에서 {worst.sensitivity_pp:.4f}%p였다. 해당 위치의 pointwise 95% 쌍체 bootstrap 구간은 [{100*worst.pointwise_ci_low:.4f}, {100*worst.pointwise_ci_high:.4f}]%p로, 0을 {'포함한다' if ci_includes_zero else '포함하지 않는다'}. 이는 사후에 고른 최대 위치에 대한 보정 구간이 아니며 위치 간 차이의 정식 검정도 아니다. 그림의 양수는 감소, 음수는 개선이다.

구간은 이미지 그룹과 패턴 반복을 재표집한 2,000회 bootstrap이며 학습 시드 불확실성이나 지도 전체 다중 비교를 포함하지 않는다. GT 분포는 상자 점유율이므로 실제 결함 픽셀 마스크 또는 위치 효과 교란 통제를 의미하지 않는다.

## 실행 및 한계

241개 추론 묶음, {241*base['n_images']:,}회 이미지 평가. 80개 조건/위치 요약, 240개 조건/반복/위치 지표, {240*base['n_images']:,}행 쌍체 이미지 기록을 저장했다. 모든 입력·모델·설정·실행 코드 해시와 패키지 버전을 기록하고 완료 캐시의 SHA256을 검증했다.

실제 센서 RAW/ISP/DPC, 제품별 출하 미검출률, 복수 학습 시드의 재현성은 검증하지 않았다. 데이터셋 간 차이에는 모델 원본 성능, 해상도, 결함 분포와 대상 도메인이 함께 작용하므로 이를 센서 위치의 보편적 특성으로 단정하지 않는다.
'''
(out/'REPORT_KO.md').write_text(text)
dump_json(out/'report_meta.json',{'source_sha256':digest_file(__file__),'run_hash':meta['run_hash'],'n_test':len(test),'original_area_range':[int(areas.min()),int(areas.max())],'mean_25_pixel_fraction':float(np.mean(25/areas))})
print('Report saved',out/'REPORT_KO.md',flush=True)
