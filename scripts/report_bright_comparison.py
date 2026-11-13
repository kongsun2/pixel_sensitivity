"""Compare identical bright conditions across the three recorded datasets."""
from pathlib import Path
import json,sys
import pandas as pd,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from core import load_jsonl,dump_json,digest_file
DATASETS=[('gc10_det','GC10-DET','gc10_s11_frozen'),('neu_det','NEU-DET','neu_det_s11_bright'),('pcb_ind','PCB-IND','pcb_ind_s11_bright')]
conditions=['bright_k1','bright_k9','bright_k25','bright_cluster9','bright_partial9'];results=[];baseline=[];grids={}
out=ROOT/'reports/bright_replication_2026-11-05';out.mkdir(parents=True,exist_ok=True)
for key,title,name in DATASETS:
 run=ROOT/'runs/evaluation'/name;meta=json.loads((run/'run_meta.json').read_text());cfg=meta['config']
 expected=json.loads((ROOT/'configs/bright_protocol.json').read_text())
 import yaml
 frozen=yaml.safe_load((ROOT/'configs/bright_frozen.yaml').read_text())
 assert {k:v for k,v in cfg.items() if k!='conditions'}=={k:v for k,v in frozen.items() if k!='conditions'}
 assert [c for c in cfg['conditions'] if c['id'] in conditions]==frozen['conditions']
 s=pd.read_csv(run/'analysis/grid_summary.csv');s=s[s.condition_id.isin(conditions)].copy();assert len(s)==80;grids[key]=s
 b=json.loads((run/'analysis/clean_metrics.json').read_text());m=pd.read_csv(run/'analysis/condition_metrics.csv')
 manifest=load_jsonl(ROOT/f'data/canonical/{key}/manifest.jsonl');test=[r for r in manifest if r['split']=='test'];areas=np.array([r['height']*r['width'] for r in test]);scales=np.array([min(640/r['height'],640/r['width']) for r in test])
 baseline.append({'dataset':title,'n_test':b['n_images'],'n_gt':b['n_gt'],'recall_pct':100*b['recall'],'precision_pct':100*b['precision'],'coco_map50_pct':100*b['coco_map50'],'coco_map50_95_pct':100*b['coco_map50_95'],
  'mean_25_fault_percent_original':float(np.mean(25/areas)*100),'median_resize_scale':float(np.median(scales)),
  'min_width':min(r['width'] for r in test),'max_width':max(r['width'] for r in test),'min_height':min(r['height'] for r in test),'max_height':max(r['height'] for r in test)})
 for c in conditions:
  g=s[s.condition_id==c];worst=g.loc[g.sensitivity_pp.idxmax()];cm=m[m.condition_id==c]
  results.append({'dataset':title,'condition':c,'mean_recall_drop_pp':g.sensitivity_pp.mean(),'min_cell_drop_pp':g.sensitivity_pp.min(),'max_cell_drop_pp':g.sensitivity_pp.max(),'max_cell':worst.grid_label,
   'max_cell_pointwise_ci_low_pp':100*worst.pointwise_ci_low,'max_cell_pointwise_ci_high_pp':100*worst.pointwise_ci_high,
   'mean_map50_drop_pp':100*(b['coco_map50']-cm.coco_map50.mean()),'mean_map50_95_drop_pp':100*(b['coco_map50_95']-cm.coco_map50_95.mean())})
pd.DataFrame(results).to_csv(out/'bright_comparison.csv',index=False);pd.DataFrame(baseline).to_csv(out/'baselines_and_dose.csv',index=False)
lim=max([float(g.sensitivity_pp.abs().max()) for g in grids.values()]+[.01]);fig,axes=plt.subplots(3,5,figsize=(21,13),layout='constrained')
for i,(key,title,_) in enumerate(DATASETS):
 for j,c in enumerate(conditions):
  ax=axes[i,j];g=grids[key];ar=g[g.condition_id==c].sort_values('cell').sensitivity_pp.to_numpy().reshape(4,4)
  im=ax.imshow(ar,cmap='RdBu_r',vmin=-lim,vmax=lim)
  for y in range(4):
   for x in range(4):ax.text(x,y,f'G{y*4+x+1}\n{ar[y,x]:.2f}',ha='center',va='center',fontsize=7,color='white' if abs(ar[y,x])>lim*.55 else 'black')
  ax.set_title(title+' | '+c);ax.set_xticks([]);ax.set_yticks([])
fig.colorbar(im,ax=list(axes.flat),label='Recall drop (percentage points), common scale',shrink=.8);fig.suptitle('Bright pixel sensitivity | same count/strength, different native resolution | YOLOv8n seed 11')
fig.savefig(out/'three_dataset_bright_maps.png',dpi=170);plt.close(fig)
table='| 데이터셋 | 시험 영상 / GT | Recall | Precision | mAP50 | mAP50:95 |\n|---|---:|---:|---:|---:|---:|\n'
for b in baseline:table+=f"| {b['dataset']} | {b['n_test']} / {b['n_gt']} | {b['recall_pct']:.2f}% | {b['precision_pct']:.2f}% | {b['coco_map50_pct']:.2f}% | {b['coco_map50_95_pct']:.2f}% |\n"
table2='| 데이터셋 | 조건 | 평균 Recall 감소 (%p) | 최대 위치 평균 감소 (%p) | 최대 위치 | 해당 위치 pointwise 95% CI (%p) |\n|---|---|---:|---:|---|---|\n'
for r in results:table2+=f"| {r['dataset']} | {r['condition']} | {r['mean_recall_drop_pp']:.4f} | {r['max_cell_drop_pp']:.4f} | {r['max_cell']} | [{r['max_cell_pointwise_ci_low_pp']:.4f}, {r['max_cell_pointwise_ci_high_pp']:.4f}] |\n"
dose='| 데이터셋 | 시험 영상 너비 범위 | 높이 범위 | 25화소의 평균 원본 면적 비율 | YOLO resize 배율 중앙값 |\n|---|---:|---:|---:|---:|\n'
for b in baseline:dose+=f"| {b['dataset']} | {b['min_width']}~{b['max_width']} | {b['min_height']}~{b['max_height']} | {b['mean_25_fault_percent_original']:.6f}% | {b['median_resize_scale']:.3f} |\n"
links='\n'.join(f'- [{title} 상세 결과](../../runs/evaluation/{name}/analysis/REPORT_KO.md)' for _,title,name in DATASETS)
report=f'''# 밝은 화소 위치 민감도: GC10-DET / NEU-DET / PCB-IND

사용자 요청에 따라 GC10에서 사용한 밝은 화소 조건을 나머지 두 데이터셋에 적용했다. 각 데이터셋은 동일한 원본 COCO 사전학습 YOLOv8n에서 독립 학습했다. GC10의 기존 결과는 변경하지 않고 동일한 밝은 조건만 추출했다.

## 공통 조건

학습 seed=11, imgsz=640, AdamW lr=0.001, batch=16, 최대 100 epoch, patience=20, 검증 best.pt 고정. 화소 손상 없이 학습하고 공간/색상 증강은 0이다.

원본 해상도에 RGB=255인 고립 화소 1/9/25개, 밝은 3×3 군집, alpha=0.5 부분 밝은 화소 9개를 주입했다. 4×4 grid, 3개 고정 패턴 반복, conf=0.25, matching IoU=0.5, 동일 COCO AP evaluator를 사용했다. 개수·값·배치 규칙은 동일하며 각 데이터셋/해상도에 맞춰 생성한 패턴을 위치만 이동한다.

## 원본 시험 성능

{table}

## 밝은 화소 조건별 결과

평균은 16개 위치와 3개 패턴 반복에 대한 평균이며, 최대 위치 값은 3개 패턴의 평균 중 최대다. 양수는 감소, 음수는 개선이다. 최대 위치는 사후 기술통계이며 16칸 동시 검정으로 확정한 취약 위치가 아니다.

{table2}

![세 데이터셋 밝은 화소 민감도 지도](three_dataset_bright_maps.png)

## 해상도와 손상 비율

{dose}

같은 25개라도 NEU-DET 200×200과 PCB-IND 300×300에서는 GC10보다 원본 면적 대비 손상 비율이 높으며 YOLO 입력으로 확대된다. GC10의 큰 원본 영상은 축소되어 단일 화소 효과가 약해질 수 있다. 이 실험은 동일 화소 수에 대한 비교이며 손상 비율 또는 모델 입력상의 손상 면적을 일치시킨 비교는 아니다. 데이터셋 간 차이를 제조 대상이나 센서 민감도 차이 하나로 설명할 수 없다.

## 데이터 및 해석 범위

PCB-IND는 공식 Zenodo v4와 원래 분할을 사용했고, 정밀한 COCO 주석에서 라벨을 변환했다. 미확인 빈 라벨 127장은 제외했다. NEU-DET는 원 배포처의 1,800장을 점검하여 서로 같은 픽셀인데 주석이 다른 2장을 제외하고, 중복 상자 3개를 정리했다. VOC 1-based inclusive 좌표를 변환하고 difficult 객체도 정답으로 포함했다. 데이터셋별 상세 audit에 모든 처리 내역을 기록했다.

PCB-IND는 면 접두어(t/b 포함) 그룹, NEU-DET는 이미지 ID를 bootstrap 그룹으로 사용했다. PCB 숫자 접두어 683개 중 124개는 여러 분할에 걸치며, 학습–시험 겹침이 63개다. 같은 숫자 ID가 실제 보드의 양면을 뜻한다면 보드 단위 독립성이 없으므로 공식 분할 기준의 조건부 결과로 해석한다. 면별 bootstrap은 보드 양면 간 상관을 완전히 반영하지 못할 수 있다. 실제 제품/코일의 독립성은 확인되지 않았다. 특히 데이터셋마다 주석 정책과 시험 난이도가 다르므로 원본 성능도 함께 보아야 한다.

신뢰구간은 이미지 그룹 및 패턴 반복을 재표집한 2,000회 pointwise 쌍체 bootstrap이다. 위치 간 차이에 대한 정식 검정, 지도 전체 다중 비교 보정, 학습 시드 불확실성을 포함하지 않는다. 단일 학습 시드의 비교이므로 재현성을 확정하려면 복수 시드가 필요하다. 실제 센서 RAW/ISP/DPC 또는 제품 출하 불량률은 평가하지 않았다.

## 파일

{links}

- [조건별 수치 CSV](bright_comparison.csv)
- [기준 성능·해상도·손상 비율 CSV](baselines_and_dose.csv)
- [실행 방법](../../README.md)
'''
(out/'COMPARISON_KO.md').write_text(report);dump_json(out/'comparison_meta.json',{'source_sha256':digest_file(__file__),'inputs':{name:digest_file(ROOT/f'runs/evaluation/{name}/analysis/grid_summary.csv') for _,_,name in DATASETS}})
print(report)
