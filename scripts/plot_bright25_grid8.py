"""Plot actual 8x8 results and distinguish them from the earlier 4x4 experiments."""
from pathlib import Path
import json,sys
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from core import digest_file,dump_json
font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc');plt.rcParams['axes.unicode_minus']=False
DATASETS=[('gc10_det','GC10-DET','gc10_s11_frozen'),('neu_det','NEU-DET','neu_det_s11_bright'),('pcb_ind','PCB-IND','pcb_ind_s11_bright')]
out=ROOT/'reports/bright25_grid8_2026-11-10';out.mkdir(exist_ok=True,parents=True)
new=[];old=[];rows=[];all_cells=[]
for d,title,prior in DATASETS:
 root=ROOT/f'runs/evaluation/{d}_s11_bright25_g8';meta=json.loads((root/'run_meta.json').read_text());a=root/'analysis'
 df=pd.read_csv(a/'grid_summary.csv').sort_values('cell');metrics=pd.read_csv(a/'condition_metrics.csv');base=json.loads((a/'clean_metrics.json').read_text())
 prev=ROOT/f'runs/evaluation/{prior}/analysis';old_df=pd.read_csv(prev/'grid_summary.csv');old_df=old_df[old_df.condition_id=='bright_k25'].sort_values('cell');old_base=json.loads((prev/'clean_metrics.json').read_text())
 if meta['config']['grid']!={'rows':8,'cols':8} or df.cell.tolist()!=list(range(64)) or len(metrics)!=192:raise ValueError('Incomplete 8x8 results')
 if len(list((root/'predictions').glob('*.done.json')))!=193:raise ValueError('Missing inference shards')
 if any(base[k]!=old_base[k] for k in ['n_images','n_gt','tp','fp','fn']):raise ValueError('Clean operational baseline differs from 4x4')
 new.append((title,base,df));old.append((title,old_base,old_df))
 maximum=df.loc[df.sensitivity_pp.idxmax()];before=old_df.loc[old_df.sensitivity_pp.idxmax()]
 row={'dataset':title,'n_test':base['n_images'],'n_gt':base['n_gt'],'clean_recall_pct':100*base['recall'],
  'grid8_mean_drop_pp':df.sensitivity_pp.mean(),'grid8_max_drop_pp':maximum.sensitivity_pp,'grid8_max_cell':maximum.grid_label,
  'grid8_max_row':int(maximum.cell)//8+1,'grid8_max_col':int(maximum.cell)%8+1,
  'grid8_max_ci_low_pp':100*maximum.pointwise_ci_low,'grid8_max_ci_high_pp':100*maximum.pointwise_ci_high,
  'grid8_mean_fault_recall_pct':100*metrics.recall.mean(),'grid8_mean_map50_drop_pp':100*(base['coco_map50']-metrics.coco_map50.mean()),
  'grid4_mean_drop_pp':old_df.sensitivity_pp.mean(),'grid4_max_drop_pp':before.sensitivity_pp,'grid4_max_cell':before.grid_label,
  'mean_actual_changed_pixels':df.mean_actual_changed_pixels.mean()}
 rows.append(row);cells=df.copy();cells.insert(0,'dataset',title);cells['row']=cells.cell//8+1;cells['col']=cells.cell%8+1;cells['fault_recall_pct']=100*cells.recall_clean-cells.sensitivity_pp;all_cells.append(cells)
pd.DataFrame(rows).to_csv(out/'summary_grid4_grid8.csv',index=False);pd.concat(all_cells,ignore_index=True).to_csv(out/'grid8_cells.csv',index=False)

def draw(ax,title,base,df,size,lim):
 ar=df.sensitivity_pp.to_numpy().reshape(size,size);im=ax.imshow(ar,cmap='RdBu_r',vmin=-lim,vmax=lim,interpolation='nearest')
 ax.set_title(f"{title} | {size} × {size}\n원본 Recall {100*base['recall']:.2f}%",fontproperties=font,fontsize=14,pad=12)
 for y in range(size):
  for x in range(size):
   v=ar[y,x];v=0. if abs(v)<.0005 else v
   ax.text(x,y,f'G{y*size+x+1}\n{v:+.2f}',ha='center',va='center',fontsize=8 if size==8 else 10,color='white' if abs(v)>lim*.55 else '#17212b')
 ax.set_xticks([]);ax.set_yticks([]);ax.set_xticks(np.arange(-.5,size,1),minor=True);ax.set_yticks(np.arange(-.5,size,1),minor=True);ax.grid(which='minor',color='#d0d6dc',linewidth=.5);ax.tick_params(which='minor',bottom=False,left=False)
 cell=int(df.loc[df.sensitivity_pp.idxmax(),'cell']);y,x=divmod(cell,size);ax.add_patch(Rectangle((x-.48,y-.48),.96,.96,fill=False,edgecolor='#17212b',linewidth=2))
 ax.text(.5,-.06,f'최대 평균 감소: G{cell+1} ({y+1}행 {x+1}열) · {ar[y,x]:.3f}%p',transform=ax.transAxes,fontproperties=font,fontsize=11,ha='center')
 return im
lim=max(.01,float(np.ceil(max(abs(g[2].sensitivity_pp).max() for g in new)*2)/2))
fig,axes=plt.subplots(1,3,figsize=(21,10));fig.subplots_adjust(left=.035,right=.91,top=.81,bottom=.23,wspace=.13)
fig.suptitle('8 × 8 위치별 민감도 지도 — 밝은 화소 25개',fontproperties=font,fontsize=25,y=.96)
fig.text(.48,.9,'기존 학습 모델·시험 자료 유지 | 64개 위치 × 패턴 3회 실제 재평가',fontproperties=font,fontsize=14,ha='center')
for ax,(title,base,df) in zip(axes,new):im=draw(ax,title,base,df,8,lim)
cax=fig.add_axes([.94,.28,.012,.48]);cb=fig.colorbar(im,cax=cax);cb.set_label('Recall 감소 (%p)',fontproperties=font,fontsize=12)
fig.text(.5,.145,'빨강: Recall 감소 | 파랑: Recall 증가 | 세 지도는 동일한 색상 범위 | 검은 테두리: 관찰된 최대 칸',fontproperties=font,fontsize=13,ha='center')
fig.text(.5,.09,'각 칸은 해당 영역에 25개 화소를 주입했을 때의 변화입니다. 최대값은 3개 패턴 평균 중 사후 선택한 기술통계입니다.',fontproperties=font,fontsize=11,ha='center',color='#53606c')
fig.text(.5,.05,'8 × 8에서는 주입 영역이 약 1/4로 줄고 패턴도 새로 생성됩니다. 4 × 4 지도를 확대하거나 보간한 결과가 아닙니다.',fontproperties=font,fontsize=11,ha='center',color='#53606c')
fig.savefig(out/'bright25_grid8_recall_map_ko.png',dpi=240,facecolor='white');fig.savefig(out/'bright25_grid8_recall_map_ko.pdf',facecolor='white');plt.close(fig)
lim=max(.01,float(np.ceil(max(abs(g[2].sensitivity_pp).max() for g in new+old)*2)/2))
fig,axes=plt.subplots(2,3,figsize=(20,15));fig.subplots_adjust(left=.04,right=.91,top=.9,bottom=.12,wspace=.13,hspace=.28)
for i,group in enumerate([old,new]):
 for ax,(title,base,df) in zip(axes[i],group):im=draw(ax,title,base,df,4 if i==0 else 8,lim)
fig.suptitle('4 × 4 및 8 × 8 재실험 비교 — 밝은 화소 25개',fontproperties=font,fontsize=24,y=.97)
cax=fig.add_axes([.94,.25,.012,.5]);cb=fig.colorbar(im,cax=cax);cb.set_label('Recall 감소 (%p), 공통 범위',fontproperties=font)
fig.text(.5,.045,'격자별 주입 영역·패턴과 최대값 선택 후보 수가 다릅니다. 최대 감소량의 차이를 해상도 효과 하나로 해석하지 않습니다.',fontproperties=font,fontsize=12,ha='center')
fig.savefig(out/'grid4_grid8_comparison_ko.png',dpi=200,facecolor='white');plt.close(fig)
table='| 데이터셋 | 원본 Recall | 8×8 평균 감소 | 8×8 최대 감소 | 최대 위치 | 최대 위치 95% pointwise CI |\n|---|---:|---:|---:|---|---|\n'
for r in rows:table+=f"| {r['dataset']} | {r['clean_recall_pct']:.3f}% | {r['grid8_mean_drop_pp']:.3f}%p | {r['grid8_max_drop_pp']:.3f}%p | {r['grid8_max_cell']} ({r['grid8_max_row']}행 {r['grid8_max_col']}열) | [{r['grid8_max_ci_low_pp']:.3f}, {r['grid8_max_ci_high_pp']:.3f}]%p |\n"
table2='| 데이터셋 | 4×4 평균 감소 | 8×8 평균 감소 | 4×4 최대 감소 | 8×8 최대 감소 |\n|---|---:|---:|---:|---:|\n'
for r in rows:table2+=f"| {r['dataset']} | {r['grid4_mean_drop_pp']:.3f}%p | {r['grid8_mean_drop_pp']:.3f}%p | {r['grid4_max_drop_pp']:.3f}%p | {r['grid8_max_drop_pp']:.3f}%p |\n"
report=f'''# 밝은 화소 25개 — 8×8 위치별 민감도 재실험

기존 GC10-DET, NEU-DET, PCB-IND의 학습된 YOLOv8n, 시험 데이터, confidence·IoU·전처리와 패턴 반복 수(3)를 유지하고 8×8 격자로 다시 추론했다. 재학습하거나 기존 4×4 지도를 보간하지 않았다.

## 조건과 실행

RGB 255 고립 화소 25개를 각 격자 안에 주입했다. 64개 위치 × 패턴 3회 = 손상 조건 192개, 원본 포함 데이터셋별 193개 추론 묶음이다. 사용한 시험 영상은 GC10 354장, NEU 270장, PCB 469장으로 총 210,949회 이미지 평가를 실행했다.

`configs/bright25_grid8.yaml`과 `protocol.json`에 설정, 기존 데이터·모델 해시, 실행 코드 해시를 기록했다. 원본 평가의 TP/FP/FN은 이전 4×4 실험과 같음을 확인했다. 민감도는 원본 Recall − 손상 Recall이며 양수는 성능 감소, 음수는 증가다.

## 8×8 결과

{table}

![8×8 민감도 지도](bright25_grid8_recall_map_ko.png)

칸 번호는 왼쪽 위 G1부터 오른쪽 아래 G64까지 행 우선이다. 좌표는 영상 내 상대 위치이며 실제 센서의 절대 위치를 뜻하지 않는다.

## 4×4와 비교

{table2}

![두 격자 비교](grid4_grid8_comparison_ko.png)

**두 격자는 주입 영역과 패턴이 다르다.** 같은 25개를 넣지만 8×8 한 칸의 면적은 4×4의 약 1/4이므로 손상이 더 좁은 영역에 모인다. 패턴 생성 시 격자 크기가 난수 seed에 포함되어 있어 새 패턴을 사용한다. 전체 영상 대비 명령 화소 수는 동일하지만 격자 내부 손상 밀도와 정답 상자와의 겹침은 달라질 수 있다. 64칸 중 최대값과 16칸 중 최대값은 선택 후보 수도 다르므로 최대값 변화만으로 위치 효과가 더 강해졌다고 확정하지 않는다.

## 통계와 범위

신뢰구간은 이미지 그룹과 패턴을 재표집한 2,000회 pointwise 쌍체 bootstrap이다. 지도 전체의 동시 구간, 다중 비교 보정, 학습 시드 불확실성 또는 위치 간 차이의 정식 검정이 아니다. 단일 학습 시드와 기존 분할에 조건부인 결과다.

PCB의 공식 분할에는 같은 숫자 ID의 t/b 면이 학습·시험에 걸칠 수 있어 독립 보드 일반화를 주장하지 않는다. NEU도 실제 제품/코일 ID가 확인되지 않았다. 원본 해상도와 resize 배율이 데이터셋마다 다르므로 같은 25개를 동일한 모델 입력 손상 면적으로 해석하지 않는다. 실제 RAW/ISP/DPC나 센서 수명은 평가하지 않았다.

## 파일

- [8×8 위치별 수치 CSV](grid8_cells.csv)
- [4×4·8×8 요약 CSV](summary_grid4_grid8.csv)
- [8×8 지도 PDF](bright25_grid8_recall_map_ko.pdf)
'''
(out/'REPORT_KO.md').write_text(report);dump_json(out/'plot_meta.json',{'source_sha256':digest_file(__file__),'inputs':{d:digest_file(ROOT/f'runs/evaluation/{d}_s11_bright25_g8/analysis/grid_summary.csv') for d,_,_ in DATASETS}})
print(pd.DataFrame(rows).to_string(index=False));print(out/'bright25_grid8_recall_map_ko.png')
