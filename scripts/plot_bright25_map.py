"""Focused, common-scale spatial sensitivity figure from existing experiment CSVs."""
from pathlib import Path
import json,sys
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle
ROOT=Path(__file__).resolve().parents[1]
font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams['axes.unicode_minus']=False
sets=[('GC10-DET','gc10_s11_frozen','2048 × 1000'),('NEU-DET','neu_det_s11_bright','200 × 200'),('PCB-IND','pcb_ind_s11_bright','300 × 300')]
out=ROOT/'reports/bright_replication_2026-11-05';data=[];csvs=[]
for title,name,res in sets:
 root=ROOT/'runs/evaluation'/name/'analysis';df=pd.read_csv(root/'grid_summary.csv');df=df[df.condition_id=='bright_k25'].sort_values('cell')
 if df.cell.tolist()!=list(range(16)):raise ValueError('Expected every 4x4 cell exactly once')
 base=json.loads((root/'clean_metrics.json').read_text());ar=df.sensitivity_pp.to_numpy().reshape(4,4)
 data.append((title,res,base,df,ar));part=df.copy();part.insert(0,'dataset',title);csvs.append(part)
lim=float(np.ceil(max(abs(x[4]).max() for x in data)*2)/2)
fig,axes=plt.subplots(1,3,figsize=(16,7.2));fig.subplots_adjust(left=.045,right=.9,top=.78,bottom=.25,wspace=.16)
fig.suptitle('밝은 화소 25개 주입에 따른 위치별 Recall 감소',fontproperties=font,fontsize=23,y=.97)
fig.text(.47,.89,'4 × 4 격자 · 동일 패턴을 위치만 이동 · 패턴 3회 평균 · YOLOv8n',fontproperties=font,fontsize=12,ha='center')
for ax,(title,res,base,df,ar) in zip(axes,data):
 im=ax.imshow(ar,cmap='RdBu_r',vmin=-lim,vmax=lim,interpolation='nearest')
 ax.set_title(title+'\n'+res+' | 원본 Recall '+f"{100*base['recall']:.2f}%",fontproperties=font,fontsize=14,pad=12)
 for y in range(4):
  for x in range(4):
   v=ar[y,x];v=0. if abs(v)<.0005 else v
   ax.text(x,y,f'G{y*4+x+1}\n{v:+.3f}',ha='center',va='center',fontsize=12,color='white' if abs(v)>lim*.55 else '#17212b')
 ax.set_xticks([]);ax.set_yticks([])
 ax.set_xticks(np.arange(-.5,4,1),minor=True);ax.set_yticks(np.arange(-.5,4,1),minor=True)
 ax.grid(which='minor',color='#cfd5db',linewidth=.8);ax.tick_params(which='minor',bottom=False,left=False)
 cell=int(df.loc[df.sensitivity_pp.idxmax(),'cell']);y,x=divmod(cell,4)
 ax.add_patch(Rectangle((x-.48,y-.48),.96,.96,fill=False,edgecolor='#18232d',linewidth=2))
 ax.text(.5,-.09,f'최대 평균 감소: G{cell+1} · {ar[y,x]:.3f}%p',transform=ax.transAxes,fontproperties=font,fontsize=12,ha='center')
cbax=fig.add_axes([.93,.29,.015,.45]);cb=fig.colorbar(im,cax=cbax)
cb.set_label('Recall 감소 (%p)',fontproperties=font,fontsize=12,labelpad=10)
fig.text(.5,.14,'빨강: 검출 성능 감소   |   파랑: 검출 성능 증가   |   세 지도는 동일한 색상 범위 사용',fontproperties=font,fontsize=12,ha='center')
fig.text(.5,.085,'각 칸의 값은 그 위치에 손상을 넣었을 때의 원본 대비 변화입니다. 실제 센서의 절대 좌표를 뜻하지 않습니다.',fontproperties=font,fontsize=10,ha='center',color='#505b66')
fig.text(.5,.045,'같은 25화소라도 데이터셋별 해상도와 resize 비율이 달라 손상 면적 비율은 다릅니다. 최대 칸은 사후 기술통계입니다.',fontproperties=font,fontsize=10,ha='center',color='#505b66')
fig.savefig(out/'bright25_recall_sensitivity_map_ko.png',dpi=220,facecolor='white')
fig.savefig(out/'bright25_recall_sensitivity_map_ko.pdf',facecolor='white')
plt.close(fig);pd.concat(csvs,ignore_index=True).to_csv(out/'bright25_recall_sensitivity_map.csv',index=False)
print(out/'bright25_recall_sensitivity_map_ko.png')
