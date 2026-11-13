"""Visualize recorded injection sites, not a reconstructed/interpolated pattern."""
from pathlib import Path
import csv,json,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
ROOT=Path(__file__).resolve().parents[1]
FONT=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
DATASETS=[('gc10_det','GC10-DET'),('neu_det','NEU-DET'),('pcb_ind','PCB-IND')]
OUT=ROOT/'reports/bright25_grid8_2026-11-10/fault_sites';OUT.mkdir(parents=True,exist_ok=True)
records={};csvrows=[]
for d,title in DATASETS:
 for rep in range(3):
  cells=[];reference=None
  for cell in range(64):
   p=ROOT/f'runs/evaluation/{d}_s11_bright25_g8/predictions/bright_k25__r{rep:03d}__g{cell:03d}.jsonl'
   with p.open() as f:r=json.loads(f.readline())
   fault=r['fault'];xy=np.asarray(fault['coordinates_xy']);x0,y0,x1,y1=fault['bounds_xyxy']
   if xy.shape!=(25,2) or len(np.unique(xy,axis=0))!=25:raise ValueError('Unexpected site count')
   if not ((xy[:,0]>=x0)&(xy[:,0]<x1)&(xy[:,1]>=y0)&(xy[:,1]<y1)).all():raise ValueError('Site outside grid cell')
   offset=xy-[x0,y0]
   if reference is None:reference=offset
   if not np.array_equal(offset,reference):raise ValueError('Expected translation of same pattern')
   cells.append(r)
   for i,(x,y) in enumerate(xy):csvrows.append({'dataset':title,'repeat':rep,'cell':cell,'grid_label':f'G{cell+1}','site':i+1,'x':int(x),'y':int(y),'local_x':int(x-x0),'local_y':int(y-y0),'width':r['width'],'height':r['height'],'target_rgb':'255,255,255','count_commanded':fault['count_commanded'],'count_changed_in_example':fault['count_changed'],'example_image_id':r['image_id']})
  if len({r['image_id'] for r in cells})!=1:raise ValueError('Example image changed')
  records[d,rep]=cells
with (OUT/'recorded_fault_coordinates.csv').open('w',newline='',encoding='utf-8-sig') as f:
 wr=csv.DictWriter(f,fieldnames=list(csvrows[0]));wr.writeheader();wr.writerows(csvrows)

for rep in range(3):
 fig,axes=plt.subplots(1,3,figsize=(21,9.5));fig.subplots_adjust(left=.035,right=.98,top=.79,bottom=.19,wspace=.12)
 fig.suptitle(f'8 × 8 각 칸의 핫픽셀 주입 위치 — 패턴 {rep+1}',fontproperties=FONT,fontsize=25,y=.96)
 fig.text(.5,.89,'각 칸 25개 · 저장된 주입 대상 좌표 · 영상 좌표 정규화 · 빨간 점은 확대 표시',fontproperties=FONT,fontsize=14,ha='center')
 for ax,(d,title) in zip(axes,DATASETS):
  cells=records[d,rep];h,w=cells[0]['height'],cells[0]['width']
  for cell,r in enumerate(cells):
   xy=np.array(r['fault']['coordinates_xy']);x0,y0,x1,y1=r['fault']['bounds_xyxy']
   # Normalize axes so grid locations can be compared; coordinates remain original pixels in CSV.
   ax.scatter((xy[:,0]+.5)/w,(xy[:,1]+.5)/h,s=7,color='#d92332',edgecolors='none',zorder=4)
   ax.text((x0+1)/w,(y0+1)/h,f'G{cell+1}',ha='left',va='top',fontsize=7,color='#17212b',zorder=3,bbox={'facecolor':'white','edgecolor':'none','alpha':.8,'pad':.6})
  for i in range(9):
   ax.axvline((i*w//8)/w,color='#87929d',linewidth=.6);ax.axhline((i*h//8)/h,color='#87929d',linewidth=.6)
  ax.set_xlim(0,1);ax.set_ylim(1,0);ax.set_aspect('equal');ax.set_xticks([]);ax.set_yticks([]);ax.set_facecolor('#fafbfd')
  ax.set_title(f'{title}\n원본 {w} × {h} | 한 칸 {w//8}~{(w+7)//8} × {h//8}~{(h+7)//8} 픽셀',fontproperties=FONT,fontsize=14,pad=13)
 fig.text(.5,.12,'전체 64칸의 위치를 모아 보여주는 배치도입니다. 실제 추론 한 번에는 한 칸에만 25개를 주입했습니다.',fontproperties=FONT,fontsize=14,ha='center',color='#17212b')
 fig.text(.5,.065,'같은 반복 안에서는 한 칸의 패턴을 그대로 평행 이동합니다. 반복 1·2·3 사이에서만 패턴이 달라집니다.',fontproperties=FONT,fontsize=12,ha='center',color='#53606c')
 fig.savefig(OUT/f'grid8_recorded_fault_sites_pattern{rep+1}.png',dpi=240,facecolor='white');fig.savefig(OUT/f'grid8_recorded_fault_sites_pattern{rep+1}.pdf',facecolor='white');plt.close(fig)
# Enlarge a single cell with unit-pixel geometry; all points are the same first-repeat offsets above.
fig,axes=plt.subplots(1,3,figsize=(18,7.5));fig.subplots_adjust(top=.76,bottom=.17,wspace=.23)
fig.suptitle('한 칸 내부의 실제 25개 화소 배치 — 첫 번째 반복, G1',fontproperties=FONT,fontsize=22,y=.97)
for ax,(d,title) in zip(axes,DATASETS):
 r=records[d,0][0];fault=r['fault'];x0,y0,x1,y1=fault['bounds_xyxy'];xy=np.array(fault['coordinates_xy'])-[x0,y0];cw,ch=x1-x0,y1-y0
 mask=np.zeros((ch,cw));mask[xy[:,1],xy[:,0]]=1
 ax.imshow(mask,cmap='gray',vmin=0,vmax=1,interpolation='nearest',extent=(-.5,cw-.5,ch-.5,-.5));ax.set_title(f'{title} | {cw} × {ch} 픽셀\n주입 대상 25개',fontproperties=FONT,fontsize=14)
 ax.set_xlabel('칸 내부 x 좌표',fontproperties=FONT);ax.set_ylabel('칸 내부 y 좌표',fontproperties=FONT)
fig.text(.5,.07,'흰색 한 점이 원본 영상의 한 화소입니다. 위치 마스크이며 원본 영상의 밝기를 나타내지 않습니다.',fontproperties=FONT,fontsize=12,ha='center')
fig.savefig(OUT/'single_cell_native_pixel_masks.png',dpi=240,facecolor='white');plt.close(fig)
print('Verified 3 datasets × 3 patterns × 64 cells × 25 unique in-cell sites:',len(csvrows),'coordinates')
print(OUT)
