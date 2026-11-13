"""Reformat our verified study using only the reference report's presentation style."""
from pathlib import Path
from copy import deepcopy
from datetime import date
from html import escape
import json,sys,hashlib
import numpy as np,pandas as pd
from PIL import Image
from docx import Document
from docx.shared import Inches,Pt,RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from core import digest_file,dump_json
BASE=ROOT/'reports/research_validation_2026-11-11';SUM=BASE/'summary';OUT=ROOT/'reports/progress_report_reorganized_2026-11-12';OUT.mkdir(parents=True,exist_ok=True)
SOURCE=Path('/home/hyochang/pilot_progress_report.docx');source_sha=digest_file(SOURCE)
DOCX=Path('/home/hyochang/pilot_progress_report_reorganized.docx')
cond=pd.read_csv(SUM/'condition_means.csv');baselines=pd.read_csv(SUM/'clean_baselines.csv');seed_results=pd.read_csv(SUM/'bright25_seed_results.csv');seed_summary=pd.read_csv(SUM/'bright25_seed_summary.csv');corr=pd.read_csv(SUM/'seed_spatial_correlations.csv')
old_dis=pd.read_csv(BASE/'distribution/dataset_summary.csv');new_dis=pd.read_csv(BASE/'corrected_pcb_distribution/dataset_summary.csv');dist=pd.concat([old_dis[old_dis.dataset.isin(['gc10_det','neu_det'])],new_dis],ignore_index=True)
spatial_sd=pd.read_csv(OUT/'descriptive_spatial_dispersion.csv');perclass=pd.read_csv(OUT/'clean_per_class_seed11.csv')
pos0=pd.read_csv(BASE/'positions/position_effect_summary.csv');pos1=pd.read_csv(BASE/'corrected_pcb_positions/position_effect_summary.csv')
pos=pd.concat([pos0[(pos0.dataset.isin(['gc10_det','neu_det']))&(pos0.grouping=='recorded_groups')],pos1[pos1.grouping=='recorded_groups']],ignore_index=True)
DATA=['gc10_det','neu_det','pcb_ind_board'];NAMES={'gc10_det':'GC10-DET','neu_det':'NEU-DET','pcb_ind_board':'PCB-IND 새 분할'}
LABELS={'bright_k25':'밝은25·분산','dark_k25':'어두운25·분산','stuck_k25':'128고정25·분산','bright_k1':'밝은1·분산','bright_k9':'밝은9·분산','bright_partial25':'부분강도25·분산','bright_cluster25':'밝은25·5×5군집'}
CS=['bright_k1','bright_k9','bright_k25','dark_k25','stuck_k25','bright_partial25','bright_cluster25']
assert len(cond)==27 and len(seed_results)==9 and len(seed_summary)==3
assert json.loads((BASE/'COMPLETE.json').read_text())['complete']
ref=Document(SOURCE);doc=Document()
# Match the reference's restrained A4/Korean-body presentation; no source paragraphs/tables are copied.
sec=doc.sections[0];sec.page_width=Inches(8.2677);sec.page_height=Inches(11.6929);sec.top_margin=sec.bottom_margin=sec.left_margin=sec.right_margin=Inches(1)
BODY_W=sec.page_width.inches-sec.left_margin.inches-sec.right_margin.inches
for name in ['Normal','Body Text','List Paragraph','Caption']:
 st=doc.styles[name];st.font.name='Malgun Gothic';st._element.get_or_add_rPr().get_or_add_rFonts().set(qn('w:eastAsia'),'Malgun Gothic');st.font.size=Pt(10.5 if name!='Caption' else 9)
 st.paragraph_format.space_after=Pt(5);st.paragraph_format.line_spacing=1.2
for name,size,color in [('Title',23,'173B55'),('Heading 1',15,'173B55'),('Heading 2',11.5,'244F6C')]:
 st=doc.styles[name];st.font.name='Malgun Gothic';st._element.get_or_add_rPr().get_or_add_rFonts().set(qn('w:eastAsia'),'Malgun Gothic');st.font.size=Pt(size);st.font.color.rgb=RGBColor.from_string(color);st.paragraph_format.space_before=Pt(10);st.paragraph_format.space_after=Pt(7);st.paragraph_format.keep_with_next=True
header=sec.header.paragraphs[0];header.text='제조 비전검사 화소 손상 위치 민감도 | 결과 재정리';header.style='Caption'
footer=sec.footer.paragraphs[0];footer.alignment=1
r=footer.add_run('2026-11-12  ·  ');r.font.size=Pt(8)
for kind,text in [('begin',None),(None,'PAGE'),('end',None)]:
 el=OxmlElement('w:fldChar' if kind else 'w:instrText')
 if kind:el.set(qn('w:fldCharType'),kind)
 else:el.text=text
 footer.add_run()._r.append(el)
settings=OxmlElement('w:updateFields');settings.set(qn('w:val'),'true');doc.settings.element.append(settings)
doc.core_properties.title='센서 화소 결함 위치 민감도 — 진행 상황 및 결과 재정리';doc.core_properties.subject='검증 완료된 pixel_sensitivity 실험 결과';doc.core_properties.author='Research report';doc.core_properties.comments='Reference DOCX used for layout/organization only. Experimental content is from the verified current project.'
blocks=[]
def para(text,style=None):
 p=doc.add_paragraph(text,style=style);blocks.append({'kind':'p','text':text,'style':style});return p
def heading(text,level=1):
 p=doc.add_heading(text,level);blocks.append({'kind':'h','text':text,'level':level});return p
def page():doc.add_page_break();blocks.append({'kind':'break'})
def table(headers,rows,widths=None):
 t=doc.add_table(rows=1, cols=len(headers));t.style='Table Grid';t.autofit=False
 if widths is None:widths=[1]*len(headers)
 widths=[BODY_W*w/sum(widths) for w in widths]
 for i,(cell,text) in enumerate(zip(t.rows[0].cells,headers)):cell.text=str(text)
 for row in rows:
  cells=t.add_row().cells
  for c,v in zip(cells,row):c.text=str(v)
 for i,col in enumerate(t.columns):col.width=Inches(widths[i])
 for ri,row in enumerate(t.rows):
  prop=row._tr.get_or_add_trPr();cant=OxmlElement('w:cantSplit');prop.append(cant)
  if ri==0:
   repeat=OxmlElement('w:tblHeader');repeat.set(qn('w:val'),'true');prop.append(repeat)
  for i,cell in enumerate(row.cells):
   cell.width=Inches(widths[i]);cell.vertical_alignment=1
   if ri==0:
    sh=OxmlElement('w:shd');sh.set(qn('w:fill'),'244F6C');cell._tc.get_or_add_tcPr().append(sh)
   elif ri%2==0:
    sh=OxmlElement('w:shd');sh.set(qn('w:fill'),'F1F5F8');cell._tc.get_or_add_tcPr().append(sh)
   for p in cell.paragraphs:
    p.paragraph_format.space_after=Pt(2);p.paragraph_format.space_before=Pt(2);p.paragraph_format.line_spacing=1.05
    for r in p.runs:
     r.font.name='Malgun Gothic';r._element.get_or_add_rPr().get_or_add_rFonts().set(qn('w:eastAsia'),'Malgun Gothic');r.font.size=Pt(8.5)
     if ri==0:r.font.bold=True;r.font.color.rgb=RGBColor(255,255,255)
 blocks.append({'kind':'table','headers':headers,'rows':[[str(v) for v in r] for r in rows]});return t
def picture(path,caption,width=None,maxheight=6.7):
 path=Path(path);w,h=Image.open(path).size;ww=min(BODY_W,width or BODY_W,maxheight*w/h)
 p=doc.add_paragraph();p.alignment=1;p.add_run().add_picture(str(path),width=Inches(ww));p.paragraph_format.space_after=Pt(3)
 para(caption,'Caption');blocks.append({'kind':'image','path':str(path),'caption':caption,'width_in':ww})
 # The caption is recorded separately above; render helpers suppress that duplicate.
 blocks[-2]['image_caption']=True

def value(d,c,col):return float(cond[(cond.dataset==d)&(cond.seed==11)&(cond.condition==c)][col].iloc[0])
def fmt(x,n=3):return f'{float(x):.{n}f}'
font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
# Compact primary-model map, derived only from the final verified numeric tables.
fig,axes=plt.subplots(1,3,figsize=(12,4.3),layout='constrained');lim=14.5
for ax,d in zip(axes,DATA):
 row=cond[(cond.dataset==d)&(cond.seed==11)&(cond.condition=='bright_k25')].iloc[0];s=pd.read_csv(Path(row.run_path)/'analysis/grid_summary.csv').sort_values('cell');ar=s.sensitivity_pp.to_numpy().reshape(8,8)
 im=ax.imshow(ar,cmap='RdBu_r',vmin=-lim,vmax=lim)
 for y in range(8):
  for x in range(8):ax.text(x,y,f'{ar[y,x]:.1f}',ha='center',va='center',fontsize=6.5,color='white' if abs(ar[y,x])>lim*.55 else '#17212b')
 ax.set_title(NAMES[d],fontproperties=font,fontsize=11);ax.set_xticks([]);ax.set_yticks([])
fig.colorbar(im,ax=list(axes),shrink=.8,label='Recall drop (pp)');fig.savefig(OUT/'primary_seed11_maps.png',dpi=230);plt.close(fig)
cells=pd.read_csv(BASE/'corrected_pcb_distribution/pcb_ind_board_cell_exposure.csv').sort_values('cell')
fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
for ax,key,title in zip(axes,['recall_drop_pp','crossfit_expected_drop_pp','descriptive_residual_drop_pp'],['관찰된 감소','겹침·크기·클래스 기준모형','기술적 잔차']):
 ar=cells[key].to_numpy().reshape(8,8);im=ax.imshow(ar,cmap='RdBu_r',vmin=-14.5,vmax=14.5)
 for y in range(8):
  for x in range(8):ax.text(x,y,f'{ar[y,x]:.1f}',ha='center',va='center',fontsize=6.5,color='white' if abs(ar[y,x])>8 else '#17212b')
 ax.set_title(title,fontproperties=font,fontsize=10);ax.set_xticks([]);ax.set_yticks([])
fig.colorbar(im,ax=list(axes),shrink=.8,label='Recall drop (pp)');fig.savefig(OUT/'corrected_pcb_exposure_residual.png',dpi=230);plt.close(fig)
# Page1: same narrative sequence as the reference, current verified content only.
para('센서 화소 결함 위치 민감도 파일럿','Title');para('진행 상황 및 결과 재정리 · 2026-11-12')
para('목차·표·그림 배치와 서술 형식은 pilot_progress_report.docx를 참고했다. 실험 내용과 수치는 pixel_sensitivity 프로젝트에서 실제 완료·검증한 결과만 사용했다. 참고 파일의 실험 수치·설정·해석은 옮기지 않았다.')
heading('요약')
for text in [
 '3개 데이터셋과 3개 학습 시드에 대한 주 모델 9개를 평가했다. 추가 학습7개와 모델·조건27개 평가 조합을 완료했으며, 각 평가는8×8위치·10개 배치를 사용했다.',
 '시드11에서 화소 수1→9→25가 증가할수록 평균 Recall 감소가 커졌고, 같은25개에서 부분 강도는 완전 치환보다 손실이 작았다.',
 '밝은25개 조건의 3학습 시드 평균 Recall 감소는 GC10 0.049±0.087%p, NEU 0.814±0.236%p, 새 PCB 2.118±0.378%p였다. ±는 시드 간 표본 표준편차이며 신뢰구간이 아니다.',
 'GC10은 Recall 감소와 최대 위치가 학습 시드에 따라 안정적으로 재현되지 않았다. NEU와 PCB는 세 모델 모두 평균 Recall이 감소했고, PCB는 세 모델 모두 G37이 최대였다.',
 '새 PCB 시드11의 공동 위치 밴드에서는 G28·G29·G36·G37이 전체 위치 평균보다 손실이 컸다. 제품 결함 노출과의 순위상관은0.889였으며, 이 연관을 센서 고유 좌표의 인과 효과로 해석하지 않는다.'
]:para(text,'List Bullet')
heading('1. 진행 현황')
table(['단계','상태','확인된 산출물'],[
 ['데이터·주석·분할 준비','완료','GC10·NEU 정리본, PCB 숫자 ID 분리본'],['기존 캐시 분포·공간 분석','완료','클래스·크기·겹침, 공동 bootstrap 밴드'],['추가 학습 및 학습 반복','완료','추가7모델; 주 실험9모델'],['유형·수·강도·군집 비교','완료','시드11 7조건 ×3자료'],['학습 시드 반복 비교','완료','밝은25개 조건의 시드11·22·33'],['평가·결과 검증','완료','27조합, 17,307예측 묶음 재해시, 테스트79개 통과']],[1.6,.6,3.0])
# Page2 settings
page();heading('2. 실험 설정')
heading('2.1 공통 모델과 평가 절차',2)
for text in [
 '모델은 객체 탐지용 YOLOv8n이다. 동일한 원본 COCO 사전학습 가중치에서 데이터셋별로 독립 학습했다. imgsz640, 최대100epoch, patience20, AdamW(lr0=0.001), 학습 batch16, deterministic=True를 사용했다.',
 '학습 시드11·22·33을 비교했다. 공간·색상 증강은 모두0으로 고정했고, 화소 손상을 넣지 않은 자료로 학습했다. 검증 자료의 최적 체크포인트를 고른 뒤 모델을 고정해 시험했다.',
 '손상은 원본 해상도의 uint8 RGB 배열에 넣은 뒤 동일한 YOLO 전처리를 통과한다. 한 번의 추론에는 8×8 중 한 칸에만 주입한다. 같은 배치를64칸으로 옮겨 평가하며,10개 배치를 사용한다.',
 '유형·수·강도·군집의7조건 비교는 학습 시드11에서 수행했다. 학습 시드22·33은 주 조건인 밝은25개·분산에서 반복했다. 모든 유형을3시드로 전수 교차한 실험은 아니다.'
]:para(text)
table(['항목','고정 규칙'],[['원본 기준','추가 화소 손상이 없는 시험 영상; 제품 결함은 원래 존재'],['운영 지표','confidence≥0.25, 같은 클래스·IoU≥0.5의 score-first 일대일 대응'],['예측 저장','conf floor0.001, NMS IoU0.7, max_det300, FP32·batch16'],['AP','동일한 공식 COCO evaluator; maxDets100'],['주 지표','Recall 감소량=원본 Recall−손상 후 Recall, 단위%p'],['보조 지표','Precision, mAP50, mAP50:95, confidence 및 FP/image'],['평균','각 위치에서 시험자료 전체 평가 →64위치·10배치 동일 가중 평균']],[1.2,4.0])
heading('2.2 주입 조건',2)
table(['조건','목표값/강도','공간 화소 수','배치'],[['밝은 화소','255 / α=1','1·9·25','분산'],['어두운 화소','0 / α=1','25','같은25좌표'],['128 고정값','128 / α=1','25','같은25좌표'],['부분 강도','255 / α=0.5','25','같은25좌표'],['군집','255 / α=1','25','연속5×5 한 블록']],[1.25,1.35,.9,1.6])
para('화소 수1·9는 같은25개 순서의 앞부분이다. 값·강도 비교는 동일한25좌표를 사용한다. 군집은 별도 layout/side seed로 생성하므로 명령 화소 수는 같지만 실제 위치와 GT 노출이 같지는 않다. 원래 값이 목표값과 같으면 실제 변경 수가 줄 수 있어 원자료에 기록했다.','Caption')
# Page3 data and baseline
page();heading('3. 데이터셋과 베이스라인')
table(['자료','원본 크기','학습 / 검증 / 시험','시험 GT','분할 단위'],[['GC10-DET','2048×1000','1639 / 299 / 354','587','파일명 A+B 대리 그룹'],['NEU-DET','200×200','1258 / 270 / 270','612','이미지 클래스별 분리'],['PCB-IND 새 분할','300×300','3295 / 728 / 639','948','숫자 ID; 양면·패치 통합']],[1.1,1.0,1.65,.65,1.65])
para('세 자료의 제조 대상과 클래스 수는 각각 금속 표면10종, 열연 강판6종, PCB 결함8종이다. 표의 PCB 주 결과는 숫자 ID로 새로 분리한 자료이며 이전 공식 분할의 성능과 같은 시험 코호트가 아니다.')
rows=[]
for d in DATA:
 for r in baselines[baselines.dataset==d].sort_values('seed').itertuples():
  rows.append([NAMES[d],str(r.seed),str(r.n_training_epochs),fmt(r.recall_pct) if hasattr(r,'recall_pct') else fmt(r.clean_recall_pct),fmt(r.precision_pct) if hasattr(r,'precision_pct') else fmt(r.clean_precision_pct),fmt(r.map50_pct) if hasattr(r,'map50_pct') else fmt(r.clean_map50_pct),fmt(r.map50_95_pct) if hasattr(r,'map50_95_pct') else fmt(r.clean_map50_95_pct)])
table(['자료','시드','완료epoch','Recall','Precision','mAP50','mAP50:95'],rows,[1.45,.45,.65,.75,.8,.8,.9])
para('표1. 손상 없는 원본 시험 성능(%). Recall·Precision은 고정 운영 기준, AP는 동일 COCO 평가기로 계산했다. 완료epoch는 실제 학습 길이이며, 평가에는 검증 최적 best.pt를 사용했다.','Caption')
heading('3.1 PCB 새 분할: 클래스별 원본 성능',2)
rows=[]
for r in perclass[perclass.dataset=='pcb_ind_board'].itertuples():rows.append([r.class_name,str(r.n_gt),fmt(r.recall_pct),fmt(r.precision_pct) if pd.notna(r.precision_pct) else '정의 안 됨',fmt(r.coco_map50_pct),fmt(r.coco_map50_95_pct)])
table(['클래스','GT','Recall','Precision','mAP50','mAP50:95'],rows,[1.6,.4,.8,.85,.8,.9])
para('표2. 시드11, 시험639장. 클래스별 AP는 기존 clean 예측 캐시만 재평가했으며 전체 AP가 저장된 기준값과 일치함을 확인했다. 클래스별 표본 수가 달라 작은 클래스의 수치 변동을 함께 고려해야 한다.','Caption')
# Page4 conditions recall +figure
page();heading('4. 실험 A — 화소 수·값·강도·군집 비교')
para('아래는 학습 시드11의 Recall(%). 각 손상 값은64위치×10배치 평균이며, 원본과 같은 시험 자료·모델·판정 기준을 사용했다. 별도 결과에 맞춘 재학습이나 임계값 조정은 하지 않았다.')
rows=[['손상 없는 원본']+[fmt(float(cond[(cond.dataset==d)&(cond.seed==11)&(cond.condition=='bright_k25')].clean_recall_pct.iloc[0])) for d in DATA]]
for c in CS:rows.append([LABELS[c]]+[fmt(value(d,c,'mean_recall_pct')) for d in DATA])
table(['조건','GC10-DET','NEU-DET','PCB 새 분할'],rows,[1.7,1.1,1.1,1.2])
para('표3. 원본과 손상 후 평균 Recall. 군집은25개의 연속 화소이며 분산25와 개수를 맞췄다.','Caption')
picture(SUM/'rq3_conditions_recall_drop.png','그림1. 조건별 전체64위치·10배치 평균 Recall 감소(%p), 시드11.',maxheight=3.0)
for text in ['화소 수1→9→25에서 평균 손실이 증가했다. 부분 강도는 같은25개 완전 치환보다 세 자료 모두 손실이 작았다.','유형의 순서는 자료마다 달랐다. NEU는 어두운25가 밝은25보다 손실이 컸고, GC10·PCB는 반대였다.','군집이 항상 더 큰 손실을 만들지는 않았다. GC10에서는 군집25가 분산25보다 컸지만, NEU는 비슷했고 PCB에서는 작았다. 배치와 GT 노출 차이를 함께 보아야 한다.']:para(text,'List Bullet')
# Page5 full conditional metrics
page();heading('4.1 조건별 전체 지표 — 시드11',2)
para('성능 값의 단위는%, Recall 감소량의 단위는%p이다. 감소량은 원본−손상 후이다. FP와 Precision은 클래스별 일대일 정답 대응을 사용하는 동일 운영 기준이다.')
rows=[]
for d in DATA:
 for c in CS:rows.append([NAMES[d],LABELS[c],fmt(value(d,c,'mean_recall_pct')),fmt(value(d,c,'mean_precision_pct')),fmt(value(d,c,'mean_map50_pct')),fmt(value(d,c,'mean_map50_95_pct')),fmt(value(d,c,'recall_drop_pp'))])
table(['자료','조건','Recall','Precision','mAP50','mAP50:95','R감소'],rows,[1.1,1.5,.72,.78,.75,.8,.7])
para('표4. 21개 기본 조건의 평균. 원본 값은 표1에 있다. 음수 변화는 개선으로 그대로 기록했다. 이 표의 조건 효과를 다른 학습 시드에도 일반화하지 않는다.','Caption')
# Page6 maps
page();heading('5. 실험 B — 8×8 위치 스윕')
para('밝은25개·분산 조건을 각 칸에 주입해 원본 대비 Recall 감소를 계산했다. 각 칸은10개 배치의 평균이다. G1은 왼쪽 위, G64는 오른쪽 아래이며 행 우선으로 번호를 부여한다.')
picture(OUT/'primary_seed11_maps.png','그림2. 학습 시드11의 위치별 평균 Recall 감소(%p). 세 지도는 동일 색상 범위를 사용한다. 숫자는 실제 센서 절대 좌표가 아닌 입력 영상 안의 위치이다.',maxheight=3.0)
rows=[]
for d in DATA:
 r=cond[(cond.dataset==d)&(cond.seed==11)&(cond.condition=='bright_k25')].iloc[0];rows.append([NAMES[d],fmt(r.recall_drop_pp),r.max_location,fmt(r.max_location_drop_pp)])
table(['자료','전체 위치 평균 감소','최대 칸','최대 칸 감소'],rows,[1.5,1.5,.8,1.5])
para('표5. 단위%p. 최대 칸은64개 후보 중 사후에 선택한 기술통계이며, 이 값만으로 해당 위치의 고유 취약성이나 다른 위치 대비 통계적 차이를 확정하지 않는다.','Caption')
heading('5.1 위치 비교에서 유지한 조건',2)
for text in ['같은 배치 안에서는 유형·개수·값 변경 규칙을 유지하고 주입 위치만 옮겼다. 한 번의 추론에64칸 모두를 손상시키지 않았다.','원본 GT는 그대로 유지했다. 주입한 화소를 모델이 제품 결함으로 검출해도 새 정답을 추가하지 않는다.','균등한64칸 평균은 위치 스캔을 요약하는 규칙이며, 실제 카메라 고장이 그 확률로 발생한다는 뜻이 아니다.']:para(text,'List Bullet')
# Page7 full seed map
page();heading('5.2 학습 시드별 위치지도',2)
picture(SUM/'training_seed_spatial_maps.png','그림3. 행은GC10·NEU·새PCB, 열은시드11·22·33이다. 밝은25개·분산, 각 칸10배치 평균. 공통 색상 범위이므로 작은 변화는 옅게 보이며 숫자와 함께 읽는다.',maxheight=7.1)
para('PCB는 세 모델 모두G37이 최대였고 상위4칸 집합이 같았다. GC10은 최대 위치와 지도 모양의 반복성이 낮았으며, NEU는 중간 정도의 지도 일치도를 보였다.')
# Page8 replication
page();heading('6. 추가 검증 — 학습 반복과 해석')
heading('6.1 학습 시드 반복: 밝은25개·분산',2)
rows=[]
for d in DATA:
 for r in seed_results[seed_results.dataset==d].sort_values('seed').itertuples():rows.append([NAMES[d],str(r.seed),fmt(r.recall_drop_pp),fmt(r.map50_drop_pp),r.max_location])
table(['자료','시드','Recall 감소','mAP50 감소','관찰 최대 칸'],rows,[1.6,.45,1.1,1.1,1.0])
para('표6. 감소량 단위%p. 같은 자료의 시드별 시험 영상과 주입 배치를 맞췄으며, 손상 후 성능은 각 모델 자신의 원본 기준과 비교했다.','Caption')
rows=[]
for d in DATA:
 r=seed_summary[seed_summary.dataset==d].iloc[0];rows.append([NAMES[d],f"{r.recall_drop_pp_seed_mean:.3f} ± {r.recall_drop_pp_seed_sd:.3f}",f"{r.map50_drop_pp_seed_mean:.3f} ± {r.map50_drop_pp_seed_sd:.3f}"])
table(['자료','Recall 감소 평균 ± SD','mAP50 감소 평균 ± SD'],rows,[1.6,2.0,2.0])
para('표7. 시드3개 평균과 표본 표준편차(ddof=1)이다. 신뢰구간이 아니며 모든 모델·촬영환경의 재현성을 보장하지 않는다. GC10은 Recall의 감소 방향도 시드에 따라 달랐지만 mAP50은 세 시드에서 작은 감소가 있었다.','Caption')
picture(SUM/'training_seed_recall_drop.png','그림4. 모델 학습 반복 결과. 세로축은 전체64위치·10배치 평균 Recall 감소(%p).',maxheight=2.25)
# Page9 descriptive confounding
page();heading('6.2 제품 결함 분포·크기·겹침',2)
para('GT 중심이 영상의 가로·세로 중앙25% 영역(전체 면적6.25%)에 있는 비율과, clean에서 검출된 객체의 손상 노출을 함께 분석했다. 상자 겹침은 실제 결함 픽셀의 정밀 마스크가 아니다.')
rows=[]
for d in DATA:
 r=dist[dist.dataset==d].iloc[0];ss=spatial_sd[spatial_sd.dataset==d].iloc[0]
 rows.append([NAMES[d],f'{100*r.central_center_fraction:.2f}%',f'{100*r.median_gt_area_fraction:.2f}%',fmt(r.spearman_drop_clean_tp_exposure),fmt(ss.raw_spatial_sd_pp),fmt(ss.residual_spatial_sd_pp)])
table(['자료','중앙 GT','GT면적 중앙값','감소–노출 ρ','원래 SD','잔차 SD'],rows,[1.2,.8,1.15,.95,.8,.8])
para('표8. GT면적은 영상 면적 대비 비율이고 SD의 단위는%p이다. 같은 정의(ddof=0)의 공간 표준편차를 비교했다. 클래스×면적3분위×상자 겹침 화소수로 기술적 기준모형을 만들고 이미지 대리 그룹5-fold 교차적합을 사용했다.','Caption')
picture(OUT/'corrected_pcb_exposure_residual.png','그림5. 새PCB의 관찰 손실·기술적 기준모형·잔차. 기준모형은 위치 셀 번호를 사용하지 않았다. 공통 색상 범위이며 인과 보정 결과가 아니다.',maxheight=2.8)
para('새PCB G37의 관찰 감소는14.061%p, 클래스·크기·겹침 기준모형의 예상 감소는11.922%p, 잔차는2.139%p였다. 전체 공간 SD는2.557%p에서 잔차1.072%p로 작아졌다. 이 변화율을 위치 효과의 인과적 설명률로 계산하지 않는다.')
para('겹침·노출과 강하게 연결되지만, 밝기·제품 구조·정답 상자의 여유 영역·모델 의사결정 경계 등 측정하지 않은 구성 요인이 남는다. 잔차가 있다는 것만으로 센서 고유 좌표 효과가 입증되는 것은 아니다.')
# Page10 statistics
page();heading('6.3 위치 간 차이의 공동 불확실성',2)
para('밝은25개·분산, 학습 시드11을 대상으로 같은 이미지 그룹과 패턴 가중치를64위치에 공통 적용해4,000회 재표집했다. 칸별 손실의 전체 위치 평균 대비 편차를 중심화하고, 최대 절대 bootstrap 오차로 명목95% 근사 동시 밴드를 만들었다.')
rows=[]
for i,d in enumerate(DATA):
 r=pos.iloc[i];selected=r.above_grand_mean_cells if pd.notna(r.above_grand_mean_cells) else '확인 안 됨'
 rows.append([NAMES[d],fmt(r.grand_mean_drop_pp),f'[{r.grand_mean_ci_low_pp:.3f}, {r.grand_mean_ci_high_pp:.3f}]',fmt(r.deviation_band_radius_pp),selected])
table(['자료','평균 감소','평균95%구간','편차 밴드 반경','평균 초과 위치'],rows,[1.3,.8,1.55,1.0,1.4])
para('표9. 단위%p. 평균 구간과 위치 편차 동시 밴드는 서로 다른 양이다. GC10·NEU에서 선정된 칸이 없다는 것은 공간 차이가 없다는 증명이 아니다.','Caption')
for text in ['새PCB는 기록된 그룹이 이미 숫자 ID103개다. 기록 그룹과 숫자 ID 재표집 결과가 같은 것은 일치성 확인이며, 두 독립 검증으로 세지 않는다.','이 공동 밴드는 학습 시드11과 현재 시험 자료·10배치에 조건부다. 학습 시드 불확실성, 전체27조건에 걸친 다중 비교, 확증적 p-value를 포함하지 않는다.','재표집 그룹은 실제 제품·코일 ID로 완전히 검증된 값이 아니다. PCB는 숫자 ID 분리로 중복 가능성을 줄였고, GC10·NEU는 확보한 대리 그룹의 범위로 해석한다.','두 방향 교차 재표집의 방법적 근거는 Owen(2007), Owen·Eckles(2012)이다. 현재 비율·최대편차 통계에 대한 유한표본 보장을 별도로 증명한 것은 아니다.']:para(text,'List Bullet')
heading('6.4 기술적 지도 일치도',2)
rows=[]
for d in DATA:
 x=corr[corr.dataset==d];rows.append([NAMES[d],f'{x.pearson_r.min():.3f} ~ {x.pearson_r.max():.3f}' if 'pearson_r' in x else '',f'{x.spearman_rho.min():.3f} ~ {x.spearman_rho.max():.3f}' if 'spearman_rho' in x else '',f'{x.top4_jaccard.min():.3f} ~ {x.top4_jaccard.max():.3f}' if 'top4_jaccard' in x else ''])
# Use exact CSV column names without inferring missing correlations.
print('CORRELATION COLUMNS',list(corr.columns))
# Data-generated table is appended after column normalization below.
for row,d in zip(rows,DATA):
 x=corr[corr.dataset==d]
 for out_i,choices in [(1,['pearson_spatial_map']),(2,['spearman_spatial_map']),(3,['top4_cell_jaccard'])]:
  key=next((k for k in choices if k in x.columns),None)
  if key:row[out_i]=f'{x[key].min():.3f} ~ {x[key].max():.3f}'
assert all(all(str(v).strip() for v in row) for row in rows), 'Missing spatial correlation values'
table(['자료','지도 Pearson r 범위','지도 Spearman ρ 범위','상위4칸 Jaccard 범위'],rows,[1.1,1.7,1.7,1.7])
para('표10. 동일 자료의 시드3개에서 가능한3쌍의 범위이다. 이는 기술적 일치도이며 위치 차이의 유의성 검정이 아니다.','Caption')
# Page11 conclusions and limits
page();heading('7. 연구 질문별 결과와 한계')
table(['질문','확인한 결과','해석 범위'],[
 ['RQ1. 화소 손상이 성능을 바꾸는가?','NEU·PCB는 밝은25에서3시드 모두 평균 Recall 감소. GC10 Recall 변화는 작고 시드별 방향도 다름.','모든 모델·센서에서 항상 저하한다고 일반화하지 않는다.'],
 ['RQ2. 같은 수라도 위치에 따라 다른가?','새PCB는 중앙4칸이 공동 밴드상 평균보다 높은 손실. 3모델 모두G37최대, 상위4칸 일치.','결함 분포·노출과 강한 연관. 센서 절대 좌표의 인과적 취약성은 확인하지 못함.'],
 ['RQ3. 유형·강도·크기/배치 영향은?','시드11에서 수·강도 증가 시 평균 손실 증가. 유형 순위·군집 효과는 자료마다 다름.','7조건은시드11 비교이고, 군집은실제좌표·노출이 달라 연결성만의 효과가 아님.']],[1.35,2.5,2.3])
heading('7.1 공통 한계',2)
for text in ['손상은 처리된RGB영상의 유사 화소 이상이다. 실제RAW센서·demosaic·ISP·DPC·온도·노출 의존 고장을 검증하지 않았다.','원본 해상도와resize배율이 다르다. GC10은축소, NEU·PCB는확대되므로 같은25개가 같은 손상 비율이나 모델 입력상의 면적을 뜻하지 않는다.','PCB숫자ID분할은 대리 그룹 분리이며 실제 물리 보드·로트 독립성은 미확인이다. GC10파일명A+B그룹과NEU이미지ID도 실제코일·제품 독립성을 보장하지 않는다.','현재 주 분석은 제품 결함 객체 단위이다. 제품 단위 출하 미검출률·정상제품 오거부율·카메라 수명 또는 교체비를 측정하지 않았다.','같은 이미지와GT가 여러 위치·배치에 반복된다. 관측행 수를 독립 표본 수로 해석하지 않고 그룹·배치 공동 재표집과 별도 학습 반복을 구분했다.','조건 비교·분포 진단은 탐색에 사용한 자료의 추가 분석이다. 독립 확증 실험이나 모든 비교를 통합한 전체 오류율 보장으로 표현하지 않는다.']:para(text,'List Bullet')
# Page12 preparation
page();heading('8. 데이터셋 준비·검증 현황')
for d in DATA:
 heading(NAMES[d],2)
 if d=='gc10_det':
  para('금속 표면10종, 2048×1000. 확보한 공개 mirror2,300장 중 미확인 빈 라벨8장을 제외해2,292장을 사용했다. 파일명 접두어A+B 대리 그룹을 분할했고, 원본 픽셀·주석·클래스 순서를 고정했다.')
 elif d=='neu_det':
  para('열연 강판6종, 200×200. 원 배포처1,800장 중 동일 픽셀이지만 주석이 충돌하는2장을 제외했다. 중복 상자3개를 정리하고 difficult객체도 정답에 포함했다. VOC1-based inclusive좌표를0-based half-open으로 바꿨다. 이미지 클래스별70/15/15, seed101이다.')
 else:
  para('산업AOI기판ROI8종, 300×300. 공식v4의4789장 중 정상 여부를 확인하지 못한 빈 라벨127장을 제외한4662장을 사용했다. 좌표 반올림 문제를 피하기 위해 공식COCO주석을 기준으로 변환했다. 숫자ID별 양면·패치를 묶어478/102/103그룹으로 분리했고 교집합은0이다.')
table(['항목','현재 확인 상태'],[['원본 보존','raw 폴더 보존, canonical PNG/YOLO·manifest는 별도'],['동일 픽셀 중복','decoded 픽셀 해시 점검; NEU주석 충돌쌍 처리'],['GT대응','조건에 따라 원래 GT를 수정하지 않음'],['분할·모델','data/classes/weights/source/설정 해시 보존'],['유효 결과','예측17,307묶음, 이미지 평가7,286,247회'],['소프트웨어검사','테스트79개 통과; 최종 예측 전체 재해시 검증']],[1.4,4.7])
picture(ROOT/'reports/bright25_grid8_2026-11-10/fault_sites/single_cell_native_pixel_masks.png','그림6. 첫 번째 배치의 G1 내부 25개 주입 대상 위치. 흰 점은 원본의 한 화소이고 검은 배경은 위치 마스크이다. 각 자료의 칸 크기 차이를 보여주며, 실제 제품 영상의 밝기와 다르다.',maxheight=2.6)
# Page13 closing and reproduction
page();heading('9. 정리 및 산출물')
para('현재 합의한 범위의 추가 검증은 완료됐다. 보고서의 주장은 입력영상·검사 모델·현재 데이터 분할에 조건부인 화소 손상 민감도로 한정한다. 강한PCB중앙 손실은 반복 모델에서도 관찰되지만, 제품 결함 노출 구성을 센서 좌표 자체의 취약성과 혼동하지 않는 것이 핵심이다.')
para('GC10에서 낮은 평균 Recall변화를 곧바로 무영향으로 판정하거나, 시드11의 작은 감소만으로 모든 모델의 일관된 저하를 주장하지 않는다. NEU·PCB의 반복된 평균 감소와 자료별 유형·배치 차이를 함께 제시한다.')
heading('부록 A. 재현과 파일 위치',2)
para('프로젝트 루트: /home/hyochang/pixel_sensitivity/ . 아래 경로는 이 루트 기준이다. 보고서 작성 과정에서는 새 학습·손상 추론을 실행하지 않고 완료 자료를 읽어 재정리했다. 클래스별cleanAP는 동일 캐시·평가기에서 재계산해 전체 기준값과 대조했다.')
table(['내용','파일 / 디렉터리'],[['주 실험 전체 요약','reports/research_validation_2026-11-11/summary/REPORT_KO.md'],['27조건 평균·패턴 원자료','summary/condition_means.csv, pattern_means.csv'],['3학습시드 결과','summary/bright25_seed_results.csv, bright25_seed_summary.csv'],['분포·공동밴드','distribution/, positions/, corrected_pcb_distribution/, corrected_pcb_positions/'],['PCB새분할검증','pcb_split/REPORT_KO.md, inputs/pcb_ind_board_source_audit.json'],['모델·예측·분석','runs/research_validation_compiled_20261111/'],['작업·완료표식','evaluation_manifest.json, evaluation_state.json, COMPLETE.json'],['설정·재현','configs/research_validation/, README.md'],['원본·실험용데이터','data/raw/, data/canonical/']],[1.35,4.8])
para('summary/ 및 분포·밴드·분할 보고서의 상대경로는 reports/research_validation_2026-11-11/ 아래다. 실제 주 모델9개와 데이터·평가 결과 지문을 함께 보존했다.','Caption')
heading('부록 B. 검증과 통계 방법',2)
para('실행 최적화는 원래FP32·batch16을 유지했다. NMS의argsort/동점순서를 보존하고 IoU경계·비정상/큰후보는 원래경로로 돌린다. 고정validation control과9개 체크포인트의 별도control에서 예측 일치를 확인했다. 이는 제한된수치검증이지 임의입력의 수학적동등성 증명은 아니다.')
para('통계 방법 근거: Owen(2007), The Pigeonhole Bootstrap, DOI10.1214/07-AOAS122; Owen·Eckles(2012), Bootstrapping data arrays of arbitrary order, DOI10.1214/12-AOAS547. 현재 비율·최대편차의 유한표본성질은 별도 보장으로 주장하지 않았다.')
para('작성 기준: 참고파일은 형식만 사용했다. 본 보고서에 등장하는 실험 설정·표·그림·판정은 현재 프로젝트의 완료 산출물과 직접 연결했다. 원본 pilot_progress_report.docx는 변경하지 않았다.','Caption')
# Save DOCX + portable content representation / same-content HTML for PDF.
doc.save(DOCX)
css="""@page{size:A4;margin:25.4mm;@bottom-center{content:counter(page);font-size:8pt;color:#66717b}}body{font-family:'Noto Sans CJK KR','Noto Sans CJK JP',sans-serif;font-size:10pt;line-height:1.35;color:#17212b}h1{font-size:15pt;color:#173b55;margin:10pt 0 7pt;break-after:avoid}h2{font-size:11.5pt;color:#244f6c;break-after:avoid}.title{font-size:23pt;color:#173b55;line-height:1.25}p{margin:0 0 5pt;orphans:2;widows:2}.caption{font-size:8.7pt;color:#586571}table{border-collapse:collapse;width:100%;font-size:8pt;line-height:1.2;margin:5pt 0 8pt;table-layout:auto}th,td{border:.5pt solid #b8c3cd;padding:4pt;vertical-align:middle;overflow-wrap:anywhere}th{background:#244f6c;color:white}tr:nth-child(even){background:#f1f5f8}thead{display:table-header-group}tr{break-inside:avoid}.pb{break-before:page}.figure{text-align:center;break-inside:avoid}.figure img{max-width:100%}.bullet{margin-left:12pt;text-indent:-8pt}.bullet:before{content:'• ';color:#244f6c}"""
html=['<!doctype html><html lang="ko"><meta charset="utf-8"><style>'+css+'</style><body>']
for b in blocks:
 if b['kind']=='p':
  if b.get('image_caption'):continue
  cls='title' if b['style']=='Title' else ('caption' if b['style']=='Caption' else ('bullet' if b['style']=='List Bullet' else ''))
  html.append(f'<p class="{cls}">{escape(b["text"])}</p>')
 elif b['kind']=='h':html.append(f'<h{b["level"]}>{escape(b["text"])}</h{b["level"]}>')
 elif b['kind']=='break':html.append('<div class="pb"></div>')
 elif b['kind']=='table':
  html.append('<table><thead><tr>'+''.join('<th>'+escape(x)+'</th>' for x in b['headers'])+'</tr></thead><tbody>')
  for row in b['rows']:html.append('<tr>'+''.join('<td>'+escape(x)+'</td>' for x in row)+'</tr>')
  html.append('</tbody></table>')
 elif b['kind']=='image':html.append(f'<div class="figure"><img src="{Path(b["path"]).as_uri()}" style="width:{b["width_in"]}in"><p class="caption">{escape(b["caption"])}</p></div>')
html.append('</body></html>');(OUT/'report.html').write_text('\n'.join(html));dump_json(OUT/'report_content.json',blocks)
# Check source preservation and required quantitative anchors.
assert digest_file(SOURCE)==source_sha
check=Document(DOCX);flat='\n'.join(p.text for p in check.paragraphs)+'\n'+'\n'.join(c.text for t in check.tables for r in t.rows for c in r.cells)
for token in ['7,286,247','17,307','0.049','2.118','14.061','11.922','G28','79개']:assert token in flat,token
for prohibited in ['column','1.23 [0.79','32.02%','6×6 위치 스윕','pixel_defect_pilot/','98~99%']:assert prohibited not in flat,prohibited
inputs=[SUM/'condition_means.csv',SUM/'clean_baselines.csv',SUM/'bright25_seed_summary.csv',BASE/'corrected_pcb_distribution/dataset_summary.csv',BASE/'corrected_pcb_positions/position_effect_summary.csv',OUT/'clean_per_class_seed11.csv']
dump_json(OUT/'report_provenance.json',{'date':'2026-11-12','source_docx':str(SOURCE),'source_docx_sha256':source_sha,'reference_use':'format and narrative organization only; no reference experimental contents reused','current_series_inputs':{str(p.relative_to(ROOT)):digest_file(p) for p in inputs},'output_docx':str(DOCX),'output_docx_sha256':digest_file(DOCX),'script_sha256':digest_file(__file__),'tables':len(check.tables),'inline_figures':len(check.inline_shapes),'word_rendering':'OOXML structure checked; companion PDF uses same content HTML, not Microsoft Word pagination'})
print('Saved',DOCX,'tables',len(check.tables),'figures',len(check.inline_shapes))
