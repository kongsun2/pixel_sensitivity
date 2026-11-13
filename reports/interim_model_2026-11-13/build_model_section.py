"""Build a standalone, evidence-based interim report model section."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

OUT = Path(__file__).resolve().parent
DEST = Path('/home/hyochang/interim_model_yolov8n.docx')

fig, ax = plt.subplots(figsize=(11, 3.1))
ax.set(xlim=(0, 11), ylim=(0, 3.1))
ax.axis('off')
boxes = [
    (0.05, 1.15, 1.65, 'Input', '3 × 640 × 640'),
    (2.05, 0.75, 2.4, 'Backbone', 'Conv + C2f + SPPF\nMulti-scale features'),
    (4.85, 0.75, 2.4, 'Neck', 'Top-down + bottom-up\nUpsample / Concat / C2f'),
    (7.65, 0.75, 3.25, 'Detection head', 'P3: 80 × 80 × 64\nP4: 40 × 40 × 128\nP5: 20 × 20 × 256'),
]
for x, y, w, title, body in boxes:
    h = 1.05 if title == 'Input' else 1.85
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.025',
                              facecolor='#EDF3FA', edgecolor='#355B80', linewidth=1.3))
    ax.text(x+w/2, y+h-.32, title, ha='center', va='center', fontsize=12, weight='bold')
    ax.text(x+w/2, y+h/2-.15, body, ha='center', va='center', fontsize=10.2, linespacing=1.5)
for x0, x1 in [(1.73, 2.0), (4.49, 4.8), (7.29, 7.6)]:
    ax.annotate('', xy=(x1, 1.65), xytext=(x0, 1.65),
                arrowprops={'arrowstyle': '->', 'color': '#355B80', 'lw': 1.7})
ax.text(9.27, .32, 'Separate box / class branches → Decode → NMS',
        ha='center', fontsize=9.1)
fig.savefig(OUT/'yolov8n_structure.png', dpi=220, bbox_inches='tight', facecolor='white')
plt.close(fig)

doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Inches(8.27), Inches(11.69)
sec.top_margin = sec.bottom_margin = Inches(.8)
sec.left_margin = sec.right_margin = Inches(.85)
for name in ['Normal', 'Title', 'Heading 1', 'Heading 2', 'Caption']:
    style = doc.styles[name]
    style.font.name = '맑은 고딕'
    style._element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'), '맑은 고딕')
doc.styles['Normal'].font.size = Pt(10.5)
doc.styles['Normal'].paragraph_format.line_spacing = 1.2
doc.styles['Normal'].paragraph_format.space_after = Pt(7)
for name in ['Heading 1', 'Heading 2']:
    doc.styles[name].font.color.rgb = RGBColor.from_string('244A70')
doc.core_properties.title = '중간 결과 보고: 모델 및 방법론 — YOLOv8n'
doc.core_properties.subject = '효창 담당 모델 설명, 실제 학습 및 화소 손상 평가 방법'

def p(text):
    doc.add_paragraph(text)

def h(text):
    doc.add_heading(text, level=1)

def table(headers, rows):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = 'Light Shading Accent 1'
    for c, v in zip(t.rows[0].cells, headers):
        c.text = str(v)
    for row in rows:
        for c, v in zip(t.add_row().cells, row):
            c.text = str(v)
    for row in t.rows:
        for cell in row.cells:
            for para in cell.paragraphs:
                for run in para.runs:
                    run.font.size = Pt(9)
    doc.add_paragraph()

doc.add_heading('[효창] 모델 및 방법론: YOLOv8n', 0)
p('본 연구는 제조 결함 탐지 모델이 입력 영상의 화소 손상에 얼마나 민감하게 반응하는지 분석한다. 이를 위해 COCO 사전학습 YOLOv8n을 각 제조 데이터셋에 독립적으로 추가 학습하고, 학습이 완료된 동일 모델에 원본 평가 영상과 손상 평가 영상을 입력하여 탐지 성능의 변화를 비교하였다. 모델 구조를 새로 제안하거나 화소 손상에 대한 보정 모듈을 추가하는 대신, 표준 모델을 공통 기준으로 사용하였다.')

h('1. 모델 선정 및 연구에서의 역할')
p('YOLOv8n은 YOLOv8 계열의 nano 규모 객체 탐지 모델이다. 영상으로부터 결함의 위치를 나타내는 바운딩 박스, 결함 클래스 및 신뢰도를 출력하는 단일 단계 탐지기이다. 본 실험은 위치와 손상 패턴을 바꾸어 추론을 반복해야 하므로, 계산 부담이 비교적 작은 n 모델을 기준 모델로 선정하였다. 이는 제조 결함 탐지에서 가장 높은 정확도를 보이는 모델이라는 의미가 아니라, 반복 평가를 수행할 수 있는 경량 기준 모델을 선택했다는 의미이다. 사용한 가중치는 객체 탐지용 yolov8n.pt이며, 구현 라이브러리는 Ultralytics 8.3.221이다. [1]')
p('세 데이터셋은 각각 별도의 모델로 학습하였다. 모든 학습은 동일한 원본 COCO 사전학습 가중치에서 시작하였으며, 한 제조 데이터셋에서 학습한 모델을 다음 데이터셋으로 이어서 학습하지 않았다. 클래스 수에 맞게 탐지 헤드의 분류 출력 수를 조정하였고 Backbone과 Neck의 기본 구조는 유지하였다.')
table(['데이터셋', '출력 클래스 수', '학습 체크포인트 파라미터 수'], [
    ['GC10-DET', 10, '3,012,798'],
    ['NEU-DET', 6, '3,012,018'],
    ['PCB(현행 평가 구성)', 8, '3,012,408'],
])
p('표의 파라미터 수는 seed 11의 best.pt를 로드한 뒤 Conv–BatchNorm 융합 전 모델에서 직접 집계하였다. 공식 문서의 COCO 80클래스 YOLOv8n은 약 320만 개 파라미터로 소개되지만, 본 실험 모델은 클래스 출력 수가 달라 약 301만 개이다. 추론 시 계층 융합 여부에 따라서도 집계값은 달라질 수 있다.')

h('2. 전체 구조: Backbone–Neck–Head')
p('YOLOv8n의 연산 흐름은 입력 영상 전처리, Backbone 특징 추출, Neck 다중 크기 특징 결합, Detection Head의 위치·클래스 예측, 후처리 순서이다. n 규모에서는 기본 구조의 깊이 배율 0.33과 채널 폭 배율 0.25를 적용하여 계층 반복 수와 채널 수를 줄인다. [2]')
doc.add_picture(str(OUT/'yolov8n_structure.png'), width=Inches(6.5))
doc.add_paragraph('그림 1. 실제 사용 모델에 대응하는 YOLOv8n의 개략 구조. 텐서 크기는 채널×높이×너비 또는 그림 내 높이×너비×채널 표기에 따른다.', style='Caption')

doc.add_heading('2.1 Backbone: 영상 특징 추출', level=2)
p('Backbone은 Conv, C2f, SPPF 모듈로 구성된다. Conv 블록은 합성곱, Batch Normalization 및 SiLU 활성화 함수로 이루어지며, stride 2 합성곱으로 영상의 공간 해상도를 줄이면서 특징을 추출한다. 초기 계층은 경계와 질감 등 국소 정보를, 깊은 계층은 더 넓은 영역의 정보를 표현한다.')
p('C2f는 입력을 1×1 합성곱으로 변환한 뒤 채널 방향으로 두 갈래로 나누고, 한 갈래에 Bottleneck을 순차적으로 적용한다. 이후 두 초기 갈래와 각 Bottleneck의 중간 출력을 이어 붙인 뒤 다시 1×1 합성곱으로 결합한다. 여러 단계의 특징을 함께 전달하는 구조이며, 본 모델의 Backbone C2f에는 잔차 연결이 사용된다. [3]')
p('Backbone의 마지막 SPPF(Spatial Pyramid Pooling–Fast)는 채널을 줄인 특징에 5×5 최대 풀링을 세 번 연속 적용하고, 원래 특징과 각 풀링 결과를 연결하여 결합한다. 이는 공간 해상도를 유지하면서 서로 다른 범위의 주변 정보를 함께 반영하는 방식이다. [3]')
table(['Backbone 단계', '공간 크기', '출력 채널', '주요 구성'], [
    ['입력', '640×640', 3, '전처리된 영상'],
    ['초기 축소', '320×320', 16, 'Conv'],
    ['P2', '160×160', 32, 'Conv + C2f'],
    ['P3', '80×80', 64, 'Conv + C2f'],
    ['P4', '40×40', 128, 'Conv + C2f'],
    ['P5', '20×20', 256, 'Conv + C2f + SPPF'],
])

doc.add_heading('2.2 Neck: 서로 다른 크기의 특징 결합', level=2)
p('Neck은 FPN과 PAN 방식의 흐름을 사용하여 서로 다른 해상도의 특징을 결합한다. 먼저 깊은 계층의 특징을 업샘플링한 뒤 상대적으로 얕은 계층의 특징과 연결하는 하향식 경로를 구성한다. 이어서 stride 2 합성곱을 통해 특징을 다시 축소하고 연결하는 상향식 경로를 구성한다. 각 결합 지점의 C2f가 연결된 특징을 처리한다. 최종적으로 80×80, 40×40, 20×20의 세 특징 맵을 탐지 헤드에 전달하며, 각각의 출력 채널은 64, 128, 256이다. [2]')
p('이러한 다중 해상도 구조는 서로 다른 크기의 결함을 예측하는 데 사용된다. 다만 작은 화소 변화가 특징 추출과 결합을 거쳐 실제 탐지 결과에 미치는 영향은 데이터와 모델에 따라 달라지므로, 본 연구에서 실험으로 확인하는 대상이다.')

doc.add_heading('2.3 Detection Head: 위치와 클래스 예측', level=2)
p('탐지 헤드는 위치 회귀와 클래스 분류를 별도 분기로 계산하는 decoupled head이다. 실제 체크포인트에서 각 분기는 두 개의 3×3 합성곱 블록과 마지막 1×1 출력 합성곱으로 구성된다. 분류 분기는 각 결함 클래스의 점수를 출력하고, 회귀 분기는 바운딩 박스의 경계를 예측한다. 별도의 objectness 출력 분기를 사용하지 않는다. [4]')
p('YOLOv8n은 사전에 폭과 높이가 정해진 앵커 박스를 사용하는 대신, 특징 맵의 격자 기준점에서 박스 경계까지의 거리를 예측하는 anchor-free 방식을 사용한다. 따라서 앵커 박스가 없다는 말이 격자 기준점 자체가 없다는 의미는 아니다. 회귀 분기는 기준점에서 좌·상·우·하 경계까지의 거리를 각각 16개 구간의 분포로 출력한다(reg_max=16). 이 분포의 기댓값으로 거리를 복원한 뒤 박스 좌표로 변환한다. [4]')
table(['탐지 특징 맵', 'Stride', '특징 크기', '후보 기준점 수'], [
    ['P3', 8, '80×80×64', '6,400'],
    ['P4', 16, '40×40×128', '1,600'],
    ['P5', 32, '20×20×256', '400'],
    ['합계', '—', '640×640 입력 기준', '8,400'],
])
p('각 기준점의 학습 출력은 박스 분포를 위한 64개 채널(4방향×16구간)과 클래스 수 nc에 해당하는 채널로 이루어진다. 8,400은 후처리 이전의 후보 기준점 수이며 최종 검출 개수가 아니다. 최종 결과는 박스 복원, 신뢰도 필터링 및 NMS를 거쳐 얻는다.')

h('3. 학습 목적 함수')
p('학습에는 박스 위치 손실, 클래스 분류 손실, Distribution Focal Loss(DFL)를 함께 사용한다. 박스 위치 손실은 CIoU 기반으로 예측 박스와 정답 박스의 겹침, 중심 거리 및 종횡비 관계를 반영한다. 분류 손실은 BCEWithLogitsLoss를 사용하며, DFL은 각 경계 거리의 분포 예측을 학습한다. 실제 학습 설정의 손실 가중치는 다음과 같다. [5]')
p('L = 7.5 L_box + 0.5 L_cls + 1.5 L_DFL')
p('각 항은 구현 내부의 정규화와 표본 가중 처리를 포함한 손실이다. 학습 정답의 할당에는 분류 점수와 박스 일치도를 고려하는 Task-Aligned Assigner가 사용된다. 본 연구는 이러한 YOLOv8 기본 학습 절차를 사용하며, 화소 손상에 특화된 손실 항을 추가하지 않았다.')

h('4. 제조 데이터셋 추가 학습 방법')
p('COCO 사전학습은 일반 영상에서 학습된 초기 특징을 제공하며, 제조 데이터셋의 바운딩 박스와 클래스 정답을 사용하여 전체 탐지 모델을 추가 학습하였다. 학습 영상에는 인위적인 핫픽셀이나 다른 화소 손상을 주입하지 않았다. 여기서 원본 학습 영상은 실제 제조 결함을 포함할 수 있으며, “인위적 화소 손상이 없는 영상”을 의미한다.')
table(['항목', '실제 사용 설정'], [
    ['초기 가중치', 'COCO 사전학습 yolov8n.pt'],
    ['입력 크기 / 배치 크기', '640×640 / 16'],
    ['최대 학습 길이', '100 epochs'],
    ['조기 종료', 'patience=20; 실제 종료 시점은 학습별로 다름'],
    ['Optimizer', 'AdamW'],
    ['학습률', 'lr0=0.001, lrf=0.01'],
    ['학습 난수 시드', '11, 22, 33'],
    ['반복 구성', '3개 데이터셋 × 3개 시드 = 9개 모델'],
    ['정밀도 / 재현성 설정', '학습 AMP=True, deterministic=True'],
    ['선택 가중치', '검증 성능에 따라 저장된 best.pt'],
    ['영상 증강', 'Mosaic, MixUp, Copy-Paste, 회전·이동·확대축소·뒤집기·HSV 변형 등을 0으로 설정'],
])
p('같은 데이터셋에서 시드를 바꾼 세 모델을 구성하여 학습 과정의 변동성을 확인하였다. 이 학습 시드 반복과 손상 화소의 위치를 바꾸는 패턴 반복은 서로 다른 실험 요소이다. 모든 손상 조건을 세 시드에서 평가한 것은 아니며, 확대 실험의 시드 적용 범위는 결과 표에 따로 표시한다. 학습 설정의 근거는 프로젝트의 src/train.py와 각 실행의 args.yaml이다.')

h('5. 화소 손상 실험과 모델의 연결')
p('학습이 끝난 모델의 가중치를 고정하고, 동일 평가 이미지의 원본과 화소 손상을 주입한 버전을 각각 추론한다. 손상은 원본 해상도의 영상에 먼저 적용하고, 이후 종횡비를 유지하는 리사이즈와 패딩(letterbox)을 거쳐 640×640 모델 입력으로 변환한다. 단일 채널 영상은 3채널로 복제하여 입력하며, 영상 값은 0–1 범위로 정규화한다. 따라서 원본 해상도에서 같은 개수의 화소를 손상하더라도 리사이즈 이후 모델에 전달되는 영향은 데이터셋별로 달라질 수 있다.')
p('위치 민감도 평가는 영상을 8×8의 64개 구역으로 나누고, 한 번에 한 구역 안에 손상 패턴을 주입하는 방식이다. 각 구역에서 서로 다른 10개 패턴을 적용하여 손상 조건 하나당 64×10=640개 위치·패턴 조합을 평가한다. 이 8×8은 입력 손상 위치를 정하기 위한 실험 격자이며, 모델 내부의 80×80·40×40·20×20 특징 맵과는 별개이다.')
p('평가의 기본 흐름은 “원본 평가 영상 → 원본/손상 버전 생성 → 동일 전처리 → 동일 YOLOv8n 추론 → 동일 후처리 및 정답 비교 → 성능 변화와 위치 지도 계산”이다. 핫픽셀을 제품 결함의 새 정답으로 추가하지 않고, 원래 제품 결함의 정답을 유지하여 기존 결함의 누락과 오검출 변화를 측정한다.')
table(['평가 항목', '설정 / 의미'], [
    ['추론 정밀도 / 배치', 'FP32 / 16'],
    ['예측 보관 신뢰도 하한', '0.001; AP 등 지표 계산용 후보 보관'],
    ['운영 신뢰도 기준', '0.25; 고정 임계값 Precision·Recall·F1 계산'],
    ['NMS IoU', '0.70; 같은 클래스의 중복 예측 억제'],
    ['정답 매칭 IoU', '0.50; 고정 임계값 지표에서 정답 박스와의 일치 판단'],
    ['최대 예측 수', '이미지당 300개; COCO AP 평가에서는 별도의 maxDets=100 적용'],
])
p('원본 성능 M_clean과 손상 성능 M_fault의 차이는 ΔM=M_fault−M_clean으로 정의할 수 있다. 이때 음수는 성능 저하를 뜻한다. 결과를 저하량 D=M_clean−M_fault로 표시할 경우에는 양수가 저하를 뜻하므로 표와 지도에 부호 정의를 명시한다. 위치 지도는 해당 구역의 손상 패턴들에서 관측된 성능 변화를 집계한 결과이며, 모델 내부의 주의도나 Grad-CAM을 직접 시각화한 것은 아니다.')

h('참고 자료 및 실행 근거')
refs = [
    '[1] Ultralytics, YOLOv8 공식 모델 문서. https://docs.ultralytics.com/models/yolov8/',
    '[2] Ultralytics v8.3.221, yolov8.yaml. https://github.com/ultralytics/ultralytics/blob/v8.3.221/ultralytics/cfg/models/v8/yolov8.yaml',
    '[3] Ultralytics v8.3.221, block.py (C2f, SPPF). https://github.com/ultralytics/ultralytics/blob/v8.3.221/ultralytics/nn/modules/block.py',
    '[4] Ultralytics v8.3.221, head.py (Detect). https://github.com/ultralytics/ultralytics/blob/v8.3.221/ultralytics/nn/modules/head.py',
    '[5] Ultralytics v8.3.221, loss.py. https://github.com/ultralytics/ultralytics/blob/v8.3.221/ultralytics/utils/loss.py',
    '실제 학습: /home/hyochang/pixel_sensitivity/src/train.py 및 runs/train/*/args.yaml',
    '실제 평가: /home/hyochang/pixel_sensitivity/src/run_compiled.py 및 configs/research_validation/bright_k25.yaml',
    '구조 검증: GC10·NEU·현행 PCB seed 11 best.pt의 모델 계층·클래스 수·파라미터 수 직접 조회. 표준 소스 설명과 실제 로드된 체크포인트 구조를 대조함.',
]
for text in refs:
    p(text)

footer = sec.footer.paragraphs[0]
footer.alignment = 2
field = OxmlElement('w:fldSimple')
field.set(qn('w:instr'), 'PAGE')
footer._p.append(field)
doc.save(DEST)
print(DEST)
print(f'paragraphs={len(doc.paragraphs)}, tables={len(doc.tables)}, inline_shapes={len(doc.inline_shapes)}')
