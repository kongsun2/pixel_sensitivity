"""Group PCB-IND v4 positive images by numeric filename prefix across both sides.

The numeric prefix is a conservative proxy, not a verified physical-board identifier.
Only membership and bootstrap group IDs change; images and annotations are preserved.
"""
from __future__ import annotations
import argparse,copy,csv,json,subprocess,sys
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from core import digest_file,dump_json,load_jsonl


def assign_numeric_group_split(rows,seed=101):
 out=copy.deepcopy(rows)
 for r in out:
  group=r['image_id'].split('_',1)[0]
  if not group.isdigit():raise ValueError(f'Non-numeric PCB group proxy: {r["image_id"]}')
  r['group_id']=group
 groups=sorted({r['group_id'] for r in out})
 if len(groups)<10:raise ValueError('At least ten groups required')
 order=np.random.default_rng(seed).permutation(groups).tolist();n1=int(len(order)*.70);n2=int(len(order)*.85)
 mapping={g:'train' if i<n1 else 'val' if i<n2 else 'test' for i,g in enumerate(order)}
 for r in out:r['split']=mapping[r['group_id']]
 return out


def group_overlap(rows):
 g=defaultdict(set)
 for r in rows:g[r['image_id'].split('_',1)[0]].add(r['split'])
 return {'n_numeric_groups':len(g),'n_groups_crossing_splits':sum(len(v)>1 for v in g.values()),
  'n_groups_in_train_and_test':sum({'train','test'}<=v for v in g.values()),
  'cross_split_ids':[key for key,v in sorted(g.items()) if len(v)>1]}


def describe(records,names,scheme):
 rows=[]
 for split in ['train','val','test']:
  subset=[r for r in records if r['split']==split];boxes=[(r,g) for r in subset for g in r['gt']]
  classes=Counter(g['class_id'] for _,g in boxes);areas=[];centers=[]
  for r,g in boxes:
   x0,y0,x1,y1=g['bbox_xyxy'];areas.append((x1-x0)*(y1-y0)/(r['width']*r['height']))
   centers.append(((x0+x1)/(2*r['width']),(y0+y1)/(2*r['height'])))
  central=sum(.375<=x<.625 and .375<=y<.625 for x,y in centers)
  row={'scheme':scheme,'split':split,'n_images':len(subset),'n_numeric_groups':len({r['image_id'].split('_')[0] for r in subset}),
   'n_gt':len(boxes),'median_gt_area_fraction':float(np.median(areas)),
   'gt_center_in_central25pct_by25pct_fraction':central/len(boxes)}
  row.update({f'class_{i}_{name}':classes[i] for i,name in enumerate(names)});rows.append(row)
 return rows


def write_csv(path,rows):
 with Path(path).open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=101);args=p.parse_args()
 inputs=ROOT/'inputs';report=ROOT/'reports/research_validation_2026-11-11/pcb_split';report.mkdir(parents=True,exist_ok=True)
 old_index=inputs/'pcb_ind_index.csv';classes=inputs/'pcb_ind_classes.json';new_index=inputs/'pcb_ind_board_index.csv';new_classes=inputs/'pcb_ind_board_classes.json'
 with old_index.open(encoding='utf-8-sig',newline='') as f:old_rows=list(csv.DictReader(f))
 if len(old_rows)!=4662:raise ValueError('Expected previously audited4662positive PCB images')
 names=json.loads(classes.read_text());rows=assign_numeric_group_split(old_rows,args.seed)
 if {r['image_id'] for r in old_rows}!={r['image_id'] for r in rows}:raise ValueError('Image census changed')
 before=group_overlap(old_rows);after=group_overlap(rows)
 if after['n_groups_crossing_splits'] or after['n_groups_in_train_and_test']:raise ValueError('Numeric group leakage')
 counts={s:np.zeros(len(names),int) for s in ['train','val','test']}
 for r in rows:
  for line in Path(r['label_path']).read_text().splitlines():
   if line.strip():counts[r['split']][int(line.split()[0])]+=1
 if any((v==0).any() for v in counts.values()):raise ValueError('Initial deterministic group split lacks a class; revise stratification policy explicitly')
 write_csv(new_index,rows);new_classes.write_bytes(classes.read_bytes())
 dest=ROOT/'data/canonical/pcb_ind_board'
 if (dest/'audit.json').exists():
  existing=json.loads((dest/'audit.json').read_text())
  if existing['index_sha256']!=digest_file(new_index):raise ValueError('Canonical output is from another index')
 elif dest.exists() and any(dest.iterdir()):raise ValueError('Partial canonical output exists; inspect before retrying')
 else:
  subprocess.run([str(ROOT/'.venv/bin/python'),str(ROOT/'src/prepare.py'),'--index',str(new_index),'--classes',str(new_classes),'--dataset','pcb_ind_board','--format','yolo','--out',str(dest)],check=True)
 old_manifest={r['image_id']:r for r in load_jsonl(ROOT/'data/canonical/pcb_ind/manifest.jsonl')}
 new_manifest=load_jsonl(dest/'manifest.jsonl')
 for r in new_manifest:
  old=old_manifest[r['image_id']]
  if old['gt']!=r['gt'] or old['image_sha256']!=r['image_sha256'] or old['source_sha256']!=r['source_sha256']:
   raise ValueError(f'Pixels or annotation content changed: {r["image_id"]}')
 all_groups={s:{r['group_id'] for r in new_manifest if r['split']==s} for s in ['train','val','test']}
 if any(all_groups[a]&all_groups[b] for a,b in [('train','val'),('train','test'),('val','test')]):raise ValueError('Canonical group overlap')
 comparisons=describe(list(old_manifest.values()),names,'original_official_split')+describe(new_manifest,names,'numeric_group_split_seed101')
 write_csv(report/'split_distribution_comparison.csv',comparisons)
 old_by_id={r['image_id']:r for r in old_rows}
 write_csv(report/'image_split_membership.csv',[{'image_id':r['image_id'],'numeric_group':r['group_id'],'old_split':old_by_id[r['image_id']]['split'],'new_split':r['split']} for r in rows])
 transition=Counter((old_by_id[r['image_id']]['split'],r['split']) for r in rows)
 write_csv(report/'split_transition_counts.csv',[{'old_split':a,'new_split':b,'n_images':n} for (a,b),n in sorted(transition.items())])
 audit={'dataset':'pcb_ind_board','source_dataset':'pcb_ind','source_index_sha256':digest_file(old_index),'index_sha256':digest_file(new_index),
  'source_manifest_sha256':digest_file(ROOT/'data/canonical/pcb_ind/manifest.jsonl'),'canonical_manifest_sha256':digest_file(dest/'manifest.jsonl'),
  'classes_sha256':digest_file(new_classes),'source_script_sha256':digest_file(__file__),'seed':args.seed,
  'policy':'Sorted numeric filename groups shuffled with numpy.default_rng(seed); floor70%train/floor85%cumulativeval; no class balancing or performance tuning needed',
  'n_images':len(rows),'n_groups':after['n_numeric_groups'],'image_counts':dict(Counter(r['split'] for r in rows)),
  'group_counts':{s:len(v) for s,v in all_groups.items()},'class_counts':{s:v.tolist() for s,v in counts.items()},
  'old_overlap':before,'new_overlap':after,'all_original_positive_images_preserved':True,
  'decoded_pixel_duplicates':'none; existing prepare.py hashes decoded RGB and rejects any duplicate',
  'image_png_and_source_hashes_identical_to_prior_canonical':True,'gt_dicts_identical_to_prior_canonical':True,
  'exclusions_preserved':'Same4662audited positive images;127unverified empty-label source images remain excluded; no additional exclusions',
  'caveat':'Numeric filename IDs are only a conservative grouping proxy, not verified physical board IDs. Test cohort changes; old/new model performance cannot be interpreted as a causal effect of leakage removal alone.'}
 dump_json(inputs/'pcb_ind_board_source_audit.json',audit);dump_json(report/'split_audit.json',audit)
 summary='| 분할 | 기존 영상 / GT | 새 영상 / GT | 새 숫자 그룹 수 |\n|---|---:|---:|---:|\n'
 for s in ['train','val','test']:
  o=next(r for r in comparisons if r['scheme']=='original_official_split' and r['split']==s);n=next(r for r in comparisons if r['scheme']=='numeric_group_split_seed101' and r['split']==s)
  summary+=f"| {s} | {o['n_images']} / {o['n_gt']} | {n['n_images']} / {n['n_gt']} | {n['n_numeric_groups']} |\n"
 class_table='| 클래스 | 새 train | 새 val | 새 test |\n|---|---:|---:|---:|\n'
 for i,name in enumerate(names):class_table+=f"| {name} | {counts['train'][i]} | {counts['val'][i]} | {counts['test'][i]} |\n"
 report_text=f'''# PCB 숫자 ID 단위 재분할

기존 공식 분할은 `5810_t`와 `5810_b`를 다른 면 그룹으로 처리한다. 숫자 접두어만 묶었을 때 {before['n_numeric_groups']}개 중 {before['n_groups_crossing_splits']}개가 여러 분할에 걸치고, {before['n_groups_in_train_and_test']}개가 학습과 시험에 함께 나타났다.

숫자 접두어를 묶어 seed={args.seed}, 그룹 수 기준70/15/15로 분리했다. 정렬한 그룹에 NumPy default_rng 순열을 적용하며 클래스별 사후 조정이나 모델 성능 기반 선택은 하지 않았다. 첫 분할에서8개 클래스가 모든 분할에 존재했다.

{summary}

{class_table}

새 분할의 숫자 ID 교집합은 train–val, train–test, val–test 모두0이다. 기존 양성4662장 전체를 보존했다. 원본과 canonical PNG의 해시 및 GT 사전이 이전 canonical 자료와 모두 일치한다. 기존127개 미확인 빈 라벨 제외 정책도 유지했다. 기존 prepare.py가 decoded RGB중복 검사를 통과했다.

**이 숫자 ID는 실제 물리 보드 ID임이 확인된 값이 아니라 보수적인 그룹 대리값이다.** 새 시험 코호트의 영상 수·클래스 분포가 달라지므로 기존/신규 모델의 성능 차이를 누수 제거만의 인과적 효과라고 주장할 수 없다. 중앙 ROI 특성과 양성 자료만 사용한 기존 범위도 유지된다.

- 학습 설정 파일: `{dest/'dataset.yaml'}`
- [분할·클래스·GT 크기/중앙 집중 비교](split_distribution_comparison.csv)
- [영상별 이전/신규 분할](image_split_membership.csv)
- [분할 전이 수](split_transition_counts.csv)
- [검증 JSON](split_audit.json)
'''
 (report/'REPORT_KO.md').write_text(report_text)
 print(json.dumps({k:audit[k] for k in ['n_images','n_groups','image_counts','group_counts','class_counts']},indent=2),flush=True)
 print('READY',dest/'dataset.yaml',flush=True)

if __name__=='__main__':main()
