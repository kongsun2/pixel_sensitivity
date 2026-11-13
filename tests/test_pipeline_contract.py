"""Synthetic software tests, not experimental evidence about YOLO or manufacturing."""
import csv,json,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import cv2
import pytest
from core import digest_file,dump_json
ROOT=Path(__file__).resolve().parents[1]


def test_prepare_on_tiny_audited_index(tmp_path):
    src=tmp_path/'source';src.mkdir();rows=[]
    (src/'classes.json').write_text('["defect"]')
    for i in range(12):
        img=np.full((20,24,3),20+i,np.uint8);cv2.imwrite(str(src/f'{i}.png'),img)
        (src/f'{i}.txt').write_text('0 .5 .5 .5 .5\n')
        rows.append({'image_id':f'x{i}','image_path':f'{i}.png','label_path':f'{i}.txt',
            'split':'train' if i<8 else 'val' if i<10 else 'test','group_id':f'group{i}'})
    with (src/'index.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    out=tmp_path/'canonical'
    subprocess.run([sys.executable,str(ROOT/'src/prepare.py'),'--index',str(src/'index.csv'),
        '--classes',str(src/'classes.json'),'--dataset','toy','--format','yolo','--out',str(out)],check=True,capture_output=True)
    manifest=[json.loads(s) for s in (out/'manifest.jsonl').read_text().splitlines()]
    assert len(manifest)==12
    assert all(r['gt'][0]['bbox_xyxy']==[6.,5.,18.,15.] for r in manifest)
    assert (out/'dataset.yaml').is_file()


def test_analyze_contract(tmp_path):
    run=tmp_path/'run';(run/'predictions').mkdir(parents=True)
    classes=tmp_path/'classes.json';classes.write_text('["defect"]')
    cfg={'grid':{'rows':2,'cols':2},'repeats':2,
        'predict':{'conf_floor':.001},'evaluation':{'conf_operating':.25,'match_iou':.5},
        'conditions':[{'id':'unit_test_fault'}]}
    dump_json(run/'run_meta.json',{'run_hash':'software-test','classes_sha256':digest_file(classes),'config':cfg})
    gt={'class_id':0,'bbox_xyxy':[1.,1.,8.,8.]}
    def record(i,score,fault=None):
        return {'image_id':str(i),'group_id':str(i),'height':20,'width':20,'gt':[gt],
            'pred':[dict(gt,score=score)],'fault':fault}
    def write(name,records):
        path=run/'predictions'/f'{name}.jsonl'
        path.write_text('\n'.join(json.dumps(r) for r in records)+'\n')
        dump_json(path.with_suffix('.done.json'),{'run_hash':'software-test','sha256':digest_file(path)})
    write('clean',[record(i,.9) for i in range(4)])
    fm={'count_changed':1,'count_commanded':1,'l1_rgb_sum':60}
    for rep in range(2):
        for b in range(4):
            write(f'unit_test_fault__r{rep:03d}__g{b:03d}',
                [record(i,.1 if b==1 and i<2 else .9,fm) for i in range(4)])
    subprocess.run([sys.executable,str(ROOT/'src/analyze.py'),'--run',str(run),'--classes',str(classes),
        '--bootstrap','50'],check=True,capture_output=True)
    with (run/'analysis/grid_summary.csv').open(encoding='utf-8-sig') as f:s=list(csv.DictReader(f))
    assert float(s[1]['sensitivity_recall'])==.5
    assert float(s[0]['sensitivity_recall'])==0.
    assert len(s)==4


def test_coco_perfect_and_empty_if_installed():
    pytest.importorskip('pycocotools')
    from coco_metrics import coco_ap
    gt={'class_id':0,'bbox_xyxy':[1.,1.,8.,8.]}
    r={'image_id':'x','height':20,'width':20,'gt':[gt],'pred':[dict(gt,score=.9)]}
    m=coco_ap([r],['defect']);assert m['coco_map50_95']==pytest.approx(1.)
    r['pred']=[];m=coco_ap([r],['defect']);assert m['coco_map50_95']==0.
