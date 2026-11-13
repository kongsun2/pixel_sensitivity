"""Import policies must preserve full-image VOC boxes and difficult objects."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from index_industrial import read_neu_annotation

def test_neu_voc_one_based_inclusive_and_difficult_retained(tmp_path):
 p=tmp_path/'a.xml'
 p.write_text('<annotation><object><name>scratch</name><difficult>1</difficult><bndbox><xmin>1</xmin><ymin>1</ymin><xmax>200</xmax><ymax>200</ymax></bndbox></object></annotation>')
 boxes=read_neu_annotation(p,['scratch'],200,200)
 assert boxes==[{'class_id':0,'bbox_xyxy':[0.,0.,200.,200.],'difficult':True}]

def test_pcb_native_coco_preserves_boundary_and_class(tmp_path):
 import json
 from index_industrial import read_pcb_coco
 p=tmp_path/'test.json';p.write_text(json.dumps({'images':[{'id':4,'file_name':'images/test/a.jpg','width':300,'height':300}],
 'categories':[{'id':0,'name':'stain'}], 'annotations':[{'image_id':4,'category_id':0,'bbox':[136.51305,185.1972,163.48695,103.9476],'iscrowd':0}]}))
 gt=read_pcb_coco(p,['stain'])['a.jpg']
 assert gt[0]['class_id']==0
 import pytest
 assert gt[0]['bbox_xyxy']==pytest.approx([136.51305,185.1972,300.,289.1448])
 assert gt[0]['bbox_xyxy'][2]==300.
