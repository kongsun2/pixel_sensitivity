"""COCO-style AP from cached GT/predictions. No model inference in this file."""
from __future__ import annotations
import contextlib, io
import numpy as np


def coco_ap(records: list[dict], names: list[str]) -> dict:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    if not sum(len(r['gt']) for r in records):
        return {'coco_map50':None,'coco_map50_95':None,'coco_max_dets':100}
    images=[];anns=[];dets=[];aid=1
    for iid,r in enumerate(records,1):
        images.append({'id':iid,'height':r['height'],'width':r['width'],'file_name':r['image_id']})
        for g in r['gt']:
            x0,y0,x1,y1=g['bbox_xyxy']
            anns.append({'id':aid,'image_id':iid,'category_id':g['class_id']+1,
                'bbox':[x0,y0,x1-x0,y1-y0],'area':(x1-x0)*(y1-y0),'iscrowd':0})
            aid+=1
        for p in r['pred']:
            x0,y0,x1,y1=p['bbox_xyxy']
            dets.append({'image_id':iid,'category_id':p['class_id']+1,
                'bbox':[x0,y0,max(x1-x0,0),max(y1-y0,0)],'score':p['score']})
    with contextlib.redirect_stdout(io.StringIO()):
        gt=COCO();gt.dataset={'images':images,'annotations':anns,
            'categories':[{'id':i+1,'name':n} for i,n in enumerate(names)],'info':{}}
        gt.createIndex()
        if dets:
            dt=gt.loadRes(dets)
        else:
            dt=COCO();dt.dataset={'images':images,'annotations':[],
                'categories':gt.dataset['categories'],'info':{}};dt.createIndex()
        ev=COCOeval(gt,dt,'bbox')
        ev.params.imgIds=[x['id'] for x in images]
        ev.params.maxDets=[1,10,100]
        ev.evaluate();ev.accumulate();ev.summarize()
    return {'coco_map50':float(ev.stats[1]),'coco_map50_95':float(ev.stats[0]),'coco_max_dets':100}
