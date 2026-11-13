"""Example independent YOLOv8 training; settings are proposed, not an optimized result."""
from __future__ import annotations
import argparse
from pathlib import Path
from ultralytics import YOLO


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',required=True);p.add_argument('--name',required=True)
    p.add_argument('--weights',default='yolov8n.pt')
    p.add_argument('--seed',type=int,default=11)
    p.add_argument('--device',default='0');p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--batch',type=int,default=16)
    a=p.parse_args()
    # Every dataset/seed begins from the same original pretrained checkpoint, not another dataset's best.pt.
    model=YOLO(a.weights)
    model.train(data=str(Path(a.data).resolve()), imgsz=640,epochs=a.epochs,patience=20,
        batch=a.batch,device=a.device,seed=a.seed,deterministic=True,optimizer='AdamW',
        lr0=.001,lrf=.01,workers=4,amp=True,cache=False,
        # Conservative fixed recipe, no synthetic pixel faults in training.
        mosaic=0.,mixup=0.,copy_paste=0.,degrees=0.,translate=0.,scale=0.,
        shear=0.,perspective=0.,flipud=0.,fliplr=0.,hsv_h=0.,hsv_s=0.,hsv_v=0.,
        project='runs/train',name=a.name,exist_ok=False)

if __name__=='__main__':main()
