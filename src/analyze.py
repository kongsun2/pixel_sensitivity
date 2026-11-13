"""Aggregate cached inference. Saves raw-unit maps, CSV, and per-image paired records.
CI is pointwise; do not use it to claim 16 independent significant discoveries.
"""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np
from core import cell_bounds,digest_file,dump_json,load_jsonl,match_predictions,operational_metrics


def paired_ci(clean_tp, fault_tp, n_gt, groups, n_boot=2000, seed=991, resample_patterns=True):
    """fault_tp: (pattern_repeats, images); resample image groups and optional global patterns.
    The same bootstrap draw is reused for clean and fault. Conditional on one trained model.
    """
    clean_tp=np.asarray(clean_tp,float);fault_tp=np.asarray(fault_tp,float);n_gt=np.asarray(n_gt,float)
    unique=sorted(set(groups));gid=np.array([unique.index(g) for g in groups])
    if len(unique)<2 or n_gt.sum()==0:return [None,None]
    gt=np.array([n_gt[gid==i].sum() for i in range(len(unique))])
    cp=np.array([clean_tp[gid==i].sum() for i in range(len(unique))])
    fp=np.stack([np.array([f[gid==i].sum() for i in range(len(unique))]) for f in fault_tp])
    rng=np.random.default_rng(seed);vals=[]
    for _ in range(n_boot):
        ix=rng.integers(0,len(unique),len(unique))
        ri=rng.integers(0,len(fp),len(fp)) if resample_patterns else np.arange(len(fp))
        den=gt[ix].sum()
        if den>0:vals.append((cp[ix].sum()-fp[ri][:,ix].mean(axis=0).sum())/den)
    if not vals:return [None,None]
    return [float(v) for v in np.quantile(vals,[.025,.975])]


def write_csv(path,rows):
    if not rows:return
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--classes',required=True)
    p.add_argument('--ap',action='store_true',help='Compute COCO AP for each complete condition')
    p.add_argument('--plots',action='store_true');p.add_argument('--bootstrap',type=int,default=2000)
    p.add_argument('--bootstrap-seed',type=int,default=991);a=p.parse_args()
    root=Path(a.run);meta=json.loads((root/'run_meta.json').read_text());cfg=meta['config']
    if digest_file(a.classes)!=meta['classes_sha256']:raise ValueError('Class map differs from inference run')
    names=json.loads(Path(a.classes).read_text());out=root/'analysis';out.mkdir(exist_ok=True)
    def read_shard(path):
        done=path.with_suffix('.done.json')
        if not done.exists():raise FileNotFoundError(f'Missing completion marker: {done}')
        d=json.loads(done.read_text())
        if d['run_hash']!=meta['run_hash'] or d['sha256']!=digest_file(path):raise ValueError('Stale or corrupt shard')
        return load_jsonl(path)
    clean=read_shard(root/'predictions/clean.jsonl')
    ids=[r['image_id'] for r in clean]
    t=cfg['evaluation']['conf_operating'];mi=cfg['evaluation']['match_iou'];floor=cfg['predict']['conf_floor']
    cm=[match_predictions(r['gt'],r['pred'],t,mi) for r in clean]
    cl=[match_predictions(r['gt'],r['pred'],floor,mi) for r in clean]
    base=operational_metrics(clean,t,mi)
    if a.ap:
        from coco_metrics import coco_ap
        base.update(coco_ap(clean,names))
    dump_json(out/'clean_metrics.json',base)
    gr,gc=cfg['grid']['rows'],cfg['grid']['cols'];B=gr*gc
    ngt=np.array([len(r['gt']) for r in clean]);ctp=np.array([m['tp'] for m in cm])
    groups=[r['group_id'] for r in clean]
    # Occupancy is bounding-box coverage, NOT true defective-pixel coverage.
    oc=np.zeros(B);presence=np.zeros(B)
    for r in clean:
        h,w=r['height'],r['width'];mask=np.zeros((h,w),bool)
        for g in r['gt']:
            x0,y0,x1,y1=g['bbox_xyxy']
            mask[max(0,int(np.floor(y0))):min(h,int(np.ceil(y1))),
                 max(0,int(np.floor(x0))):min(w,int(np.ceil(x1)))]=True
        for b in range(B):
            x0,y0,x1,y1=cell_bounds(h,w,gr,gc,b);section=mask[y0:y1,x0:x1]
            oc[b]+=float(section.mean());presence[b]+=float(section.any())
    occupancy=[{'cell':b,'grid_label':f'G{b+1}','box_area_occupancy':float(oc[b]/len(clean)),
        'image_presence_rate':float(presence[b]/len(clean))} for b in range(B)]
    write_csv(out/'occupancy.csv',occupancy)
    summary=[];per_run=[]
    pair_tmp=out/'paired_images.csv.partial'
    pair_file=pair_tmp.open('w',encoding='utf-8-sig',newline='')
    pair_writer=None
    for cond in cfg['conditions']:
        for b in range(B):
            f_tps=[];sens=[];cd_sum=0.;cd_n=0;soft=[];changed=[];l1=[];aps=[]
            for rep in range(cfg['repeats']):
                path=root/'predictions'/f'{cond["id"]}__r{rep:03d}__g{b:03d}.jsonl'
                fault=read_shard(path)
                if [r['image_id'] for r in fault]!=ids:raise ValueError('Clean/fault image order differs')
                metrics=operational_metrics(fault,t,mi)
                ft=[]
                for i,(cr,fr) in enumerate(zip(clean,fault)):
                    if cr['gt']!=fr['gt']:raise ValueError('GT must not change with synthetic sensor faults')
                    fm=match_predictions(fr['gt'],fr['pred'],t,mi)
                    fl=match_predictions(fr['gt'],fr['pred'],floor,mi);ft.append(fm['tp'])
                    selected=np.array(cm[i]['gt_matched'],bool)
                    delta=np.array(cl[i]['gt_scores'])-np.array(fl['gt_scores'])
                    cd_sum+=float(delta[selected].sum());cd_n+=int(selected.sum())
                    soft.append(fl['soft_fp_sum']-cl[i]['soft_fp_sum'])
                    changed.append(fr['fault']['count_changed']);l1.append(fr['fault']['l1_rgb_sum'])
                    pair_row={'condition_id':cond['id'],'cell':b,'repeat':rep,
                        'image_id':cr['image_id'],'group_id':cr['group_id'],'n_gt':len(cr['gt']),
                        'tp_clean':cm[i]['tp'],'tp_fault':fm['tp'],'fp_clean':cm[i]['fp'],'fp_fault':fm['fp'],
                        'clean_tp_conf_drop_sum':float(delta[selected].sum()),'n_clean_tp':int(selected.sum()),
                        'count_commanded':fr['fault']['count_commanded'],'count_changed':fr['fault']['count_changed'],
                        'l1_rgb_sum':fr['fault']['l1_rgb_sum']}
                    if pair_writer is None:
                        pair_writer=csv.DictWriter(pair_file,fieldnames=list(pair_row));pair_writer.writeheader()
                    pair_writer.writerow(pair_row)
                f_tps.append(ft)
                sensitivity=(base['recall']-metrics['recall']) if base['recall'] is not None else None
                if sensitivity is not None:sens.append(sensitivity)
                if a.ap:
                    from coco_metrics import coco_ap
                    values=coco_ap(fault,names);metrics.update(values);aps.append(values)
                per_run.append({'condition_id':cond['id'],'cell':b,'repeat':rep,
                    **metrics,'sensitivity_recall':sensitivity})
            interval=paired_ci(ctp,np.array(f_tps),ngt,groups,a.bootstrap,a.bootstrap_seed,True)
            row={'condition_id':cond['id'],'cell':b,'grid_label':f'G{b+1}',
                'recall_clean':base['recall'],'sensitivity_recall':float(np.mean(sens)) if sens else None,
                'sensitivity_pp':float(np.mean(sens)*100) if sens else None,
                'sensitivity_pattern_sd':float(np.std(sens,ddof=1)) if len(sens)>1 else None,
                'pointwise_ci_low':interval[0],'pointwise_ci_high':interval[1],
                'confidence_drop_clean_tp':cd_sum/cd_n if cd_n else None,
                'soft_fp_change_per_image':float(np.mean(soft)),
                'mean_actual_changed_pixels':float(np.mean(changed)),
                'mean_l1_rgb_sum':float(np.mean(l1)),
                'box_area_occupancy':float(oc[b]/len(clean))}
            if a.ap:
                row['mean_coco_map50']=float(np.mean([x['coco_map50'] for x in aps])) if base['coco_map50'] is not None else None
                row['mean_coco_map50_95']=float(np.mean([x['coco_map50_95'] for x in aps])) if base['coco_map50_95'] is not None else None
            summary.append(row)
    write_csv(out/'condition_metrics.csv',per_run);write_csv(out/'grid_summary.csv',summary)
    pair_file.close();pair_tmp.replace(out/'paired_images.csv')
    dump_json(out/'analysis_meta.json',{'run_hash':meta['run_hash'],'conf_operating':t,'match_iou':mi,
        'conf_floor':floor,'bootstrap':a.bootstrap,'bootstrap_seed':a.bootstrap_seed,
        'source_sha256':{name:digest_file(Path(__file__).parent/name) for name in ['analyze.py','core.py','coco_metrics.py']},
        'interval_scope':'pointwise group+pattern bootstrap; conditional on this trained checkpoint and fixed protocol',
        'macro_map':'COCO official evaluator, maxDets=100, only if --ap',
        'sensitivity_unit':'fraction; multiply by 100 for percentage points'})
    if a.plots:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        vals=[r['sensitivity_pp'] for r in summary if r['sensitivity_pp'] is not None]
        lim=max([abs(v) for v in vals]+[.01])
        for c in cfg['conditions']:
            ar=np.array([np.nan if r['sensitivity_pp'] is None else r['sensitivity_pp'] for r in summary if r['condition_id']==c['id']]).reshape(gr,gc)
            fig,ax=plt.subplots(figsize=(6,5))
            im=ax.imshow(ar,vmin=-lim,vmax=lim)
            for y in range(gr):
                for x in range(gc):ax.text(x,y,f'G{y*gc+x+1}\n{ar[y,x]:.2f}',ha='center',va='center')
            ax.set_title(c['id']+' | Recall drop (pp)');ax.set_xlabel('Image column');ax.set_ylabel('Image row')
            fig.colorbar(im,ax=ax,label='percentage points');fig.tight_layout()
            fig.savefig(out/(c['id']+'_recall_map.png'),dpi=200);plt.close(fig)
    print(out)

if __name__=='__main__':main()
