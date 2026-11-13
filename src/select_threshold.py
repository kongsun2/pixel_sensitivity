"""Choose an optional operating threshold on CLEAN VALIDATION only.
Default manual protocol may instead fix 0.25 a priori; never tune on test.
"""
import argparse,json
from pathlib import Path
import numpy as np
from core import dump_json,load_jsonl,operational_metrics,digest_file


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();root=Path(a.run)
    meta=json.loads((root/'run_meta.json').read_text())
    if meta['split']!='val':raise ValueError('Threshold tuning is allowed only on val')
    if meta['config']['predict']['conf_floor']>.01:raise ValueError('Prediction floor is too high')
    path=root/'predictions/clean.jsonl';mark=json.loads(path.with_suffix('.done.json').read_text())
    if mark['sha256']!=digest_file(path):raise ValueError('Invalid cached validation predictions')
    data=load_jsonl(path);mi=meta['config']['evaluation']['match_iou']
    scores=[{'threshold':float(t),**operational_metrics(data,float(t),mi)} for t in np.arange(.05,.951,.05)]
    # Explicit proposed rule: micro F1; tie -> larger recall; tie -> smaller threshold.
    best=max(scores,key=lambda d:(-1 if d['f1'] is None else d['f1'],
                                   -1 if d['recall'] is None else d['recall'],-d['threshold']))
    dump_json(a.out,{'rule':'clean val maximum micro F1; tie recall then lower threshold',
        'selected_threshold':best['threshold'],'validation_run_hash':meta['run_hash'],'all_candidates':scores})
    print(best['threshold'])

if __name__=='__main__':main()
