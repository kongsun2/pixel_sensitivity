"""Reference implementation: image-space pixel faults and fixed-threshold metrics.
No physical sensor, ISP, maintenance, or adversarial attack is implemented here.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Any
import numpy as np


def digest_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def digest_json(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def stable_seed(*parts: Any) -> int:
    return int(digest_json(parts)[:16], 16)


def dump_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.partial')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2,
        allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def load_jsonl(path: str | Path) -> list[dict]:
    rows = [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines() if s.strip()]
    ids = [r['image_id'] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate image_id in manifest/shard')
    return rows


def cell_bounds(height: int, width: int, rows: int, cols: int, cell: int) -> tuple[int,int,int,int]:
    if min(height, width, rows, cols) <= 0 or rows > height or cols > width:
        raise ValueError('Invalid image/grid dimensions')
    if not 0 <= cell < rows * cols:
        raise ValueError('Cell out of range')
    r, c = divmod(cell, cols)
    return (c * width // cols, r * height // rows,
            (c + 1) * width // cols, (r + 1) * height // rows)


def fault_pattern(height: int, width: int, rows: int, cols: int,
                  count: int, layout: str, side: int, seed: int) -> np.ndarray:
    """Return unique (y,x) offsets reusable in every grid cell of this image shape.
    At most 1-pixel remainder per cell axis is excluded so the same offsets fit all cells.
    Count controls spatial sites, not RGB channels. Pattern seed must NOT include cell.
    Dispersed sites and separate clusters have at least a one-pixel Chebyshev gap.
    """
    ch, cw = height // rows, width // cols
    if min(ch, cw) <= 0 or count < 0 or count > ch * cw:
        raise ValueError('Invalid fault count or grid capacity')
    if count == 0:
        return np.empty((0, 2), dtype=np.int64)
    if layout not in {'dispersed', 'cluster'}:
        raise ValueError('layout must be dispersed or cluster')
    if layout == 'dispersed':
        side = 1
    if side < 1 or side > min(ch, cw) or count % (side * side):
        raise ValueError('Cluster count must be divisible by side**2 and fit grid cell')
    rng = np.random.default_rng(seed)
    # The same ordered candidate set gives nested patterns when only count increases.
    nh, nw = ch - side + 1, cw - side + 1
    candidates = rng.permutation(nh * nw)
    occupied = np.zeros((ch, cw), dtype=bool)
    pts: list[tuple[int,int]] = []
    for flat in candidates:
        y, x = divmod(int(flat), nw)
        if occupied[max(0,y-1):min(ch,y+side+1), max(0,x-1):min(cw,x+side+1)].any():
            continue
        occupied[y:y+side, x:x+side] = True
        pts.extend((yy, xx) for yy in range(y,y+side) for xx in range(x,x+side))
        if len(pts) == count:
            return np.asarray(pts, dtype=np.int64)
    raise ValueError('Pattern cannot fit with spacing constraint: reduce count/grid density')


def inject_fault(clean_rgb: np.ndarray, offsets_yx: np.ndarray,
                 bounds: tuple[int,int,int,int], target: int,
                 alpha: float = 1.0, channels: str = 'all') -> tuple[np.ndarray,dict]:
    """Blend selected image-space channels toward a fixed target; always copy source.
    alpha<1 is a partial contrast stress model, NOT a physically dead/stuck pixel.
    """
    if clean_rgb.dtype != np.uint8 or clean_rgb.ndim != 3 or clean_rgb.shape[2] != 3:
        raise ValueError('Expected uint8 RGB HxWx3')
    if not 0 <= target <= 255 or not 0 <= alpha <= 1:
        raise ValueError('target must be 0..255; alpha must be 0..1')
    if channels not in {'all', 'r', 'g', 'b'}:
        raise ValueError('Invalid channel policy')
    x0,y0,x1,y1 = bounds
    h,w = clean_rgb.shape[:2]
    if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
        raise ValueError('Bounds outside image')
    off = np.asarray(offsets_yx, dtype=np.int64).reshape(-1,2)
    if len(np.unique(off, axis=0)) != len(off):
        raise ValueError('Duplicate fault sites')
    ys, xs = off[:,0] + y0, off[:,1] + x0
    if np.any(ys < y0) or np.any(ys >= y1) or np.any(xs < x0) or np.any(xs >= x1):
        raise ValueError('Fault crosses its assigned cell')
    out = clean_rgb.copy()
    ch = [0,1,2] if channels == 'all' else [{'r':0,'g':1,'b':2}[channels]]
    for c in ch:
        old = clean_rgb[ys,xs,c].astype(np.float64)
        out[ys,xs,c] = np.rint((1-alpha)*old + alpha*target).clip(0,255).astype(np.uint8)
    delta = out[ys,xs].astype(np.int16) - clean_rgb[ys,xs].astype(np.int16)
    changed = np.any(delta != 0, axis=1)
    meta = {
        'count_commanded': int(len(off)), 'count_changed': int(changed.sum()),
        'count_changed_channels': int(np.count_nonzero(delta)),
        'fault_fraction_image': float(len(off)/(h*w)),
        'fault_fraction_cell': float(len(off)/((x1-x0)*(y1-y0))),
        'l1_rgb_sum': int(np.abs(delta).sum()),
        'linf_rgb': int(np.abs(delta).max(initial=0)),
        'coordinates_xy': [[int(x),int(y)] for y,x in zip(ys,xs)],
        'channel_policy': channels, 'target': int(target), 'alpha': float(alpha),
    }
    return out, meta


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float).reshape(-1,4)
    b = np.asarray(b, dtype=float).reshape(-1,4)
    tl = np.maximum(a[:,None,:2], b[None,:,:2])
    br = np.minimum(a[:,None,2:], b[None,:,2:])
    inter = np.maximum(br-tl,0).prod(axis=2)
    area_a = np.maximum(a[:,2:]-a[:,:2],0).prod(axis=1)
    area_b = np.maximum(b[:,2:]-b[:,:2],0).prod(axis=1)
    union = area_a[:,None] + area_b[None,:] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union>0)


def match_predictions(gt: list[dict], pred: list[dict], threshold: float,
                      match_iou: float) -> dict:
    """Score-first, class-aware, one-to-one matching for this protocol.
    Ties: score desc -> original pred index; GT tie: smallest gt list index.
    Not a replacement for official COCO AP evaluation.
    """
    if not 0 <= threshold <= 1 or not 0 <= match_iou <= 1:
        raise ValueError('Invalid metric thresholds')
    eligible = [j for j,p in enumerate(pred) if p['score'] >= threshold]
    eligible.sort(key=lambda j: (-float(pred[j]['score']),j))
    ious = iou_matrix([p['bbox_xyxy'] for p in pred], [g['bbox_xyxy'] for g in gt])
    used: set[int] = set()
    matches = []
    fp = []
    scores = [0.0]*len(gt)
    overlaps = [0.0]*len(gt)
    for j in eligible:
        choices = [i for i,g in enumerate(gt)
                   if i not in used and g['class_id'] == pred[j]['class_id'] and ious[j,i] >= match_iou]
        if not choices:
            fp.append(j)
            continue
        i = max(choices, key=lambda i: (ious[j,i],-i))
        used.add(i)
        scores[i] = float(pred[j]['score'])
        overlaps[i] = float(ious[j,i])
        matches.append([i,j])
    return {
        'tp':len(matches), 'fp':len(fp), 'fn':len(gt)-len(matches),
        'gt_matched':[i in used for i in range(len(gt))],
        'gt_scores':scores, 'gt_ious':overlaps,
        'matches':matches, 'fp_indices':fp,
        'soft_fp_sum':float(sum(pred[j]['score'] for j in fp)),
    }


def operational_metrics(records: list[dict], threshold: float, match_iou: float) -> dict:
    tp=fp=fn=0
    for r in records:
        m = match_predictions(r['gt'],r['pred'],threshold,match_iou)
        tp+=m['tp']; fp+=m['fp']; fn+=m['fn']
    recall=tp/(tp+fn) if tp+fn else None
    precision=tp/(tp+fp) if tp+fp else None
    return {
        'n_images':len(records), 'n_gt':tp+fn, 'tp':tp,'fp':fp,'fn':fn,
        'recall':recall,'miss_rate':1-recall if recall is not None else None,
        'precision':precision,'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
        'fp_per_image':fp/len(records) if records else None,
    }
