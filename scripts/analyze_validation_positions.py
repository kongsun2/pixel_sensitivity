"""Exploratory joint position bands from fixed-model cached detection counts.

All64locations share one group draw and one mask draw in each bootstrap replicate.
Independent resampling of locations would destroy the experiment's pairing.
No permutation or exact/confirmatory p-value is computed.
"""
from __future__ import annotations
import os
for _name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[_name]='4'
import argparse,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from core import digest_file,dump_json
DEFAULT_OUT=ROOT/'reports/research_validation_2026-11-11/positions'
DEFAULTS=[('gc10_det',ROOT/'runs/evaluation/gc10_det_s11_bright25_g8_p10'),('neu_det',ROOT/'runs/evaluation/neu_det_s11_bright25_g8_p10'),('pcb_ind',ROOT/'runs/evaluation/pcb_ind_s11_bright25_g8_p10')]
TITLES={'gc10_det':'GC10-DET','neu_det':'NEU-DET','pcb_ind':'PCB-IND'}


def joint_position_bootstrap(loss_counts,gt_counts,groups,n_boot=4000,seed=12011,alpha=.05,chunk=128):
    """Return micro-Recall loss bands conditional on one trained checkpoint.

    loss_counts[r,b,i] = TP_clean_i - TP_fault[r,b,i], signed so recoveries stay.
    gt_counts[i] counts each image's GT once, never repeated over r or b.
    Images in a supplied cluster stay together. Resample group levels G and mask
    levels R independently with multinomial weights; product weights preserve the
    complete crossed array and location pairing. The same weighted GT denominator
    is used in all B locations in a draw. This is a pigeonhole-style approximation.
    """
    loss=np.asarray(loss_counts,dtype=float);gt=np.asarray(gt_counts,dtype=float)
    if loss.ndim!=3 or gt.ndim!=1 or loss.shape[2]!=len(gt) or len(groups)!=len(gt):raise ValueError('Bad paired-array shapes')
    if not np.isfinite(loss).all() or not np.isfinite(gt).all() or (gt<0).any() or gt.sum()<=0:raise ValueError('Finite positive GT denominator required')
    if not 0<alpha<1 or n_boot<20 or min(loss.shape[:2])<2:raise ValueError('Need multiple masks/locations and bootstrap draws')
    unique=sorted(set(str(g) for g in groups));mapping={g:i for i,g in enumerate(unique)};idx=np.array([mapping[str(g)] for g in groups]);G=len(unique)
    if G<2:raise ValueError('At least two resampling groups required')
    R,B,N=loss.shape;grouped=np.zeros((R,B,G),float)
    for g in range(G):grouped[:,:,g]=loss[:,:,idx==g].sum(axis=2)
    group_gt=np.bincount(idx,weights=gt,minlength=G)
    sensitivity=loss.mean(axis=0).sum(axis=1)/gt.sum()
    deviation=sensitivity-sensitivity.mean()
    rng=np.random.default_rng(seed);draws=[];attempted=0
    while sum(len(a) for a in draws)<n_boot:
        remaining=n_boot-sum(len(a) for a in draws);K=min(chunk,remaining)
        wg=rng.multinomial(G,np.full(G,1/G),size=K).astype(float)
        wr=rng.multinomial(R,np.full(R,1/R),size=K).astype(float)/R
        denominator=wg@group_gt
        weighted=np.einsum('kr,rbg->kbg',wr,grouped,optimize=True)
        numerator=np.einsum('kbg,kg->kb',weighted,wg,optimize=True)
        valid=denominator>0
        if valid.any():draws.append(numerator[valid]/denominator[valid,None])
        attempted+=K
        if attempted>n_boot*10:raise ValueError('Too many zero-GT draws')
    samples=np.concatenate(draws,axis=0)[:n_boot]
    dev_samples=samples-samples.mean(axis=1,keepdims=True)
    errors=samples-sensitivity[None,:];dev_errors=dev_samples-deviation[None,:]
    # A single centered max-absolute radius applies simultaneously to all B cells.
    raw_radius=float(np.quantile(np.max(np.abs(errors),axis=1),1-alpha,method='higher'))
    deviation_radius=float(np.quantile(np.max(np.abs(dev_errors),axis=1),1-alpha,method='higher'))
    pointwise=np.quantile(samples,[alpha/2,1-alpha/2],axis=0)
    grand_samples=samples.mean(axis=1)
    return {'sensitivity':sensitivity,'deviation':deviation,'grand_mean':float(sensitivity.mean()),
        'draws_sensitivity':samples,'raw_critical_radius':raw_radius,'deviation_critical_radius':deviation_radius,
        'raw_simultaneous_low':sensitivity-raw_radius,'raw_simultaneous_high':sensitivity+raw_radius,
        'deviation_simultaneous_low':deviation-deviation_radius,'deviation_simultaneous_high':deviation+deviation_radius,
        'pointwise_low':pointwise[0],'pointwise_high':pointwise[1],
        'grand_mean_percentile_ci':np.quantile(grand_samples,[alpha/2,1-alpha/2]).tolist(),
        'n_groups':G,'group_ids':unique,'n_images':N,'n_patterns':R,'n_cells':B,'n_boot':n_boot,'seed':seed,'attempted_draws':attempted}


def load_paired_arrays(run:Path):
    meta=json.loads((run/'run_meta.json').read_text());cfg=meta['config']
    R=cfg['repeats'];B=cfg['grid']['rows']*cfg['grid']['cols'];path=run/'analysis/paired_images.csv'
    df=pd.read_csv(path,usecols=['condition_id','repeat','cell','image_id','group_id','n_gt','tp_clean','tp_fault'],dtype={'image_id':str,'group_id':str})
    if df.condition_id.nunique()!=1:raise ValueError('One injection condition per position analysis')
    ids=sorted(df.image_id.unique().tolist());N=len(ids)
    if ids!=sorted(meta['selected_ids']):raise ValueError('Paired CSV test IDs differ from run metadata')
    image_codes=pd.Categorical(df.image_id,categories=ids).codes
    if not df['repeat'].between(0,R-1).all() or not df.cell.between(0,B-1).all():raise ValueError('Bad repeat/cell')
    flat=((df['repeat'].to_numpy()*B+df.cell.to_numpy())*N+image_codes)
    if len(df)!=R*B*N or len(np.unique(flat))!=R*B*N:raise ValueError('Incomplete or duplicated paired records')
    base=df[(df['repeat']==0)&(df.cell==0)].set_index('image_id').loc[ids]
    gt=base.n_gt.to_numpy();clean=base.tp_clean.to_numpy();groups=base.group_id.to_numpy()
    if not np.array_equal(df.n_gt.to_numpy(),gt[image_codes]) or not np.array_equal(df.tp_clean.to_numpy(),clean[image_codes]):raise ValueError('Clean GT/TP must be fixed across all positions and masks')
    if not np.array_equal(df.group_id.to_numpy(),groups[image_codes]):raise ValueError('Image group changed')
    loss=np.empty(R*B*N,dtype=np.int16);loss[flat]=df.tp_clean.to_numpy()-df.tp_fault.to_numpy();loss=loss.reshape(R,B,N)
    summary=pd.read_csv(run/'analysis/grid_summary.csv').sort_values('cell')
    empirical=loss.mean(axis=0).sum(axis=1)/gt.sum()
    if not np.allclose(empirical,summary.sensitivity_recall.to_numpy(),atol=1e-12,rtol=0):raise ValueError('Micro-Recall reconstruction differs from existing analysis')
    return loss,gt,groups,ids,meta,path


def summarize_dataset(dataset,run,out,n_boot,seed):
    loss,gt,groups,ids,meta,path=load_paired_arrays(run);modes=[('recorded_groups',groups)]
    if dataset.startswith('pcb'):
        modes.append(('numeric_id_groups',np.array([name.split('_')[0] for name in ids])))
    recorded_numeric=np.array_equal(np.asarray(groups).astype(str),np.array([name.split('_')[0] for name in ids])) if dataset.startswith('pcb') else False
    summaries=[];all_rows=[]
    for mode,labels in modes:
        result=joint_position_bootstrap(loss,gt,labels,n_boot=n_boot,seed=seed)
        rows=[]
        for b in range(result['n_cells']):
            lo=result['deviation_simultaneous_low'][b];hi=result['deviation_simultaneous_high'][b]
            row={'dataset':dataset,'grouping':mode,'cell':b,'grid_label':f'G{b+1}',
                'sensitivity_pp':100*result['sensitivity'][b],'grand_mean_pp':100*result['grand_mean'],'deviation_from_grand_mean_pp':100*result['deviation'][b],
                'raw_pointwise_low_pp':100*result['pointwise_low'][b],'raw_pointwise_high_pp':100*result['pointwise_high'][b],
                'raw_simultaneous_low_pp':100*result['raw_simultaneous_low'][b],'raw_simultaneous_high_pp':100*result['raw_simultaneous_high'][b],
                'deviation_simultaneous_low_pp':100*lo,'deviation_simultaneous_high_pp':100*hi,
                'above_grand_mean_joint_band':bool(lo>0),'below_grand_mean_joint_band':bool(hi<0)}
            rows.append(row);all_rows.append(row)
        frame=pd.DataFrame(rows);frame.to_csv(out/f'{dataset}_{mode}_position_bands.csv',index=False)
        np.savez_compressed(out/f'{dataset}_{mode}_bootstrap_draws.npz',sensitivity=result['draws_sensitivity'],observed=result['sensitivity'],deviation=result['deviation'])
        above=frame.loc[frame.above_grand_mean_joint_band,'grid_label'].tolist();below=frame.loc[frame.below_grand_mean_joint_band,'grid_label'].tolist()
        summary={'dataset':dataset,'grouping':mode,'n_images':len(ids),'n_gt':int(gt.sum()),'n_patterns':result['n_patterns'],'n_cells':result['n_cells'],'n_groups':result['n_groups'],
            'grand_mean_drop_pp':100*result['grand_mean'],'grand_mean_ci_low_pp':100*result['grand_mean_percentile_ci'][0],'grand_mean_ci_high_pp':100*result['grand_mean_percentile_ci'][1],
            'min_sensitivity_pp':float(frame.sensitivity_pp.min()),'max_sensitivity_pp':float(frame.sensitivity_pp.max()),'max_cell':frame.loc[frame.sensitivity_pp.idxmax(),'grid_label'],
            'deviation_band_radius_pp':100*result['deviation_critical_radius'],'raw_band_radius_pp':100*result['raw_critical_radius'],
            'above_grand_mean_cells':','.join(above),'below_grand_mean_cells':','.join(below),'n_above_grand_mean':len(above),'n_below_grand_mean':len(below),
            'constant_location_effect_vector_in_joint_region':not bool(above or below),
            'n_boot':n_boot,'seed':seed,'run_hash':meta['run_hash'],'manifest_sha256':meta['manifest_sha256'],'weights_sha256':meta['weights_sha256'],'recorded_groups_are_numeric_ids':bool(recorded_numeric),'run_path':str(run),'paired_csv_sha256':digest_file(path)}
        summaries.append(summary);print(json.dumps(summary,ensure_ascii=False),flush=True)
    return summaries,all_rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',action='append',help='Optional DATASET=RUN_DIRECTORY; repeat for custom fixed-model runs')
    p.add_argument('--out',type=Path,default=DEFAULT_OUT);p.add_argument('--bootstrap',type=int,default=4000);p.add_argument('--seed',type=int,default=12011);a=p.parse_args()
    items=DEFAULTS if not a.run else [(x.split('=',1)[0],Path(x.split('=',1)[1]).resolve()) for x in a.run]
    out=a.out.resolve();out.mkdir(parents=True,exist_ok=True);summary=[];positions=[]
    for d,run in items:
        ss,pp=summarize_dataset(d,run,out,a.bootstrap,a.seed);summary+=ss;positions+=pp
    pd.DataFrame(summary).to_csv(out/'position_effect_summary.csv',index=False);pd.DataFrame(positions).to_csv(out/'all_position_bands.csv',index=False)
    table='| 데이터셋 / 재표집 그룹 | 그룹 수 | 전체 평균 Recall 감소 | 위치별 최소~최대 | 평균보다 높은 칸 | 평균보다 낮은 칸 |\n|---|---:|---:|---:|---|---|\n'
    for s in summary:
        table+=f"| {TITLES.get(s['dataset'],s['dataset'])} / {s['grouping']} | {s['n_groups']} | {s['grand_mean_drop_pp']:.3f}%p | {s['min_sensitivity_pp']:.3f} ~ {s['max_sensitivity_pp']:.3f}%p | {s['above_grand_mean_cells'] or '없음'} | {s['below_grand_mean_cells'] or '없음'} |\n"
    findings=[]
    for s in summary:
        title=TITLES.get(s['dataset'],s['dataset']);findings.append(f"- {title} ({s['grouping']}): 전체 평균 감소 {s['grand_mean_drop_pp']:.3f}%p, 전체 평균의 percentile 95% 구간 [{s['grand_mean_ci_low_pp']:.3f}, {s['grand_mean_ci_high_pp']:.3f}]%p. 64칸의 평균 대비 편차에 대한 동시 밴드 반경은 ±{s['deviation_band_radius_pp']:.3f}%p이며, {s['n_above_grand_mean']}칸은 전체 위치 평균보다 높은 감소, {s['n_below_grand_mean']}칸은 낮은 감소를 보였다.")
    pcb_rows=[v for v in summary if v['dataset'].startswith('pcb')]
    if not pcb_rows:
        pcb_note='이번 입력에는 PCB가 없어 숫자 ID 그룹 민감도 분석을 적용하지 않았다.'
    elif all(v['recorded_groups_are_numeric_ids'] for v in pcb_rows):
        pcb_note='이번 PCB 입력의 기록된 그룹은 이미 숫자 ID다. 기록 그룹과 숫자 ID 재표집 결과가 같은 것은 동일한 그룹 구성을 다시 적용한 일치성 확인이며, 서로 다른 두 독립 검증이 아니다.'
        audit_path=ROOT/'reports/research_validation_2026-11-11/pcb_split/split_audit.json'
        if audit_path.exists():
            audit=json.loads(audit_path.read_text())
            if all(v['manifest_sha256']==audit.get('canonical_manifest_sha256') for v in pcb_rows):
                overlap=audit['new_overlap']['n_groups_crossing_splits']
                pcb_note+=f" 분할 audit의 canonical manifest SHA256이 현재 입력과 일치하며, 숫자 ID가 여러 분할에 걸치는 그룹 수는 {overlap}개로 기록됐다. 이는 파일명 숫자 ID 기준 분리이며 실제 물리 보드 식별자의 의미까지 검증한 것은 아니다."
        old_base_path=ROOT/'runs/evaluation/pcb_ind_s11_bright25_g8_p10/analysis/clean_metrics.json'
        if old_base_path.exists():
            old_base=json.loads(old_base_path.read_text());current=pcb_rows[0]
            pcb_note+=f" 기존 공식 분할 PCB 시험자료는 {old_base['n_images']}장/{old_base['n_gt']}개 GT였고, 이번 입력은 {current['n_images']}장/{current['n_gt']}개 GT다. 시험 코호트와 학습 모델이 함께 바뀌었으므로 두 결과의 수치 차이를 분할 수정의 인과 효과나 순수 누수 제거 효과로 해석하지 않는다."
    else:
        pcb_note='PCB는 기록된 면 접두어(t/b 포함)와 숫자 ID를 묶는 두 가지 재표집을 비교했다. 숫자 ID 묶음은 시험자료 내 같은 ID 양면의 상관을 더 보수적으로 다루는 민감도 분석이다. **학습–시험 간 숫자 ID 중복이나 정보 누수를 제거하지 않으며, 독립 보드 일반화를 검증하지 않는다.** 이는 별도 보드 분할 재학습 실험에서 다뤄야 한다.'
    report=f'''# 위치 효과의 공동 재표집 분석

기존 고정 YOLOv8n, 핫픽셀25개, 8×8격자, 분산형 패턴10개의 저장된 평가 결과를 사용했다. 각 위치를 서로 독립 표본으로 취급하지 않고 같은 시험 영상과 같은 패턴이 64개 위치에서 반복된 구조를 유지했다. 결과는 **기존 학습 체크포인트와 시험 분할에 조건부인 탐색적 분석**이다.

## 수치 요약

{table}

위 표의 높음/낮음은 **해당 데이터셋 64개 위치의 전체 평균 효과와의 차이**다. 원본 대비 성능 감소가 0보다 큰지를 뜻하는 것과 다르다. 상세 CSV에는 원본 대비 효과와 평균 대비 편차, 각자의 밴드를 모두 저장했다.

{chr(10).join(findings)}

## 계산식

이미지 i의 정답 수를 n_i, 원본에서 맞춘 객체 수를 C_i, 패턴 r·위치 b에서 맞춘 수를 F_rbi라 두었다. 위치 효과는 S_b = [평균_r 합_i(C_i−F_rbi)] / 합_i n_i 이다. 같은 객체의 회복이 있으면 음수 차이도 보존한다. 비교 대상은 D_b = S_b − 평균_b(S_b)로 정의했다.

이미지를 주어진 그룹 g로 합산하고, 각 재표집에서 그룹 수준을 G회, 패턴 수준을 R회 독립적으로 복원 추출했다. 같은 그룹 가중치와 같은 패턴 가중치를 모든64위치에 공통 적용했다. 분자는 그룹×패턴 가중치의 곱을 사용하고, 분모는 선택된 그룹의 정답 수로 매번 다시 계산했다. 따라서 원본–손상 및 위치 간 짝을 유지한다. 영상이나 위치마다 다른 패턴 draw를 적용하지 않았다.

{a.bootstrap:,}회 재표집에서 S*_b와 D*_b를 구했다. 모든 위치를 함께 고려한 오차 M* = max_b |D*_b−D_b|의 95% 분위수 q를 사용해 **[D_b−q, D_b+q]**를 만들었다. 원본 대비 효과 S_b에도 같은 방식의 별도 공동 밴드를 만들었다. 64개 pointwise구간을 각각 검사하는 대신 전체 벡터의 최대 오차를 기준으로 삼는다. 위치별 표준오차로 나누는 studentization을 하지 않아 작은 분산의 위치에는 넓은 밴드가 될 수 있다.

이 공동 영역 안에 상수 위치 효과에 대응하는 D=0 벡터가 포함되는지와, 각 위치 밴드가0을 벗어나는지를 기술했다. **교환가능성을 가정한 permutation p-value 또는 정확한 확증적 p-value는 계산하지 않았다.**

## 방법의 근거와 한계

그룹과 전역 패턴은 교차 요인이다. 두 요인의 수준을 독립 재표집하고 관측에 곱 가중치를 주는 아이디어는 [Owen (2007), The pigeonhole bootstrap](https://doi.org/10.1214/07-AOAS122), [Owen & Eckles (2012), Bootstrapping data arrays of arbitrary order](https://doi.org/10.1214/12-AOAS547)를 참고했다. 이 문헌은 교차 자료의 bootstrap이 일반적으로 정확하지 않으며, 가정에 따라 보수적 분산 추정이 가능함을 다룬다. **해당 문헌 인용이 본 비율 통계량·64차원 최대편차 밴드의 유한표본 95% 피복률을 보장한다는 뜻은 아니다.** 여기에 사용한 것은 그 재표집 분포를 이용한 명목95% 근사 동시 밴드다.

패턴이10개뿐이고 그룹 수 및 크기가 제한되므로 근사 오차와 Monte Carlo 변동이 있다. 그룹을 독립 추출 단위로 볼 수 있다는 가정 역시 실제 코일·보드 정보로 검증되지 않았다. 학습 시드 불확실성, 모델 선택, 여러 조건/데이터셋/추가 분석에 걸친 전체 다중 비교는 포함하지 않는다. 같은 시험자료를 앞선 탐색에 사용했으므로 독립 확인 실험도 아니다.

## PCB 그룹과 코호트 해석

{pcb_note}

공동 밴드가 평균 대비 위치 차이를 뒷받침하더라도, 제품 결함의 공간 분포·객체 크기·화소 손상 겹침을 통제한 인과 효과가 아니다. 중앙 ROI에 결함이 몰린 영향은 별도 분포 분석과 함께 해석해야 한다. 센서의 절대 위치가 본질적으로 취약하다고 일반화하지 않는다.

## 재현

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \\
  .venv/bin/python scripts/analyze_validation_positions.py --bootstrap {a.bootstrap} --seed {a.seed}
```

새 체크포인트 결과는 `--run DATASET=/absolute/run/path --out /absolute/output/path`로 동일 계산을 적용할 수 있다. run은 완료된 `analysis/paired_images.csv`, `grid_summary.csv`, `run_meta.json`을 포함해야 한다. 입력 SHA256과 실행 설정은 `analysis_meta.json`에, 재표집 수치 배열은 각 `.npz`에 저장했다.
'''
    (out/'REPORT_KO.md').write_text(report)
    dump_json(out/'analysis_meta.json',{'script_sha256':digest_file(__file__),'bootstrap':a.bootstrap,'seed':a.seed,'nominal_level':.95,
       'method':'crossed group x pattern multinomial product bootstrap; common draw across all locations; centered unstudentized max-absolute simultaneous bands',
       'exact_or_confirmatory_p_values':False,'inputs':[{ 'dataset':s['dataset'],'grouping':s['grouping'],'run_hash':s['run_hash'],'paired_csv_sha256':s['paired_csv_sha256']} for s in summary],
       'sources':['https://doi.org/10.1214/07-AOAS122','https://doi.org/10.1214/12-AOAS547']})
    print(out/'REPORT_KO.md',flush=True)

if __name__=='__main__':main()
