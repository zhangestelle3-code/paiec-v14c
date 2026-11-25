#!/usr/bin/env python3
"""Paired local diagnostic: V14C (5%) vs V19A (7.5%), on an older local SQLite.

NOT a replication of the official PAIEC scoring environment. No hidden-test information.
Both predictors use identical subject, items, feedback labels, and evaluation responses.
The V14C text branch is vectorized algebraically; both models share that exact branch.
"""
import argparse, collections, csv, hashlib, importlib.util, json, math, os, random, sqlite3, statistics, sys, zipfile
from pathlib import Path
import numpy as np

BENCHMARKS = ('matharena', 'multi_swebench', 'real_webagents', 'researchcodebench', 'swe_rebench')
BUDGETS = (0,1,3,7,15,31)
ALC_WEIGHTS = (0.1, 0.2, 0.2, 0.2, 0.2, 0.1)

def load_zip_model(path, name, workdir):
    workdir.mkdir(exist_ok=True, parents=True)
    with zipfile.ZipFile(path) as z:
        data=z.read('model.py')
    p=workdir/(name+'.py'); p.write_bytes(data)
    spec=importlib.util.spec_from_file_location(name,str(p))
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod, hashlib.sha256(data).hexdigest()

def extract_data(db, model):
    c=sqlite3.connect(db)
    itemdata={}
    for iid,bench,question in c.execute('select item_id,bench,question from items'):
        question=question if isinstance(question,str) else ''
        itemdata[iid]=(bench,question,model._features({'item_content':question}))
    profiles={}
    for sid,bench,name,provider,norm in c.execute('select subject_id,bench,name,provider,normalized from subjects'):
        profiles[(bench,sid)]={'normalized_name':norm or name or '', 'provider':provider or '', 'harness':'', 'harness_version':'', 'reasoning_effort':''}
    groups=collections.defaultdict(lambda:collections.defaultdict(list))
    q='select bench,subject_id,item_id,score from scores where (score=0 or score=1)'
    for bench,sid,iid,score in c.execute(q):
        if bench in BENCHMARKS and iid in itemdata and itemdata[iid][0]==bench:
            groups[(bench,sid)][iid].append(int(score))
    c.close()
    groups={k:v for k,v in groups.items() if len(v)>=80 and k in profiles}
    return groups, itemdata, profiles

def text_deltas_for_eval(labeled_items, eval_ids, itemdata, model):
    # Mimic model._text_adjustment on exactly the same 15 features and gate conditions.
    # Do not train on evaluation response labels; target text is always visible to predict.
    tot=len(labeled_items)
    if tot<model.MIN_TEXT_LABELS:
        return np.zeros(len(eval_ids),dtype=float)
    valid=[(itemdata[iid][2],float(y)) for iid,y in labeled_items if itemdata[iid][2] is not None]
    n=len(valid)
    if n < model.MIN_TEXT_LABELS or n<0.8*tot:
        return np.zeros(len(eval_ids),dtype=float)
    # Sorting replicates canonical source code (it has negligible effect on FP results).
    valid.sort(key=lambda a:(tuple(a[0]),a[1]))
    X=np.asarray([x for x,y in valid],dtype=float)
    y=np.asarray([y for x,y in valid],dtype=float)
    means=X.mean(axis=0)
    stds=np.maximum(0.05, X.std(axis=0,ddof=0))
    Z=(X-means)/stds
    beta=np.linalg.solve(Z.T@Z+model.RIDGE*np.eye(len(means)), Z.T@(y-y.mean()))
    low=X.min(axis=0)-0.5*stds
    high=X.max(axis=0)+0.5*stds
    labeled_chars=sum(len(itemdata[iid][1]) for iid,y in labeled_items)
    out=np.zeros(len(eval_ids),dtype=float)
    for idx,iid in enumerate(eval_ids):
        feat=itemdata[iid][2]
        if feat is None or labeled_chars+len(itemdata[iid][1])>model.MAX_TOTAL_TEXT_CHARS:
            continue
        f=np.asarray(feat,dtype=float)
        if not (np.all(f>=low) and np.all(f<=high)):
            continue
        q=(f-means)/stds
        if np.max(np.abs(q))>3:
            continue
        delta=float(q@beta)
        if math.isfinite(delta):
            out[idx]=min(model.MAX_ADJUSTMENT,max(-model.MAX_ADJUSTMENT,delta))
    return out

def group_seed_scores(bench,sid,replies,itemdata,mod14,mod19,seed):
    # Distinct item split, never reuse an item between feedback and evaluation.
    ids=sorted(replies)
    token=f'{seed}|{bench}|{sid}'.encode('utf-8')
    rng=random.Random(int(hashlib.sha256(token).hexdigest()[:16],16))
    rng.shuffle(ids)
    n_acq=len(ids)//2
    acquisition_ids=ids[:n_acq]
    evaluation_ids=ids[n_acq:]
    # A revealed label is one response trial for a distinct item (deterministic trial selection).
    acquired=[(iid,rng.choice(replies[iid])) for iid in acquisition_ids[:31]]
    yy=np.array([v for iid in evaluation_ids for v in replies[iid]],dtype=float)
    # weights ensure every response trial, including repeated trials, gets scored.
    item_weight=np.array([len(replies[iid]) for iid in evaluation_ids])
    item_mean=np.array([sum(replies[iid])/len(replies[iid]) for iid in evaluation_ids],dtype=float)
    assert yy.size>=1
    outcome_variance=np.array([np.mean((np.asarray(replies[iid])-item_mean[k])**2) for k,iid in enumerate(evaluation_ids)])
    out={}
    for budget in BUDGETS:
        label_items=acquired[:budget]
        successes=sum(y for iid,y in label_items)
        raw=(1+successes)/(2+budget)
        delta=text_deltas_for_eval(label_items,evaluation_ids,itemdata,mod14) if budget>=15 else np.zeros(len(evaluation_ids))
        for name,shrink in [('V14C',0.05),('V19A',0.075)]:
            pred=np.clip((1-shrink)*raw+shrink*0.5+delta,0.001,0.999)
            losses=(pred-item_mean)**2+outcome_variance
            score=float(np.average(losses,weights=item_weight))
            out[(name,budget)]=score
    return out, len(evaluation_ids), yy.size, sum(1 for iid in evaluation_ids if itemdata[iid][2] is not None)

def write_csv(path,columns,rows):
    with open(path,'w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');w.writeheader();w.writerows(rows)

def run(args):
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    mod14,h14=load_zip_model(args.v14,'v14c_imported',out)
    mod19,h19=load_zip_model(args.v19,'v19a_imported',out)
    # Verify that the ONLY source-code difference is shrink weights; block on any other change.
    s14=(out/'v14c_imported.py').read_text();s19=(out/'v19a_imported.py').read_text()
    normalized19=s19.replace('p = 0.925 * raw + 0.075 * 0.5','p = 0.95 * raw + 0.05 * 0.5')
    if normalized19 != s14:
        raise RuntimeError('Code difference is not exclusively shrink strength. Stop paired test.')
    groups,itemdata,profiles=extract_data(args.db,mod14)
    group_counts=collections.Counter(k[0] for k in groups)
    rows=[]
    for seed in args.seeds:
        for k,replies in sorted(groups.items()):
            res,n_eval,n_trials,n_feature=group_seed_scores(*k,replies,itemdata,mod14,mod19,seed)
            row={'seed':seed,'benchmark':k[0],'subject_id':k[1],
                 'num_distinct_items':len(replies),'eval_unique_items':n_eval,'eval_response_trials':n_trials,'eval_text_feature_valid':n_feature}
            for model in ('V14C','V19A'):
                for budget in BUDGETS:row[f'{model}_Brier_{budget}']=res[(model,budget)]
                row[f'{model}_ALC']=sum(res[(model,n)]*w for n,w in zip(BUDGETS,ALC_WEIGHTS))
            for budget in BUDGETS:
                row[f'diff_19minus14_{budget}']=row[f'V19A_Brier_{budget}']-row[f'V14C_Brier_{budget}']
            row['diff_19minus14_ALC']=row['V19A_ALC']-row['V14C_ALC']
            rows.append(row)
    fields=list(rows[0]);write_csv(out/'paired_results_all_seeds.csv',fields,rows)
    means={}
    for model in ('V14C','V19A'):
        means[model]={f'Brier_{n}':statistics.mean(r[f'{model}_Brier_{n}'] for r in rows) for n in BUDGETS}
        means[model]['ALC']=statistics.mean(r[f'{model}_ALC'] for r in rows)
    mean_by_group=collections.defaultdict(list)
    for r in rows:mean_by_group[(r['benchmark'],r['subject_id'])].append(r)
    per_group=[]
    for (bench,sid),items in sorted(mean_by_group.items()):
        d={'benchmark':bench,'subject_id':sid,'replicates':len(items),'unique_items':items[0]['num_distinct_items']}
        for name in ('V14C','V19A'):
            d[name+'_ALC']=statistics.mean(r[name+'_ALC'] for r in items)
        d['diff_19minus14_ALC']=d['V19A_ALC']-d['V14C_ALC']
        for n in BUDGETS:
            d[f'diff_19minus14_{n}']=statistics.mean(r[f'diff_19minus14_{n}'] for r in items)
        per_group.append(d)
    write_csv(out/'paired_summary_by_subject.csv',list(per_group[0]),per_group)
    bench_summaries=[]
    for bench in BENCHMARKS:
        r0=[r for r in rows if r['benchmark']==bench]
        if r0:
            bench_summaries.append({'benchmark':bench,'groups':group_counts[bench],
                                    'V14C_ALC':statistics.mean(r['V14C_ALC'] for r in r0),
                                    'V19A_ALC':statistics.mean(r['V19A_ALC'] for r in r0),
                                    'diff_19minus14_ALC':statistics.mean(r['diff_19minus14_ALC'] for r in r0),
                                    'mean_eval_trials':statistics.mean(r['eval_response_trials'] for r in r0)})
    write_csv(out/'paired_summary_by_benchmark.csv',list(bench_summaries[0]),bench_summaries)
    # Group bootstrap, paired per subject-benchmark and averaged over seeds.
    paired_diffs=np.array([r['diff_19minus14_ALC'] for r in per_group])
    boot_rng=np.random.default_rng(20261103)
    N=len(paired_diffs)
    boot=np.array([paired_diffs[boot_rng.integers(0,N,N)].mean() for _ in range(5000)])
    ci=np.quantile(boot,[0.025,0.975])
    # Benchmark-level equal-group weighting is the reported ALC, but across-benchmark
    # correlation means this group bootstrap CI should not be taken as a definitive claim.
    seedsummaries=[]
    for seed in args.seeds:
        subset=[r for r in rows if r['seed']==seed]
        seedsummaries.append({'seed':seed,'V14C_ALC':statistics.mean(r['V14C_ALC'] for r in subset),
                             'V19A_ALC':statistics.mean(r['V19A_ALC'] for r in subset),
                             'diff_19minus14_ALC':statistics.mean(r['diff_19minus14_ALC'] for r in subset)})
    write_csv(out/'paired_summary_by_seed.csv',list(seedsummaries[0]),seedsummaries)
    result={'source_db':str(args.db),'v14_sha256':h14,'v19_sha256':h19,
      'same_code_except_shrink':True,'method':'fixed 50/50 unique-item split, one random binary acquisition trial per unique item, all binary evaluation trials; matched sampling; old local database',
      'excluded':['mmdocrag: mixed continuous score scale, 0/1 subset could be misleading'],
      'groups_by_benchmark':dict(group_counts),'seed_values':args.seeds,'paired_groups':len(group_counts) and len(groups),
      'seed_group_rows':len(rows),'scores':means,
      'delta_ALC_V19_minus_V14':means['V19A']['ALC']-means['V14C']['ALC'],
      'delta_budget_V19_minus_V14':{str(n):means['V19A'][f'Brier_{n}']-means['V14C'][f'Brier_{n}'] for n in BUDGETS},
      'group_bootstrap_diff_95pct':ci.tolist(),
      'groups_V19_better':sum(r['diff_19minus14_ALC']<-1e-10 for r in per_group),
      'groups_V14_better':sum(r['diff_19minus14_ALC']>1e-10 for r in per_group),
      'groups_equal':sum(abs(r['diff_19minus14_ALC'])<=1e-10 for r in per_group),
      'benchmarks':bench_summaries,'seed_summaries':seedsummaries,
      'caveat':'NOT official benchmark; missing subject configuration metadata; heterogeneous repeated response trials; incomplete text; same named benchmarks; not permission-verified; score aggregation/splits approximated.'}
    (out/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    # Verify expected baseline and numerical equivalence of vectorization on sampled calls.
    # Run sampled direct predictions, across every budget, including 15 and 31 text branch.
    tested=0; max_diff=0
    for seed in args.seeds[:1]:
        for bench,sid in sorted(groups)[:3]:
            ids=sorted(groups[(bench,sid)])
            rng=random.Random(int(hashlib.sha256(f'{seed}|{bench}|{sid}'.encode()).hexdigest()[:16],16))
            rng.shuffle(ids)
            n_acq=len(ids)//2; acq=ids[:n_acq]; eval_ids=ids[n_acq:]
            obs=[(iid,rng.choice(groups[(bench,sid)][iid])) for iid in acq[:31]]
            subject=profiles[(bench,sid)]
            for budget in BUDGETS:
                labels=[([subject,{'benchmark_id':bench,'item_content':itemdata[iid][1]}],y) for iid,y in obs[:budget]]
                for iid in eval_ids[:4]:
                    inp=[subject,{'benchmark_id':bench,'item_content':itemdata[iid][1]}]
                    direct14=mod14.predict(inp,labels);direct19=mod19.predict(inp,labels)
                    raw=(1+sum(y for _,y in obs[:budget]))/(2+budget)
                    txt=mod14._text_adjustment(inp,labels) if budget>=15 else 0
                    f14=max(0.001,min(.999,.95*raw+.05*.5+txt))
                    f19=max(0.001,min(.999,.925*raw+.075*.5+txt))
                    max_diff=max(max_diff,abs(direct14-f14),abs(direct19-f19));tested+=1
    result['direct_model_spotchecks']=tested
    result['max_spotcheck_discrepancy']=max_diff
    (out/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({k:result[k] for k in ('groups_by_benchmark','paired_groups','seed_group_rows','scores','delta_ALC_V19_minus_V14','group_bootstrap_diff_95pct','groups_V19_better','groups_V14_better','direct_model_spotchecks','max_spotcheck_discrepancy','benchmarks','seed_summaries')},ensure_ascii=False,indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--db',default='/mnt/data/files/paiec_old_sixbench.sqlite')
    ap.add_argument('--v14',default='/mnt/data/V14C_F_online_text_ridge.zip')
    ap.add_argument('--v19',default='/mnt/data/V19A_v14c_shrink075.zip')
    ap.add_argument('--out',default='/mnt/data/PAIEC_fixed_validation')
    ap.add_argument('--seeds',default='101,202,303,404,505',type=lambda s:[int(x) for x in s.split(',')])
    run(ap.parse_args())
