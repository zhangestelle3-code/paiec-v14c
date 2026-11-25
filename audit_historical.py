"""Reaggregate archived metrics; does not rerun any predictor or dataset."""
import collections,csv,json,pathlib,random,statistics
W={0:.1,1:.2,3:.2,7:.2,15:.2,31:.1}
def audit(path):
    rows=list(csv.DictReader(path.open(encoding='utf-8-sig')))
    groups=collections.defaultdict(list)
    for r in rows:
        for m in ['V14C','NoText','Length1D']:
            computed=sum(w*float(r[f'{m}_Brier_{b}']) for b,w in W.items())
            if abs(computed-float(r[f'{m}_ALC']))>1e-12:raise ValueError('ALC mismatch')
        groups[(r['benchmark'],r['group_id'])].append(r)
    if len(rows)!=1105 or len(groups)!=221 or any(len(v)!=5 for v in groups.values()):raise ValueError('Unexpected archived dimensions')
    means={m:statistics.mean(statistics.mean(float(r[f'{m}_ALC']) for r in rr) for rr in groups.values()) for m in ['V14C','NoText','Length1D']}
    comparisons={}
    for alternative in ['NoText','Length1D']:
        d=[statistics.mean(float(r[f'{alternative}_ALC'])-float(r['V14C_ALC']) for r in rr) for rr in groups.values()]
        rng=random.Random(20261120)
        boot=sorted(statistics.mean(rng.choice(d) for _ in d) for _ in range(5000))
        comparisons[alternative+'_minus_V14C']={'difference':statistics.mean(d),'group_bootstrap_95_percent_interval':[boot[124],boot[4874]],'seed':20261120,'resamples':5000}
    by_benchmark={}
    for b in sorted({g[0] for g in groups}):
        rr=[r for r in rows if r['benchmark']==b]
        by_benchmark[b]={m:statistics.mean(float(r[f'{m}_ALC']) for r in rr) for m in means}
    return {'rows':len(rows),'groups':len(groups),'seeds':sorted({int(r['seed']) for r in rows}),'means':means,'comparisons':comparisons,'by_benchmark':by_benchmark}
if __name__=='__main__':
    root=pathlib.Path(__file__).resolve().parent
    result={'scope':'arithmetic reanalysis of archived loss rows; original predictor/data generation not rerun; provenance unresolved','development':audit(root/'historical_metrics/development_paired_rows.csv'),'confirmation':audit(root/'historical_metrics/confirmation_paired_rows.csv')}
    (root/'historical_audit_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
