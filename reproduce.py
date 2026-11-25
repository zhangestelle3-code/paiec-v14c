"""Audit archived tables or regenerate predictions from the exact historical input."""
import argparse,csv,hashlib,json,pathlib,platform,statistics,subprocess,sys,time,zipfile
ROOT=pathlib.Path(__file__).resolve().parent
METHODS=('NoText','V14C','Length1D')
BUDGETS=(0,1,3,7,15,31)
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write_csv(p,rows):
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['audit','full'],default='audit')
    p.add_argument('--db',type=pathlib.Path)
    p.add_argument('--out',type=pathlib.Path,default=ROOT/'run_results')
    a=p.parse_args()
    if a.mode=='full' and (a.db is None or not a.db.is_file()):p.error('full mode needs --db pointing to the exact historical SQLite; audit mode needs no database')
    out=a.out.resolve();out.mkdir(parents=True,exist_ok=True)
    report={'mode':a.mode,'scope':'archived-loss arithmetic only' if a.mode=='audit' else 'prediction regeneration from exact historical database','python':platform.python_version(),'passed':False,'hashes':{}}
    start=time.monotonic()
    try:
        expected=json.loads((ROOT/'reproduction_integrity.json').read_text())
        for name,value in expected.items():
            actual=digest(ROOT/name)
            if actual!=value:raise ValueError('Packaged source or loss rows changed: '+name)
            report['hashes'][name]=actual
        with zipfile.ZipFile(ROOT/'frozen_submission.zip') as z:
            if z.read('model.py')!=(ROOT/'model.py').read_bytes():raise ValueError('Frozen ZIP and predictor disagree')
        with (out/'interface.log').open('w') as f:
            subprocess.run([sys.executable,str(ROOT/'check_interface.py')],cwd=out,stdout=f,stderr=subprocess.STDOUT,check=True)
        data=ROOT/'historical_metrics'
        if a.mode=='full':
            data=out/'full'
            with (out/'prediction_run.log').open('w') as f:
                subprocess.run([sys.executable,str(ROOT/'rerun_recovered.py'),'--db',str(a.db.resolve()),'--output',str(data)],cwd=out,stdout=f,stderr=subprocess.STDOUT,check=True)
        from audit_historical import audit
        audit_results={}
        table=[];budget_table=[];benchmark_table=[]
        reference=json.loads((ROOT/'rerun_results/reproduction_results.json').read_text())
        for split in ['development','confirmation']:
            src=data/(split+'_paired_rows.csv')
            result=audit(src);audit_results[split]=result
            for m in METHODS:
                if abs(result['means'][m]-reference['runs'][split]['means'][m])>1e-10:raise ValueError('Reference aggregate mismatch')
                table.append({'split':split,'method':m,'ALC':result['means'][m]})
            rows=list(csv.DictReader(src.open(encoding='utf-8-sig')))
            for m in METHODS:
                for b in BUDGETS:budget_table.append({'split':split,'method':m,'label_budget':b,'Brier':statistics.mean(float(r[f'{m}_Brier_{b}']) for r in rows)})
            for bench,values in result['by_benchmark'].items():
                for m,v in values.items():benchmark_table.append({'split':split,'benchmark':bench,'method':m,'ALC':v})
        write_csv(out/'main_table.csv',table);write_csv(out/'budget_table.csv',budget_table);write_csv(out/'benchmark_table.csv',benchmark_table)
        (out/'audit_results.json').write_text(json.dumps(audit_results,indent=2)+'\n')
        report['passed']=True
        print('PASS:',report['scope'],'; outputs:',out)
    except Exception as exc:
        report['error']=str(exc)
        raise
    finally:
        report['elapsed_seconds']=time.monotonic()-start
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
