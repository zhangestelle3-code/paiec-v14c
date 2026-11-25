"""Synthetic checks only; no benchmark data and no performance evaluation."""
import hashlib
import json
import math
import pathlib
import random
import sys
import model

EXPECTED = '700673cc9ea395fe72d839a52015440701823842cf57d51779428de93839b904'
SUBJECT = {'normalized_name': 'synthetic-model', 'provider': 'test', 'harness': 'test', 'harness_version': '1', 'reasoning_effort': 'low'}
def item(i=0, benchmark='synthetic'):
    return {'benchmark_id': benchmark, 'item_content': 'Example input output constraint function: ' + 'word ' * (i+5) + str(i) + '?'}
def row(i,y):
    return [[dict(SUBJECT), item(i)], y]
def base(successes, count):
    return .95 * (successes+1)/(count+2) + .025

def main():
    checks=[]
    def record(name, condition):
        if not condition: raise AssertionError(name)
        checks.append(name)
    record('frozen_model_hash', hashlib.sha256(pathlib.Path(model.__file__).read_bytes()).hexdigest()==EXPECTED)
    target=[SUBJECT,item(12)]
    record('cold_start',model.predict(target,[])==.5)
    record('one_success',abs(model.predict(target,[row(0,1)])-base(1,1))<1e-12)
    record('one_failure',abs(model.predict(target,[row(0,0)])-base(0,1))<1e-12)
    other=[[dict(SUBJECT),item(0,'different')],1]
    record('other_benchmark_excluded',model.predict(target,[other])==.5)
    different=[[dict(SUBJECT,normalized_name='other'),item(0)],1]
    record('other_named_subject_excluded',model.predict(target,[different])==.5)
    record('malformed_rows_skipped',model.predict(target,[None,[],['bad'],row(0, float('nan')),row(1,2)])==.5)
    record('below_text_threshold',model._text_adjustment(target,[row(i,i%2) for i in range(14)])==0)
    labels=[row(i,int(i>=15)) for i in range(31)]
    delta=model._text_adjustment([SUBJECT,item(23)],labels)
    record('text_branch_exercised',abs(delta)>1e-8)
    record('text_correction_bounded',abs(delta)<=.05)
    p=model.predict([SUBJECT,item(23)],labels)
    record('finite_probability',math.isfinite(p) and .001<=p<=.999)
    shuffled=list(labels);random.Random(2026).shuffle(shuffled)
    record('evidence_order_stability',abs(p-model.predict([SUBJECT,item(23)],shuffled))<1e-12)
    record('repeatability',p==model.predict([SUBJECT,item(23)],labels))
    record('redacted_text_fallback',model._text_adjustment([SUBJECT,{'benchmark_id':'synthetic','item_content':'[redacted] '*10}],labels)==0)
    # Document the known matching limitation rather than concealing it.
    record('known_extra_metadata_collision',model._same_subject(dict(SUBJECT,subject_features_extra='a'),dict(SUBJECT,subject_features_extra='b')))
    result={'scope':'synthetic interface checks, not empirical performance validation','python':sys.version.split()[0],'passed':len(checks),'checks':checks,'synthetic_example_probability':p,'synthetic_text_adjustment':delta}
    pathlib.Path('interface_check_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
