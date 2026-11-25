"""Reproduce three historical local controls from the recovered SQLite.

Requires NumPy. This is a local diagnostic, not the official scoring harness.
The database is a private local input; no raw texts or labels are exported.
"""
import argparse
import collections
import csv
import hashlib
import importlib.util
import json
import platform
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'historical_sources'))
from run_fixed_test import extract_data, text_deltas_for_eval

BUDGETS = (0, 1, 3, 7, 15, 31)
WEIGHTS = (.1, .2, .2, .2, .2, .1)
METHODS = ('NoText', 'V14C', 'Length1D')
EXPECTED_DB = '0062957a600ce37717f6b0d2a2030423c79a9b89ef779029984a3f7bde6953a5'


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def length_delta(observed, targets, items):
    valid = [(items[i][2][0], float(y)) for i, y in observed if items[i][2] is not None]
    result = np.zeros(len(targets))
    if len(valid) < 7 or len(valid) < .8 * len(observed):
        return result
    x = np.array([v[0] for v in valid])
    y = np.array([v[1] for v in valid])
    mean, scale = float(x.mean()), max(float(x.std()), .05)
    z = (x - mean) / scale
    beta = float(z @ (y - y.mean()) / (z @ z + 15.))
    for j, item in enumerate(targets):
        f = items[item][2]
        if f is None or not x.min() - .5 * scale <= f[0] <= x.max() + .5 * scale:
            continue
        q = (f[0] - mean) / scale
        if abs(q) <= 3:
            result[j] = np.clip(q * beta, -.05, .05)
    return result


def episode(bench, sid, replies, seed):
    ids = sorted(replies)
    rng = random.Random(int(hashlib.sha256(f'{seed}|{bench}|{sid}'.encode()).hexdigest()[:16], 16))
    rng.shuffle(ids)
    acquisition, targets = ids[:len(ids)//2], ids[len(ids)//2:]
    observed = [(i, rng.choice(replies[i])) for i in acquisition[:31]]
    assert not set(acquisition) & set(targets)
    return observed, targets


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT / 'rerun_results')
    args = parser.parse_args()
    db_hash = hashlib.file_digest(args.db.open('rb'), 'sha256').hexdigest()
    if db_hash != EXPECTED_DB:
        raise ValueError('Database differs from recovered historical input; do not overwrite this reproduction.')
    args.output.mkdir(parents=True, exist_ok=True)
    frozen = load_module(ROOT / 'model.py', 'frozen_v14c')
    length = load_module(ROOT / 'historical_sources/historical_v20a_model.py', 'historical_length')
    groups, items, profiles = extract_data(args.db.resolve(), frozen)
    if len(groups) != 221:
        raise ValueError('Expected 221 historical subject-benchmark groups')
    group_ids = {k: f'G{i:04d}' for i, k in enumerate(sorted(groups), 1)}
    # Independently call original predictor functions on real-data examples.
    # Select two groups per benchmark, both inactive and active text corrections.
    parity = {'checks': 0, 'max_abs_error': 0., 'nonzero_v14c': 0, 'nonzero_length': 0}
    chosen = collections.Counter()
    start = time.monotonic()
    for (bench, sid), replies in sorted(groups.items()):
        if chosen[bench] >= 2:
            continue
        chosen[bench] += 1
        obs, targets = episode(bench, sid, replies, 101)
        subject = profiles[(bench, sid)]
        for budget in BUDGETS:
            labeled = [[[subject, {'benchmark_id': bench, 'item_content': items[i][1]}], y] for i, y in obs[:budget]]
            base = .95 * (1 + sum(y for _, y in obs[:budget])) / (2 + budget) + .025
            text = text_deltas_for_eval(obs[:budget], targets, items, frozen)
            ld = length_delta(obs[:budget], targets, items)
            # Include nonzero and zero branches, rather than only first targets.
            indices = set([0, len(targets)//2, len(targets)-1])
            indices.update(int(i) for i in np.flatnonzero(abs(text) > 1e-12)[:2])
            indices.update(int(i) for i in np.flatnonzero(abs(ld) > 1e-12)[:2])
            for j in sorted(indices):
                inp = [subject, {'benchmark_id': bench, 'item_content': items[targets[j]][1]}]
                for model, delta in [(frozen, text[j]), (length, ld[j])]:
                    expected = float(np.clip(base + delta, .001, .999))
                    error = abs(model.predict(inp, labeled) - expected)
                    parity['checks'] += 1
                    parity['max_abs_error'] = max(parity['max_abs_error'], error)
                    if error > 1e-10:
                        raise ValueError('Vectorized prediction disagrees with original code')
                parity['nonzero_v14c'] += int(abs(text[j]) > 1e-12)
                parity['nonzero_length'] += int(abs(ld[j]) > 1e-12)
    if not parity['nonzero_v14c'] or not parity['nonzero_length']:
        raise ValueError('Parity checks did not exercise correction branches')
    print('Original-code parity', parity, flush=True)
    report = {'scope': 'Re-executed historical local diagnostics from recovered SQLite; not official hidden evaluation',
              'database_sha256': db_hash, 'python': platform.python_version(), 'numpy': np.__version__,
              'model_sha256': hashlib.sha256((ROOT / 'model.py').read_bytes()).hexdigest(),
              'length_model_sha256': hashlib.sha256((ROOT / 'historical_sources/historical_v20a_model.py').read_bytes()).hexdigest(),
              'groups': len(groups), 'groups_by_benchmark': dict(collections.Counter(k[0] for k in groups)),
              'parity': parity, 'runs': {}}
    for name, seeds in [('development', [101,202,303,404,505]), ('confirmation', [606,707,808,909,1010])]:
        rows = []
        for seed in seeds:
            for (bench, sid), replies in sorted(groups.items()):
                obs, targets = episode(bench, sid, replies, seed)
                weight = np.array([len(replies[i]) for i in targets], dtype=float)
                mean = np.array([statistics.mean(replies[i]) for i in targets])
                variance = np.array([np.mean((np.array(replies[i]) - m)**2) for i, m in zip(targets, mean)])
                row = {'seed': seed, 'benchmark': bench, 'group_id': group_ids[(bench, sid)]}
                for budget in BUDGETS:
                    observed = obs[:budget]
                    base = .95 * (1 + sum(y for _, y in observed)) / (2 + len(observed)) + .025
                    corrections = {'NoText': np.zeros(len(targets)),
                                   'V14C': text_deltas_for_eval(observed, targets, items, frozen),
                                   'Length1D': length_delta(observed, targets, items)}
                    for method, delta in corrections.items():
                        pred = np.clip(base + delta, .001, .999)
                        row[f'{method}_Brier_{budget}'] = float(np.average((pred - mean)**2 + variance, weights=weight))
                for method in METHODS:
                    row[f'{method}_ALC'] = sum(w * row[f'{method}_Brier_{b}'] for w, b in zip(WEIGHTS, BUDGETS))
                rows.append(row)
            print(f'{name} seed {seed} done; elapsed {time.monotonic()-start:.1f}s', flush=True)
        write_csv(args.output / f'{name}_paired_rows.csv', rows)
        # Compare each recomputed budget loss with the existing archived losses.
        archived = list(csv.DictReader((ROOT / f'historical_metrics/{name}_paired_rows.csv').open(encoding='utf-8-sig')))
        # Sorted group IDs are local replacements; align by seed/benchmark and
        # stable within-benchmark group order used by the historical derived CSV.
        old = sorted(archived, key=lambda r:(int(r['seed']),r['benchmark'],r['group_id']))
        new = sorted(rows, key=lambda r:(int(r['seed']),r['benchmark'],r['group_id']))
        if len(old) != len(new):
            raise ValueError('Archived dimensions differ')
        max_error = 0.
        for a, b in zip(old, new):
            if (int(a['seed']), a['benchmark']) != (b['seed'], b['benchmark']):
                raise ValueError('Archived group order differs')
            for key in b:
                if '_Brier_' in key or key.endswith('_ALC'):
                    max_error = max(max_error, abs(float(a[key]) - b[key]))
        if max_error > 1e-10:
            raise ValueError(f'Archived losses differ: {max_error}')
        from audit_historical import audit
        report['runs'][name] = audit(args.output / f'{name}_paired_rows.csv')
        report['runs'][name]['max_abs_difference_from_archive'] = max_error
    report['elapsed_seconds'] = time.monotonic()-start
    (args.output / 'reproduction_results.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
