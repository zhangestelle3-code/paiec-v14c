"""V14C_F_online_text_ridge: exploratory PAIEC candidate.
Parent: frozen V13F. No offline-fitted weights or external dependencies.
"""
import math
import re

PRIOR_A = 1.0
PRIOR_B = 1.0


def _same_subject(left, right):
    if left == right:
        return True
    if not isinstance(left, dict) or not isinstance(right, dict):
        return False
    # Only compare documented system metadata, never protected benchmark identities.
    keys = ('normalized_name', 'provider', 'harness', 'harness_version', 'reasoning_effort')
    a = tuple(str(left.get(k, '') or '') for k in keys)
    b = tuple(str(right.get(k, '') or '') for k in keys)
    return any(a) and a == b


def predict(input: list[dict], labeled: list | None = None) -> float:
    """Predict P(success) using same-system, same-benchmark revealed binary labels.

    Interface: official PAIEC predict([subject, item], labeled) contract.
    """
    subject, item = input
    if not isinstance(item, dict):
        raise TypeError('item must be a dict')
    target_benchmark = item.get('benchmark_id')
    successes = 0
    total = 0
    for entry in labeled or []:
        try:
            pair, outcome = entry
            other_subject, other_item = pair
        except (TypeError, ValueError):
            continue
        if not isinstance(other_item, dict):
            continue
        if target_benchmark is None or other_item.get('benchmark_id') != target_benchmark:
            continue
        if not _same_subject(subject, other_subject):
            continue
        if not (isinstance(outcome, (int, float)) and math.isfinite(outcome) and outcome in (0, 1)):
            continue
        successes += int(outcome)
        total += 1
    # Smoothed Bernoulli posterior mean. Future system-capability priors require
    # legitimate, reproducible training on explicitly allowed sources.
    raw = (PRIOR_A + successes) / (PRIOR_A + PRIOR_B + total)
    p = 0.95 * raw + 0.05 * 0.5
    p += _text_adjustment(input, labeled)
    return float(max(0.001, min(0.999, p)))

# Structural branch is learned only from this call's acquired evidence.
# These are declared exploration settings, NOT offline-fitted parameters.
MIN_TEXT_LABELS = 15
RIDGE = 30.0
MAX_ADJUSTMENT = 0.05
MAX_TOTAL_TEXT_CHARS = 200000

def _features(item):
    text = item.get('item_content', '')
    if not isinstance(text, str) or not 20 <= len(text) <= 50000:
        return None
    lo = text.lower().strip()
    if lo in {'placeholder', 'not available', 'n/a', '[redacted]', '[placeholder]'}:
        return None
    if any(t in lo for t in ('[placeholder]', '[redacted]', 'content unavailable')):
        return None
    if sum(c == '\ufffd' or (ord(c) < 32 and c not in '\n\r\t') for c in text) / len(text) > 0.01:
        return None
    n = len(text)
    words = re.findall(r'[A-Za-z]+', text)
    return [math.log1p(n) / 12.0,
            math.log1p(text.count('```') / 2.0) / 5.0,
            math.log1p(text.count('$')) / 5.0,
            sum(c.isdigit() for c in text) / n,
            sum(c.isupper() for c in text) / n,
            text.count(' ') / n, text.count('\n') / n,
            sum(c in '.,;:!?()[]{}' for c in text) / n,
            sum(map(len, words)) / max(len(words), 1) / 15.0,
            *[float(bool(re.search(r'\b' + word + r'\b', lo)))
              for word in ('input', 'output', 'example', 'constraint')],
            float(bool(re.search(r'\b(def|function|class)\b', lo))),
            text.count('?') / n]

def _solve_spd(matrix, rhs):
    # Cholesky; ridge makes the symmetric matrix positive definite.
    d = len(rhs)
    L = [[0.0] * d for _ in range(d)]
    for i in range(d):
        for j in range(i + 1):
            value = matrix[i][j] - math.fsum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                if value <= 0 or not math.isfinite(value):
                    return None
                L[i][j] = math.sqrt(value)
            else:
                L[i][j] = value / L[j][j]
    z = [0.0] * d
    for i in range(d):
        z[i] = (rhs[i] - math.fsum(L[i][j] * z[j] for j in range(i))) / L[i][i]
    answer = [0.0] * d
    for i in range(d - 1, -1, -1):
        answer[i] = (z[i] - math.fsum(L[j][i] * answer[j] for j in range(i + 1, d))) / L[i][i]
    return answer

def _text_adjustment(input, labeled):
    subject, item = input
    target = _features(item)
    benchmark = item.get('benchmark_id')
    if target is None or not isinstance(benchmark, str) or not benchmark:
        return 0.0
    records = []
    total = 0
    text_chars = len(item.get('item_content', ''))
    for row in labeled or []:
        try:
            (other_subject, other_item), y = row
        except (TypeError, ValueError):
            continue
        if not isinstance(other_item, dict) or other_item.get('benchmark_id') != benchmark:
            continue
        if not _same_subject(subject, other_subject):
            continue
        if not (isinstance(y, (int, float)) and math.isfinite(y) and y in (0, 1)):
            continue
        total += 1
        text = other_item.get('item_content', '')
        if isinstance(text, str):
            text_chars += len(text)
        if text_chars > MAX_TOTAL_TEXT_CHARS:
            return 0.0
        x = _features(other_item)
        if x is not None:
            records.append((tuple(x), float(y)))
    if len(records) < MIN_TEXT_LABELS or len(records) < 0.8 * total:
        return 0.0
    # Fixed sorting avoids material dependence on evidence ordering.
    records.sort()
    n = len(records); d = len(target)
    means = [math.fsum(x[j] for x, _ in records) / n for j in range(d)]
    scales = [max(0.05, math.sqrt(math.fsum((x[j] - means[j]) ** 2 for x, _ in records) / n)) for j in range(d)]
    for j in range(d):
        # Simple support check; this is not a calibrated OOD confidence model.
        if not min(x[j] for x, _ in records) - 0.5 * scales[j] <= target[j] <= max(x[j] for x, _ in records) + 0.5 * scales[j]:
            return 0.0
    q = [(target[j] - means[j]) / scales[j] for j in range(d)]
    if max(map(abs, q)) > 3.0:
        return 0.0
    Z = [[(x[j] - means[j]) / scales[j] for j in range(d)] for x, _ in records]
    y_mean = math.fsum(y for _, y in records) / n
    matrix = [[math.fsum(z[i] * z[j] for z in Z) + (RIDGE if i == j else 0.0) for j in range(d)] for i in range(d)]
    rhs = [math.fsum(Z[k][j] * (records[k][1] - y_mean) for k in range(n)) for j in range(d)]
    beta = _solve_spd(matrix, rhs)
    if beta is None:
        return 0.0
    delta = math.fsum(q[j] * beta[j] for j in range(d))
    if not math.isfinite(delta):
        return 0.0
    return max(-MAX_ADJUSTMENT, min(MAX_ADJUSTMENT, delta))
