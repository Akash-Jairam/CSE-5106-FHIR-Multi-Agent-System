"""Experiment statistics (P3-21): patient-clustered bootstrap CIs, paired differences, and latency.

Questions from one patient share a record, so they are not independent.
Every interval resamples patients with replacement, keeping all of a
resampled patient's questions, and reports the 2.5th and 97.5th
percentiles. A paired difference compares two arms on the same questions:
each resample takes the same patients for both, so between-question
variation cancels.

All resamples come from one seeded generator per call, so the results are
reproducible. Inputs are per-question arrays; nothing here sees answers.
"""
import numpy as np

DEFAULT_BOOTSTRAP = 10_000
LEVEL = 0.95


def _patient_totals(values, patients):
    """(per-patient sums of `values`, per-patient question counts)."""
    _, index = np.unique(np.asarray(patients, dtype=object).astype(str), return_inverse=True)
    values = np.asarray(values, dtype=float)
    return np.bincount(index, weights=values), np.bincount(index).astype(float)


def _resampled_means(values, patients, n_boot, seed):
    sums, counts = _patient_totals(values, patients)
    rng = np.random.default_rng(seed)
    # Each row: how many times each patient is drawn in one resample of len(patients) draws
    weights = rng.multinomial(len(counts), np.full(len(counts), 1 / len(counts)), size=n_boot)
    return weights @ sums / (weights @ counts)


def clustered_mean(values, patients, n_boot=DEFAULT_BOOTSTRAP, seed=0):
    """Mean with a patient-clustered percentile bootstrap CI."""
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return {"n": 0, "patients": 0, "estimate": None, "ci_low": None, "ci_high": None}
    boot = _resampled_means(values, patients, n_boot, seed)
    tail = (1 - LEVEL) / 2 * 100
    low, high = np.percentile(boot, [tail, 100 - tail])
    return {"n": int(len(values)), "patients": int(len(set(map(str, patients)))),
            "estimate": float(values.mean()), "ci_low": float(low), "ci_high": float(high)}


def paired_difference(a, b, patients, n_boot=DEFAULT_BOOTSTRAP, seed=0):
    """Mean of a - b over the same questions, with a patient-clustered CI.

    `p_bootstrap` is the two-sided share of resamples on the other side of 0
    from the estimate (a bootstrap p-value; 0 means below 1 / n_boot).
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("A paired difference needs both arms on the same questions")
    result = clustered_mean(a - b, patients, n_boot, seed)
    if result["n"]:
        boot = _resampled_means(a - b, patients, n_boot, seed)
        result["p_bootstrap"] = float(min(1.0, 2 * min((boot <= 0).mean(), (boot >= 0).mean())))
    return result


def latency_summary(latencies):
    """Median and 95th percentile of per-question latency in seconds."""
    values = np.asarray([v for v in latencies if v is not None], dtype=float)
    if len(values) == 0:
        return {"n": 0, "median": None, "p95": None}
    return {"n": int(len(values)), "median": float(np.median(values)), "p95": float(np.percentile(values, 95))}
