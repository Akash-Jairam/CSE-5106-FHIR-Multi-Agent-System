"""Experiment statistics (P3-21): patient-clustered bootstrap CIs, paired differences, and latency.

Questions from one patient share a record, so they are not independent.
Every interval resamples patients with replacement, keeping all of a
resampled patient's questions, and reports the 2.5th and 97.5th
percentiles. The estimand is question-weighted, not the mean of patient
means. At least two patients are needed to estimate an interval.
A paired difference compares two arms on the same questions:
each resample takes the same patients for both, so between-question
variation cancels.

All resamples come from one seeded generator per call, so the results are
reproducible. Inputs are per-question arrays; nothing here sees answers.
"""
import numpy as np

DEFAULT_BOOTSTRAP = 10_000
LEVEL = 0.95


def _validated_inputs(values, patients, n_boot):
    values = np.asarray(values, dtype=float)
    patients = np.asarray(patients, dtype=object)
    if values.ndim != 1 or patients.ndim != 1 or values.shape != patients.shape:
        raise ValueError("Values and patient IDs must be one-dimensional arrays of the same length")
    if not np.isfinite(values).all():
        raise ValueError("Values must be finite")
    if any(p is None or (isinstance(p, (float, np.floating)) and not np.isfinite(p))
           or not str(p).strip() for p in patients):
        raise ValueError("Every observation needs a nonempty patient ID")
    if isinstance(n_boot, (bool, np.bool_)) or not isinstance(n_boot, (int, np.integer)) or n_boot < 1:
        raise ValueError("n_boot must be a positive integer")
    return values, patients.astype(str)


def _patient_totals(values, patients):
    """(per-patient sums of `values`, per-patient question counts)."""
    _, index = np.unique(np.asarray(patients, dtype=object).astype(str), return_inverse=True)
    values = np.asarray(values, dtype=float)
    return np.bincount(index, weights=values), np.bincount(index).astype(float)


def _resampled_means(values, patients, n_boot, seed):
    sums, counts = _patient_totals(values, patients)
    rng = np.random.default_rng(seed)
    # Draw as many clusters as there are distinct patients, preserving each cluster's size.
    weights = rng.multinomial(len(counts), np.full(len(counts), 1 / len(counts)), size=n_boot)
    return weights @ sums / (weights @ counts)


def clustered_mean(values, patients, n_boot=DEFAULT_BOOTSTRAP, seed=0):
    """Question-weighted mean with a patient-clustered percentile bootstrap CI.

    A single observed patient cannot estimate between-patient uncertainty;
    its mean is reported with unavailable interval bounds.
    """
    values, patients = _validated_inputs(values, patients, n_boot)
    if len(values) == 0:
        return {"n": 0, "patients": 0, "estimate": None, "ci_low": None, "ci_high": None}
    result = {"n": int(len(values)), "patients": int(len(set(patients))),
              "estimate": float(values.mean()), "ci_low": None, "ci_high": None}
    if result["patients"] < 2:
        return result
    boot = _resampled_means(values, patients, n_boot, seed)
    tail = (1 - LEVEL) / 2 * 100
    low, high = np.percentile(boot, [tail, 100 - tail])
    return {**result, "ci_low": float(low), "ci_high": float(high)}


def paired_difference(a, b, patients, n_boot=DEFAULT_BOOTSTRAP, seed=0):
    """Mean of a - b over the same questions, with a patient-clustered CI.

    Pair alignment is the caller's responsibility. The ordinary bootstrap
    distribution estimates uncertainty around the observed difference;
    its tail across zero is not reported as a calibrated hypothesis test.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("A paired difference needs both arms on the same questions")
    a, patients = _validated_inputs(a, patients, n_boot)
    b, _ = _validated_inputs(b, patients, n_boot)
    return clustered_mean(a - b, patients, n_boot, seed)


def latency_summary(latencies):
    """Median and 95th percentile of per-question latency in seconds."""
    values = np.asarray([v for v in latencies if v is not None], dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Observed latencies must be finite, non-negative seconds")
    if len(values) == 0:
        return {"n": 0, "median": None, "p95": None}
    return {"n": int(len(values)), "median": float(np.median(values)), "p95": float(np.percentile(values, 95))}
