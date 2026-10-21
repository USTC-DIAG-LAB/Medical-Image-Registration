"""Sinkhorn transport solver adapted from Marco Cuturi's Sinkhorn Transport.

Original implementation: Copyright (c) Marco Cuturi 2013, 2014.
Source: https://marcocuturi.net/Code/sinkhornTransport.m
Python adaptation distributed with FOTMIR, 2026.
Modifications: NumPy array operations, input validation and numerical-error
handling. The original fixed-point and counter conventions are retained.
See ../../THIRD_PARTY_NOTICES.md for provenance and adaptation history.

The contents of this file are subject to the Mozilla Public License Version
1.1 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at
https://www.mozilla.org/MPL/1.1/ and in ../../Licenses/MPL-1.1.txt.

Software distributed under the License is distributed on an "AS IS" basis,
WITHOUT WARRANTY OF ANY KIND, either express or implied. See the License
for the specific language governing rights and limitations under the License.

The Original Code is Sinkhorn Transport, (C) 2013, Marco Cuturi.
The Initial Developer of the Original Code is Marco Cuturi,
mcuturi@i.kyoto-u.ac.jp. Portions created by the Initial Developer are
Copyright (C) 2013 the Initial Developer. All Rights Reserved.
"""
import numpy as np


def _columns(value):
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2:
        raise ValueError("Histograms must be vectors or column-oriented matrices.")
    return array


def Transport(a, b, K, U, lambda_, stoppingCriterion="marginalDifference",
              p_norm=np.inf, tolerance=0.005, maxIter=5000, VERBOSE=0):
    """Return D, L, u, v; supports both one-vs-N and paired-histogram modes.

    Retains the source's additional counter increment after convergence checks.
    Numerical failure raises an error instead of exporting NaN results.
    """
    a, b = _columns(a), _columns(b)
    K, U = np.asarray(K, dtype=np.float64), np.asarray(U, dtype=np.float64)
    if K.shape != (a.shape[0], b.shape[0]) or U.shape != K.shape:
        raise ValueError("K/U dimensions must agree with a and b.")
    if a.shape[1] not in (1, b.shape[1]):
        raise ValueError("a must have one column or the same column count as b.")
    if any(not np.all(np.isfinite(x)) for x in (a, b, K, U)):
        raise ValueError("Transport inputs must be finite.")
    if any(np.any(x < 0) for x in (a, b, K, U)):
        raise ValueError("Transport inputs must be nonnegative.")
    if np.any(a.sum(axis=0) <= 0) or not np.allclose(a.sum(axis=0), b.sum(axis=0), rtol=1e-12, atol=1e-14):
        raise ValueError("Each pair of histograms must have equal, positive mass.")
    if not np.isfinite(lambda_) or lambda_ <= 0 or not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("lambda_ and tolerance must be positive and finite.")
    if maxIter < 1 or int(maxIter) != maxIter:
        raise ValueError("maxIter must be a positive integer.")
    if stoppingCriterion not in ("marginalDifference", "distanceRelativeDecrease"):
        raise ValueError("Stopping criterion not recognized.")
    one_vs_n = a.shape[1] == 1
    big_n = b.shape[1] > b.shape[0]
    support = np.ones(a.shape[0], dtype=bool)
    if one_vs_n:
        support = a[:, 0] > 0
        K, U, a = K[support], U[support], a[support]
        ainvK = K / a
    u = np.ones((a.shape[0], b.shape[1]), dtype=np.float64) / a.shape[0]
    old_distance = np.ones(b.shape[1])
    counter = 0
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        while counter < maxIter:
            if one_vs_n:
                denominator = K.T @ u if big_n else (u.T @ K).T
                u = 1.0 / (ainvK @ (b / denominator))
            else:
                denominator = (u.T @ K).T if big_n else K.T @ u
                u = a / (K @ (b / denominator))
            counter += 1
            if counter % 20 == 1 or counter == maxIter:
                denominator = K.T @ u if big_n else (u.T @ K).T
                v = b / denominator
                u = 1.0 / (ainvK @ v) if one_vs_n else a / (K @ v)
                if stoppingCriterion == "distanceRelativeDecrease":
                    D = np.sum(u * (U @ v), axis=0)
                    criterion = np.linalg.norm(D / old_distance - 1, ord=p_norm)
                    old_distance = D
                else:
                    criterion = np.linalg.norm(np.sum(np.abs(v * (K.T @ u) - b), axis=0), ord=p_norm)
                if not np.isfinite(criterion) or not np.all(np.isfinite(u)) or not np.all(np.isfinite(v)):
                    raise FloatingPointError("Sinkhorn scaling became nonfinite; check point degeneracy and regularization.")
                if criterion < tolerance:
                    break
                counter += 1
                if VERBOSE:
                    print(f"Iteration: {counter}, criterion: {criterion:g}")
        if stoppingCriterion == "marginalDifference":
            D = np.sum(u * (U @ v), axis=0)
        alpha, beta = np.log(u), np.log(v)
        beta[np.isneginf(beta)] = 0
        if one_vs_n:
            L = (a.T @ alpha).reshape(-1) + np.sum(b * beta, axis=0)
        else:
            alpha[np.isneginf(alpha)] = 0
            L = np.sum(a * alpha, axis=0) + np.sum(b * beta, axis=0)
        L = L / lambda_
    if not all(np.all(np.isfinite(x)) for x in (D, L, u, v)):
        raise FloatingPointError("Transport produced nonfinite outputs.")
    if one_vs_n and not np.all(support):
        restored = np.zeros((support.size, b.shape[1]))
        restored[support] = u
        u = restored
    return D, L, u, v
