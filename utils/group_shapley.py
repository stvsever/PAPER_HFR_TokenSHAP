# group_shapley.py
from __future__ import annotations
from typing import Callable, Dict, List, Optional, Set, Tuple, Any
import random
import math
import time


def _eta(elapsed: float, done: int, total: int) -> float:
    if done <= 0:
        return float("inf")
    rate = elapsed / done
    return rate * max(total - done, 0)


def monte_carlo_group_shapley(
    score_fn: Callable[[Set[str]], float],
    feature_ids: List[str],
    groups: Optional[Dict[str, List[str]]] = None,
    K: int = 10,
    seed: int = 0,
    verbose: bool = False,
    progress_cb: Optional[Callable[[int, int, float, float], Any]] = None,
) -> Dict[str, float]:
    """
    K = number of random permutations.
    progress_cb(perm_idx, K, elapsed_sec, eta_sec) if provided.
    """
    if K < 1:
        raise ValueError("K must be >= 1")

    if groups is None:
        groups = {fid: [fid] for fid in feature_ids}

    group_ids = list(groups.keys())
    phi = {gid: 0.0 for gid in group_ids}
    rng = random.Random(seed)

    t0 = time.perf_counter()

    for k in range(K):
        perm = group_ids[:]
        rng.shuffle(perm)

        active: Set[str] = set()
        s_prev = score_fn(active)

        if verbose:
            print(f"  [Shapley] perm {k+1:02d}/{K} start s={s_prev:+.4f}")

        for gid in perm:
            active.update(groups[gid])
            s_new = score_fn(active)
            phi[gid] += (s_new - s_prev)
            s_prev = s_new

        if progress_cb is not None:
            elapsed = time.perf_counter() - t0
            eta = _eta(elapsed, k + 1, K)
            progress_cb(k + 1, K, elapsed, eta)

    for gid in phi:
        phi[gid] /= K

    return phi


def shapley_with_repeats(
    score_fn: Callable[[Set[str]], float],
    feature_ids: List[str],
    groups: Optional[Dict[str, List[str]]] = None,
    K: int = 10,
    runs: int = 3,
    seed: int = 0,
    verbose: bool = False,
    progress_cb: Optional[Callable[[int, int, int, int, float, float], Any]] = None,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Repeat Shapley estimator multiple times.

    progress_cb(run_idx, runs, perm_done, K, elapsed_sec, eta_sec) if provided.
    """
    if runs < 1:
        raise ValueError("runs must be >= 1")

    all_phis: List[Dict[str, float]] = []
    t0 = time.perf_counter()

    for r in range(runs):
        def perm_progress(perm_done: int, K_: int, elapsed_perm: float, eta_perm: float):
            if progress_cb is None:
                return
            # overall ETA: use elapsed total vs completed runs/perm fraction
            elapsed_total = time.perf_counter() - t0
            total_work = runs * K
            done_work = r * K + perm_done
            eta_total = _eta(elapsed_total, done_work, total_work)
            progress_cb(r + 1, runs, perm_done, K_, elapsed_total, eta_total)

        phi_r = monte_carlo_group_shapley(
            score_fn=score_fn,
            feature_ids=feature_ids,
            groups=groups,
            K=K,
            seed=seed + 9973 * r,
            verbose=verbose,
            progress_cb=perm_progress,
        )
        all_phis.append(phi_r)

    keys = list(all_phis[0].keys())
    mean_phi: Dict[str, float] = {k: 0.0 for k in keys}
    std_phi: Dict[str, float] = {k: 0.0 for k in keys}

    for k in keys:
        vals = [d[k] for d in all_phis]
        m = sum(vals) / len(vals)
        mean_phi[k] = m
        if len(vals) == 1:
            std_phi[k] = 0.0
        else:
            var = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
            std_phi[k] = math.sqrt(var)

    return mean_phi, std_phi
