# demo_run.py
from __future__ import annotations

import csv
import inspect
import json
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any

import torch

import ig_attribution as ig
import group_shapley as gs


# -----------------------
# CONFIG
# -----------------------
PHENOTYPE = "DEPRESSION"
MODEL_NAME = "Qwen/Qwen2-1.5B-Instruct"
N_PROFILES = 100  # MUST be even if you want 50/50 CASE vs CONTROL

IG_STEPS = 8

SHAPLEY_K = 5
SHAPLEY_RUNS = 3

PRINT_PROMPT_PER_SUBJECT = True
PRINT_LLM_GENERATION_PER_SUBJECT = True

# Show detailed Shapley permutation contributions? (very verbose + slow)
PRINT_SHAPLEY_VERBOSE_INTERNAL = False

# Show progress lines frequently
PROGRESS_EVERY_SEC = 2.0

EPS = 1e-9

# OUTPUT DIRS (your requested paths)
TABLES_DIR = Path(
    "/INFERENCE_PIPELINE/utils/architectures/demonstration_results/tables"
)

# Two clinically relevant features + eight distractor "word features"
RELEVANT_FEATURES = [
    "sleep_quality",
    "childhood_trauma_exposure",
]

DISTRACTOR_FEATURES = [
    "color_blue",
    "chicken_soup",
    "computer",
    "floor",
    "window",
    "pencil",
    "bicycle",
    "cloud",
]

FEATURE_IDS = RELEVANT_FEATURES + DISTRACTOR_FEATURES

# Surface text that the LLM actually sees (kept close to your examples)
SURFACE_TEXT = {
    "sleep_quality": "sleep quality",
    "childhood_trauma_exposure": "childhood trauma exposure",
    "color_blue": "color blue",
    "chicken_soup": "chicken soup",
    "computer": "computer",
    "floor": "floor",
    "window": "window",
    "pencil": "pencil",
    "bicycle": "bicycle",
    "cloud": "cloud",
}

# Ground-truth weights:
# - Relevant ~ 1.0
# - Distractors abs(weight) <= 0.2
TRUE_W: Dict[str, float] = {
    "sleep_quality": 1.0,
    "childhood_trauma_exposure": 1.0,
    "color_blue": 0.00,
    "chicken_soup": 0.05,
    "computer": -0.08,
    "floor": 0.02,
    "window": 0.00,
    "pencil": -0.12,
    "bicycle": 0.07,
    "cloud": -0.03,
}


# -----------------------
# UTIL
# -----------------------
def fmt_time(sec: float) -> str:
    if sec == float("inf"):
        return "ETA: ?"
    if sec < 60:
        return f"{sec:5.1f}s"
    m = int(sec // 60)
    s = sec - 60 * m
    if m < 60:
        return f"{m:02d}:{s:04.1f}"
    h = int(m // 60)
    m = m - 60 * h
    return f"{h:d}:{m:02d}:{s:04.1f}"


def normalize_abs_vector(vec: Dict[str, float]) -> Dict[str, float]:
    denom = sum(abs(v) for v in vec.values()) + EPS
    return {k: (abs(v) / denom) for k, v in vec.items()}


def mae_feature_importance(est: Dict[str, float], truth: Dict[str, float]) -> float:
    est_n = normalize_abs_vector(est)
    tru_n = normalize_abs_vector(truth)
    return sum(abs(est_n[k] - tru_n[k]) for k in FEATURE_IDS) / len(FEATURE_IDS)


def safe_call(fn, **kwargs):
    """
    Call fn(**kwargs), but drop kwargs it doesn't accept.
    This keeps demo_run.py compatible if your ig/group_shapley modules differ slightly.
    """
    sig = inspect.signature(fn)
    ok = {}
    for k, v in kwargs.items():
        if k in sig.parameters:
            ok[k] = v
    return fn(**ok)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _eta(elapsed: float, done: int, total: int) -> float:
    if done <= 0:
        return float("inf")
    rate = elapsed / done
    return rate * max(total - done, 0)


def shapley_with_repeats_and_store_runs(
    score_fn,
    feature_ids: List[str],
    K: int,
    runs: int,
    seed: int,
    verbose: bool,
    progress_cb=None,
) -> Tuple[Dict[str, float], Dict[str, float], List[Dict[str, float]]]:
    """
    Runs monte_carlo_group_shapley 'runs' times, returns (mean, std, per_run_phis).
    Uses the same seeding pattern as shapley_with_repeats in your group_shapley.py:
        seed + 9973 * r
    """
    if runs < 1:
        raise ValueError("runs must be >= 1")

    per_run: List[Dict[str, float]] = []
    t0 = time.perf_counter()

    total_work = runs * K

    for r in range(runs):

        def perm_progress(perm_done: int, K_: int, elapsed_perm: float, eta_perm: float):
            if progress_cb is None:
                return
            elapsed_total = time.perf_counter() - t0
            done_work = r * K + perm_done
            eta_total = _eta(elapsed_total, done_work, total_work)
            progress_cb(r + 1, runs, perm_done, K_, elapsed_total, eta_total)

        phi_r = gs.monte_carlo_group_shapley(
            score_fn=score_fn,
            feature_ids=feature_ids,
            groups=None,
            K=K,
            seed=seed + 9973 * r,
            verbose=verbose,
            progress_cb=perm_progress,
        )
        per_run.append(phi_r)

    keys = list(per_run[0].keys())
    mean_phi: Dict[str, float] = {k: 0.0 for k in keys}
    std_phi: Dict[str, float] = {k: 0.0 for k in keys}

    for k in keys:
        vals = [d[k] for d in per_run]
        m = sum(vals) / len(vals)
        mean_phi[k] = m
        if len(vals) == 1:
            std_phi[k] = 0.0
        else:
            var = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
            std_phi[k] = var ** 0.5

    return mean_phi, std_phi, per_run


# -----------------------
# DATA + GROUND TRUTH
# -----------------------
def sample_one_profile(rng: random.Random) -> Dict[str, int]:
    """
    Values in {-1,+1} for all features.
    Relevant features drive most of the ground-truth signal (weights ~ 1.0).
    Distractors have tiny weights (<=0.2 abs), so they should matter little.
    """
    p: Dict[str, int] = {}
    for fid in FEATURE_IDS:
        p[fid] = rng.choice([-1, +1])
    return p


def ground_truth_logit(profile: Dict[str, int]) -> float:
    return sum(TRUE_W[f] * float(profile[f]) for f in FEATURE_IDS)


def ground_truth_label(profile: Dict[str, int]) -> str:
    return f"{PHENOTYPE} CASE" if ground_truth_logit(profile) > 0 else f"{PHENOTYPE} CONTROL"


def make_balanced_profiles(n: int = 20, seed: int = 1, max_draws: int = 200_000) -> List[Dict[str, int]]:
    """
    IMPORTANT: does NOT return until we have EXACTLY equal CASE vs CONTROL (50/50).
    """
    if n % 2 != 0:
        raise ValueError("N_PROFILES must be even to enforce exact 50/50 CASE vs CONTROL.")

    target = n // 2
    rng = random.Random(seed)

    cases: List[Dict[str, int]] = []
    ctrls: List[Dict[str, int]] = []

    draws = 0
    while (len(cases) < target) or (len(ctrls) < target):
        draws += 1
        if draws > max_draws:
            raise RuntimeError(
                f"Failed to sample balanced dataset within max_draws={max_draws}. "
                f"Current counts: CASE={len(cases)}, CONTROL={len(ctrls)}."
            )

        p = sample_one_profile(rng)
        lab = ground_truth_label(p)
        if lab.endswith("CASE"):
            if len(cases) < target:
                cases.append(p)
        else:
            if len(ctrls) < target:
                ctrls.append(p)

    profiles = cases + ctrls
    rng.shuffle(profiles)
    print(f"[DATA] Balanced dataset built: CASE={len(cases)} CONTROL={len(ctrls)} (draws={draws})")
    return profiles


# -----------------------
# PROMPT BUILDING
# -----------------------
def _modality(fid: str) -> str:
    if fid in RELEVANT_FEATURES:
        return "clinical_relevant"
    return "distractor_word"


def build_prompt_and_spans(profile: Dict[str, int], active: Set[str]) -> Tuple[str, Dict[str, Tuple[int, int]]]:
    """
    Prompt is explicitly "linearly-scored" to help the LLM behave more like the
    ground-truth rule:
        score = sum(weight_i * value_i)
        CASE if score > 0 else CONTROL

    The two clinically relevant features have weight ~ 1.0.
    Distractor words have tiny weights (abs <= 0.2).
    """
    header = (
        f"You are a classifier. Output exactly one label:\n"
        f"  '{PHENOTYPE} CASE'  or  '{PHENOTYPE} CONTROL'\n\n"
        f"Task:\n"
        f"  Compute a risk score as: score = sum(weight * value) across the feature lines below.\n"
        f"  Then output '{PHENOTYPE} CASE' if score > 0 else '{PHENOTYPE} CONTROL'.\n\n"
        f"Important:\n"
        f"  Only TWO lines are clinically relevant (sleep quality, childhood trauma exposure) and they have weight ≈ 1.0.\n"
        f"  The other EIGHT lines are distractor words with tiny weights (|weight| ≤ 0.2), so they should barely matter.\n\n"
        f"Interpretation of values:\n"
        f"  value=+1.00 suggests risk, value=-1.00 suggests protection.\n"
        f"  value=+0.00 and vote=UNKNOWN means missing/ablated.\n\n"
        f"Features:\n"
    )

    lines: List[str] = []
    spans: Dict[str, Tuple[int, int]] = {}
    cursor = len(header)

    for fid in FEATURE_IDS:
        if fid in active:
            v = profile[fid]
            val_str = f"{float(v):+.2f}"
            vote = "CASE" if v > 0 else "CONTROL"
        else:
            val_str = f"{0.0:+.2f}"
            vote = "UNKNOWN"

        w = TRUE_W[fid]
        modality = _modality(fid)
        text = SURFACE_TEXT.get(fid, fid)

        line = (
            f"@@ FEAT_ID={fid} | modality={modality} | text='{text}' | "
            f"weight={w:+.2f} | value={val_str} | vote={vote} @@\n"
        )
        start = cursor
        end = cursor + len(line)
        spans[fid] = (start, end)
        lines.append(line)
        cursor = end

    footer = "\nAnswer with the label only.\nLabel:"
    prompt = header + "".join(lines) + footer
    return prompt, spans


def build_prompt_only(profile: Dict[str, int], active: Set[str]) -> str:
    prompt, _ = build_prompt_and_spans(profile, active)
    return prompt


# -----------------------
# MAIN
# -----------------------
def main():
    ensure_dir(TABLES_DIR)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("ig_attribution loaded from:", ig.__file__)
    print("group_shapley loaded from:", gs.__file__)
    print("group_shapley exports:", [a for a in dir(gs) if "shap" in a.lower()])

    if N_PROFILES % 2 != 0:
        raise ValueError("Set N_PROFILES to an even number to enforce exact 50/50 CASE/CONTROL.")

    model, tokenizer, device = ig.load_model(model_name=MODEL_NAME, prefer_mps=True)
    labels = ig.prepare_label_tokens(tokenizer, case_str=" CASE", control_str=" CONTROL")

    print(f"\nModel: {MODEL_NAME} | device: {device}")
    print(f"CASE token ids: {labels.case_ids} | CONTROL token ids: {labels.control_ids}")
    if not labels.single_token_labels:
        print("WARNING: labels are multi-token. Scoring still works, but IG in this demo expects 1-token labels.\n")

    profiles = make_balanced_profiles(n=N_PROFILES, seed=1)

    # Tables (rows)
    subjects_rows: List[Dict[str, Any]] = []
    features_rows: List[Dict[str, Any]] = []
    importance_rows: List[Dict[str, Any]] = []
    shapley_runs_rows: List[Dict[str, Any]] = []

    # Metrics
    mae_logit_sum = 0.0
    mae_label_sum = 0.0
    ig_fi_mae_sum = 0.0
    shap_fi_mae_sum = 0.0

    global_ig_abs = {fid: 0.0 for fid in FEATURE_IDS}
    global_shap_abs = {fid: 0.0 for fid in FEATURE_IDS}

    # ETA across subjects
    subj_times: List[float] = []
    t_start_all = time.perf_counter()

    for i, prof in enumerate(profiles, start=1):
        t_subj0 = time.perf_counter()

        subject_id = i
        active_all = set(FEATURE_IDS)
        prompt, spans = build_prompt_and_spans(prof, active=active_all)

        gt_log = ground_truth_logit(prof)
        gt_lab = ground_truth_label(prof)

        print("\n" + "=" * 100)
        print(f"SUBJECT {i:02d}/{N_PROFILES}")

        if PRINT_PROMPT_PER_SUBJECT:
            print("\n--- PROMPT ---")
            print(prompt)

        # 1) score
        t0 = time.perf_counter()
        s, lp_case, lp_control = ig.score_case_control(model, tokenizer, prompt, labels, device)
        t_score = time.perf_counter() - t0
        pred_lab = f"{PHENOTYPE} CASE" if s > 0 else f"{PHENOTYPE} CONTROL"

        # 2) generation
        t0 = time.perf_counter()
        gen = ig.generate_next_token(model, tokenizer, prompt, device, max_new_tokens=1)
        t_gen = time.perf_counter() - t0

        if PRINT_LLM_GENERATION_PER_SUBJECT:
            print("\n--- LLM OUTPUT (next token, greedy) ---")
            print(repr(gen))

        print("\n--- SCORES ---")
        print(f"GroundTruth: label={gt_lab}  logit={gt_log:+.3f}")
        print(f"ModelScore : pred={pred_lab}  s={s:+.4f}  (lp_case={lp_case:+.4f}, lp_ctrl={lp_control:+.4f})")
        print(
            f"Relevant vals: sleep_quality={prof['sleep_quality']:+d}  "
            f"childhood_trauma_exposure={prof['childhood_trauma_exposure']:+d}"
        )
        print(f"Timing: score={t_score:.2f}s | gen={t_gen:.2f}s")

        # store per-feature ground truth & values
        for fid in FEATURE_IDS:
            features_rows.append(
                dict(
                    run_id=run_id,
                    subject_id=subject_id,
                    feature_id=fid,
                    modality=_modality(fid),
                    value=int(prof[fid]),
                    true_weight=float(TRUE_W[fid]),
                    true_contribution=float(TRUE_W[fid]) * float(prof[fid]),
                )
            )

        # 3) IG
        print("\n--- IG RUN ---")
        last_print = [0.0]
        ig_t0 = time.perf_counter()

        def ig_progress(step_idx: int, steps: int, elapsed: float, eta: float):
            now = time.perf_counter()
            if (now - last_print[0]) >= PROGRESS_EVERY_SEC or step_idx == steps:
                last_print[0] = now
                print(f"  IG step {step_idx:02d}/{steps} | elapsed {fmt_time(elapsed)} | eta {fmt_time(eta)}")

        ig_scores = safe_call(
            ig.integrated_gradients_feature_importance,
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            feature_spans=spans,
            labels=labels,
            device=device,
            steps=IG_STEPS,
            baseline_mode="eos",
            progress_cb=ig_progress,
        )
        t_ig = time.perf_counter() - ig_t0
        print(f"  IG done in {fmt_time(t_ig)}")

        ig_norm_abs = normalize_abs_vector(ig_scores)

        for fid in FEATURE_IDS:
            importance_rows.append(
                dict(
                    run_id=run_id,
                    subject_id=subject_id,
                    method="IG",
                    feature_id=fid,
                    importance=float(ig_scores[fid]),
                    abs_importance=abs(float(ig_scores[fid])),
                    norm_abs=float(ig_norm_abs[fid]),
                    extra_std="",  # NA
                )
            )

        # 4) Shapley repeats (store each run + mean/std)
        print("\n--- SHAPLEY RUN ---")
        last_print = [0.0]
        shap_t0 = time.perf_counter()

        def score_fn(active: Set[str]) -> float:
            ptxt = build_prompt_only(prof, active)
            s_local, _, _ = ig.score_case_control(model, tokenizer, ptxt, labels, device)
            return s_local

        def shap_progress(run_idx: int, runs: int, perm_done: int, K: int, elapsed_total: float, eta_total: float):
            now = time.perf_counter()
            if (now - last_print[0]) >= PROGRESS_EVERY_SEC or (run_idx == runs and perm_done == K):
                last_print[0] = now
                total_perm = runs * K
                done_perm = (run_idx - 1) * K + perm_done
                print(
                    f"  Shapley progress: run {run_idx}/{runs}, perm {perm_done}/{K} "
                    f"({done_perm}/{total_perm}) | elapsed {fmt_time(elapsed_total)} | eta {fmt_time(eta_total)}"
                )

        shap_mean, shap_std, shap_per_run = shapley_with_repeats_and_store_runs(
            score_fn=score_fn,
            feature_ids=FEATURE_IDS,
            K=SHAPLEY_K,
            runs=SHAPLEY_RUNS,
            seed=1000 + i,
            verbose=PRINT_SHAPLEY_VERBOSE_INTERNAL,
            progress_cb=shap_progress,
        )
        t_shap = time.perf_counter() - shap_t0
        print(f"  Shapley done in {fmt_time(t_shap)}")

        # Store per-run Shapley (for later “within-subject” distributions)
        for r_idx, phi_r in enumerate(shap_per_run, start=1):
            for fid in FEATURE_IDS:
                shapley_runs_rows.append(
                    dict(
                        run_id=run_id,
                        subject_id=subject_id,
                        method="Shapley_MC",
                        run_idx=r_idx,
                        feature_id=fid,
                        importance=float(phi_r[fid]),
                    )
                )

        shap_norm_abs = normalize_abs_vector(shap_mean)
        for fid in FEATURE_IDS:
            importance_rows.append(
                dict(
                    run_id=run_id,
                    subject_id=subject_id,
                    method="Shapley_MC",
                    feature_id=fid,
                    importance=float(shap_mean[fid]),
                    abs_importance=abs(float(shap_mean[fid])),
                    norm_abs=float(shap_norm_abs[fid]),
                    extra_std=float(shap_std[fid]),
                )
            )

        # Print top features
        top_ig = sorted(ig_scores.items(), key=lambda kv: abs(kv[1]), reverse=True)[:5]
        top_shap = sorted(shap_mean.items(), key=lambda kv: abs(kv[1]), reverse=True)[:5]

        print("\n--- TOP FEATURES (IG) ---")
        for fid, sc in top_ig:
            print(f"  {fid:26s}  {sc:+.6f}")

        print("\n--- TOP FEATURES (Shapley MC mean ± std) ---")
        for fid, sc in top_shap:
            print(f"  {fid:26s}  {sc:+.6f}  ±{shap_std[fid]:.6f}")

        # Update globals
        for fid in FEATURE_IDS:
            global_ig_abs[fid] += abs(float(ig_scores[fid]))
            global_shap_abs[fid] += abs(float(shap_mean[fid]))

        # Performance metrics
        y_true = 1.0 if gt_lab.endswith("CASE") else 0.0
        y_pred = 1.0 if pred_lab.endswith("CASE") else 0.0
        mae_label_sum += abs(y_true - y_pred)
        mae_logit_sum += abs(s - gt_log)

        true_contrib = {fid: TRUE_W[fid] * float(prof[fid]) for fid in FEATURE_IDS}
        ig_fi_mae_sum += mae_feature_importance(ig_scores, true_contrib)
        shap_fi_mae_sum += mae_feature_importance(shap_mean, true_contrib)

        # subject timing
        t_subj = time.perf_counter() - t_subj0
        subj_times.append(t_subj)
        avg = sum(subj_times) / len(subj_times)
        remain = avg * (N_PROFILES - i)
        elapsed_all = time.perf_counter() - t_start_all

        print("\n--- SUBJECT TIMING ---")
        print(f"Subject total: {fmt_time(t_subj)} | total elapsed: {fmt_time(elapsed_all)} | ETA remaining: {fmt_time(remain)}")

        # store subject row
        subjects_rows.append(
            dict(
                run_id=run_id,
                subject_id=subject_id,
                gt_label=gt_lab,
                gt_logit=float(gt_log),
                pred_label=pred_lab,
                model_s=float(s),
                lp_case=float(lp_case),
                lp_control=float(lp_control),
                sleep_quality=int(prof["sleep_quality"]),
                childhood_trauma_exposure=int(prof["childhood_trauma_exposure"]),
                timing_score_s=float(t_score),
                timing_gen_s=float(t_gen),
                timing_ig_s=float(t_ig),
                timing_shapley_s=float(t_shap),
                ig_steps=int(IG_STEPS),
                shapley_k=int(SHAPLEY_K),
                shapley_runs=int(SHAPLEY_RUNS),
                model_name=str(MODEL_NAME),
                device=str(device),
            )
        )

    n = float(N_PROFILES)

    print("\n" + "#" * 100)
    print("FINAL SUMMARY\n")

    print("1) Prediction performance vs ground truth")
    print(f"  MAE(label)  = {mae_label_sum / n:.3f}   (error-rate)")
    print(f"  MAE(logit)  = {mae_logit_sum / n:.3f}   (|model_s - true_logit|)")

    print("\n2) Feature-importance quality vs ground-truth contributions (normalized abs, MAE)")
    print(f"  IG MAE(feature-importance)         = {ig_fi_mae_sum / n:.3f}")
    print(f"  Shapley MC MAE(feature-importance) = {shap_fi_mae_sum / n:.3f}")

    print("\n3) GLOBAL mean(|importance|) across subjects")
    print("  IG:")
    for fid, v in sorted({k: v / n for k, v in global_ig_abs.items()}.items(), key=lambda kv: kv[1], reverse=True):
        print(f"    {fid:26s}  {v:.6f}")

    print("\n  Shapley MC:")
    for fid, v in sorted({k: v / n for k, v in global_shap_abs.items()}.items(), key=lambda kv: kv[1], reverse=True):
        print(f"    {fid:26s}  {v:.6f}")

    # -----------------------
    # WRITE TABLES
    # -----------------------
    subjects_path = TABLES_DIR / f"subjects_{run_id}.csv"
    features_path = TABLES_DIR / f"features_{run_id}.csv"
    importance_path = TABLES_DIR / f"importance_{run_id}.csv"
    shap_runs_path = TABLES_DIR / f"shapley_runs_{run_id}.csv"
    meta_path = TABLES_DIR / f"run_meta_{run_id}.json"

    write_csv(
        subjects_path,
        subjects_rows,
        fieldnames=list(subjects_rows[0].keys()) if subjects_rows else ["run_id", "subject_id"],
    )
    write_csv(
        features_path,
        features_rows,
        fieldnames=list(features_rows[0].keys()) if features_rows else ["run_id", "subject_id", "feature_id"],
    )
    write_csv(
        importance_path,
        importance_rows,
        fieldnames=list(importance_rows[0].keys()) if importance_rows else ["run_id", "subject_id", "method", "feature_id"],
    )
    write_csv(
        shap_runs_path,
        shapley_runs_rows,
        fieldnames=list(shapley_runs_rows[0].keys())
        if shapley_runs_rows
        else ["run_id", "subject_id", "method", "run_idx", "feature_id", "importance"],
    )

    meta = dict(
        run_id=run_id,
        created_at=datetime.now().isoformat(timespec="seconds"),
        phenotype=PHENOTYPE,
        model_name=MODEL_NAME,
        n_profiles=N_PROFILES,
        relevant_features=RELEVANT_FEATURES,
        distractor_features=DISTRACTOR_FEATURES,
        feature_ids=FEATURE_IDS,
        surface_text=SURFACE_TEXT,
        true_w=TRUE_W,
        ig_steps=IG_STEPS,
        shapley_k=SHAPLEY_K,
        shapley_runs=SHAPLEY_RUNS,
        paths=dict(
            tables_dir=str(TABLES_DIR),
            subjects_csv=str(subjects_path),
            features_csv=str(features_path),
            importance_csv=str(importance_path),
            shapley_runs_csv=str(shap_runs_path),
        ),
    )
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("\n--- TABLES WRITTEN ---")
    print("  ", subjects_path)
    print("  ", features_path)
    print("  ", importance_path)
    print("  ", shap_runs_path)
    print("  ", meta_path)

    print("\nDone.")


if __name__ == "__main__":
    torch.manual_seed(0)
    main()

#NOTE: 'rm -rf ~/.cache/huggingface' if memory-related error
