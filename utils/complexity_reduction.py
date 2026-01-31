#!/usr/bin/env python3
"""
complexity_reduction.py

Purpose
-------
Visualize the *percentage reduction* in value-function (LLM) evaluations achieved by aHFR-TokenSHAP
relative to Flat TokenSHAP, using the evaluation-count logic in aHFR_TokenSHAP.py.

Outputs
-------
- plot_vs_K.png
- plot_vs_T.png
- plot_vs_L.png
- plot_vs_P.png
- complexity_reduction_percent_2x2.png

Notes
-----
This script isolates *evaluation-count* savings (calls to score_fn / v(S)). It does not model:
(i) per-call token-processing cost differences, (ii) caching/batching, or (iii) hierarchy CPU overhead.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, ScalarFormatter


# -----------------------
# Stripe styling (GLOBAL, shared across all panels)
# -----------------------
STRIPE_REL_WIDTH = 0.03  # ±3% around x0 (multiplicative width; works on log axes too)
STRIPE_LINEWIDTH = 1.1
STRIPE_HATCH = "///"
STRIPE_COLOR = "black"


# -----------------------
# Core evaluation-count logic (matches provided aHFR_TokenSHAP.py)
# -----------------------
def calibration_budget(K: int, frac: float = 0.2) -> int:
    """Match HFR_TokenSHAP._calibration_budget."""
    if K <= 1:
        return 1
    return max(1, int(round(frac * K)))


def eval_counts_flat(K: int, T: int) -> int:
    """Flat TokenSHAP eval count = K*(T+1)."""
    if K < 1 or T < 0:
        raise ValueError("Require K>=1 and T>=0.")
    return int(K) * (int(T) + 1)


def eval_counts_ahfr(K: int, L: int, P: int, K0: Optional[int] = None, k0_frac: float = 0.2) -> int:
    """aHFR eval count = K*(L+1) + K0*(P+1)."""
    if K < 1 or L < 1 or P < 1:
        raise ValueError("Require K>=1, L>=1, P>=1.")
    if K0 is None:
        K0 = calibration_budget(int(K), frac=k0_frac)
    return int(K) * (int(L) + 1) + int(K0) * (int(P) + 1)


def percent_reduction(K: int, T: int, L: int, P: int, K0: Optional[int] = None, k0_frac: float = 0.2) -> float:
    """
    Percentage reduction in eval count:
      100 * (1 - evals_ahfr / evals_flat)

    Note: can be negative if evals_ahfr > evals_flat.
    """
    flat = float(eval_counts_flat(K, T))
    ahfr = float(eval_counts_ahfr(K, L, P, K0=K0, k0_frac=k0_frac))
    if flat <= 0.0:
        raise ValueError("Invalid flat eval count (must be > 0).")

    red = 100.0 * (1.0 - (ahfr / flat))
    if red >= 100.0:
        red = 100.0 - 1e-9
    return red


# -----------------------
# Plotting helpers
# -----------------------
def _pct_formatter(x: float, pos: int) -> str:
    return f"{x:.0f}%"


def _k_formatter(x: float, pos: int) -> str:
    """Pretty-print counts as 1k, 2k, 16k, etc."""
    x = float(x)
    if x >= 1000:
        if abs(x % 1000) < 1e-9:
            return f"{int(x/1000)}k"
        return f"{x/1000:.1f}k"
    return f"{int(x)}"


def _make_subtitle(base: Dict[str, int], exclude: str, k0_fixed: Optional[int]) -> str:
    parts = []
    if exclude != "K":
        parts.append(f"K={base['K']}")
    if exclude != "T":
        parts.append(f"T={base['T']}")
    if exclude != "L":
        parts.append(f"L={base['L']}")
    if exclude != "P":
        parts.append(f"P={base['P']}")
    baseline = ", ".join(parts)

    if k0_fixed is None:
        k0_part = r"$K_0=\max(1,\mathrm{round}(0.2K))$"
    else:
        k0_part = rf"$K_0={k0_fixed}$"
    return f"Baseline: {baseline} | {k0_part}"


def _stripe_at_value(ax, x_value: int, y_max: float, label: str, xytext_mult: float = 1.35) -> None:
    """Draw a hatched vertical band centered on x_value (consistent look across all panels)."""
    x0 = float(x_value)
    if x0 <= 0:
        return

    left = x0 * (1.0 - STRIPE_REL_WIDTH)
    right = x0 * (1.0 + STRIPE_REL_WIDTH)

    ax.axvspan(
        left,
        right,
        facecolor="none",
        edgecolor=STRIPE_COLOR,
        hatch=STRIPE_HATCH,
        linewidth=STRIPE_LINEWIDTH,
        zorder=6,
    )

    ax.annotate(
        label,
        xy=(x0, y_max - 3.0),
        xytext=(x0 * xytext_mult, y_max - 18.0),
        textcoords="data",
        arrowprops=dict(
            arrowstyle="->",
            linewidth=STRIPE_LINEWIDTH,
            color=STRIPE_COLOR,
        ),
        fontsize=9,
        ha="left",
        va="top",
        zorder=7,
        bbox=dict(
            boxstyle="round,pad=0.25",
            facecolor="white",
            alpha=0.90,
            linewidth=0.6,
        ),
    )


def _style_ax(
    ax,
    *,
    xlabel: str,
    title: str,
    subtitle: str,
    y_min: float,
    y_max: float,
    xlog: bool = False,
    xformatter=None,
) -> None:
    """Standardized panel styling (ensures consistent layout across all plots)."""
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Evaluation-count reduction (%)")
    ax.set_title(title)

    ax.grid(True, which="major", linestyle="--", linewidth=0.7, alpha=0.55)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.35)

    ax.yaxis.set_major_formatter(FuncFormatter(_pct_formatter))
    ax.set_ylim(y_min, y_max)

    if xlog:
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(ScalarFormatter())

    if xformatter is not None:
        ax.xaxis.set_major_formatter(FuncFormatter(xformatter))

    ax.text(
        0.02,
        0.03,
        subtitle,
        transform=ax.transAxes,
        fontsize=9,
        va="bottom",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.30", facecolor="white", alpha=0.88, linewidth=0.6),
    )


def _save_fig(fig, path: Path) -> None:
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    # Consistent typography
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
        }
    )

    # Output directory (local if exists; otherwise sandbox-friendly)
    preferred = Path(
        "/Users/stijnvanseveren/PythonProjects/HFR_TokenSHAP/feature_importance_estimation/demonstration_results/visuals"
    )
    out_dir = preferred if preferred.exists() else Path("/mnt/data/demonstration_results/visuals")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Baseline plausible setting
    base: Dict[str, int] = dict(K=100, T=500, L=10, P=5)
    base_k0 = calibration_budget(base["K"])

    # ------------------------------------------------------------------
    # Parameter ranges (expanded)
    # ------------------------------------------------------------------
    Ks = np.array([2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000], dtype=int)

    # T: start at 1 and grow by powers of 2 (1 -> 524,288)
    Ts = np.array([2**i for i in range(0, 20)], dtype=int)

    # L: expanded; includes base["T"] so L=T marker is on-grid
    Ls = np.array(
        sorted(
            set(
                [2**i for i in range(0, 18)]  # 1 .. 131072
                + [5, 10, 20, 40, 80, 160, 320, 640, 1000, base["T"]]
            )
        ),
        dtype=int,
    )

    Ps = np.array([1, 2, 3, 5, 8, 12, 16, 24, 32, 64, 128, 256, 512], dtype=int)

    # ------------------------------------------------------------------
    # Compute percent reductions
    # ------------------------------------------------------------------
    # K plot: K0 varies with K (default budget)
    yK = np.array([percent_reduction(int(k), base["T"], base["L"], base["P"]) for k in Ks], dtype=float)

    # Other plots: K fixed -> fix K0 to base_k0
    yT = np.array([percent_reduction(base["K"], int(t), base["L"], base["P"], K0=base_k0) for t in Ts], dtype=float)
    yL = np.array([percent_reduction(base["K"], base["T"], int(l), base["P"], K0=base_k0) for l in Ls], dtype=float)
    yP = np.array([percent_reduction(base["K"], base["T"], base["L"], int(p), K0=base_k0) for p in Ps], dtype=float)

    # Shared y-limits across all panels
    y_all = np.concatenate([yK, yT, yL, yP])
    y_max = 100.0
    y_min = float(np.floor(min(0.0, y_all.min() - 5.0)))
    y_min = max(y_min, -100.0)  # readability clamp

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    # ------------------------------------------------------------------
    # Single-panel figures (4)
    # ------------------------------------------------------------------
    # (a) K
    fig, ax = plt.subplots(figsize=(6.6, 4.6), dpi=200)
    ax.plot(Ks, yK, marker="o", linestyle="-", linewidth=2.25, markersize=5.1, color=colors[0])
    _style_ax(
        ax,
        xlabel=r"K  (main MC permutations; log$_2$ scale)",
        title="(a) Sensitivity to K (MC budget)",
        subtitle=_make_subtitle(base, exclude="K", k0_fixed=None) + "",
        y_min=y_min,
        y_max=y_max,
        xlog=True,
    )
    _save_fig(fig, out_dir / "plot_vs_K.png")

    # (b) T
    fig, ax = plt.subplots(figsize=(6.6, 4.6), dpi=200)
    ax.plot(Ts, yT, marker="o", linestyle="-", linewidth=2.25, markersize=5.1, color=colors[1])
    _style_ax(
        ax,
        xlabel=r"T  (prompt tokens; log$_2$ scale)",
        title="(b) Sensitivity to T (prompt length)",
        subtitle=_make_subtitle(base, exclude="T", k0_fixed=base_k0) + "",
        y_min=y_min,
        y_max=y_max,
        xlog=True,
        xformatter=_k_formatter,
    )
    _stripe_at_value(ax, x_value=base["L"], y_max=y_max, label="T = L (token-level boundary)", xytext_mult=1.6)
    _save_fig(fig, out_dir / "plot_vs_T.png")

    # (c) L
    fig, ax = plt.subplots(figsize=(6.6, 4.6), dpi=200)
    ax.plot(Ls, yL, marker="o", linestyle="-", linewidth=2.25, markersize=5.1, color=colors[2])
    _style_ax(
        ax,
        xlabel=r"L  (leaf features; log$_2$ scale)",
        title="(c) Sensitivity to L (feature resolution)",
        subtitle=_make_subtitle(base, exclude="L", k0_fixed=base_k0) + "",
        y_min=y_min,
        y_max=y_max,
        xlog=True,
    )
    _stripe_at_value(ax, x_value=base["T"], y_max=y_max, label="L = T (token-level limit)", xytext_mult=1.35)
    _save_fig(fig, out_dir / "plot_vs_L.png")

    # (d) P
    fig, ax = plt.subplots(figsize=(6.6, 4.6), dpi=200)
    ax.plot(Ps, yP, marker="o", linestyle="-", linewidth=2.25, markersize=5.1, color=colors[3])
    _style_ax(
        ax,
        xlabel="P  (primary groups; linear scale)",
        title="(d) Sensitivity to P (root breadth)",
        subtitle=_make_subtitle(base, exclude="P", k0_fixed=base_k0),
        y_min=y_min,
        y_max=y_max,
        xlog=False,
    )
    _save_fig(fig, out_dir / "plot_vs_P.png")

    # ------------------------------------------------------------------
    # Combined 2x2 figure (1)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.4), dpi=190, constrained_layout=True)

    # (a) K
    ax = axes[0, 0]
    ax.plot(Ks, yK, marker="o", linestyle="-", linewidth=2.25, markersize=5.1, color=colors[0])
    _style_ax(
        ax,
        xlabel=r"K  (main MC permutations; log$_2$ scale)",
        title="(a) Sensitivity to K (MC budget)",
        subtitle=_make_subtitle(base, exclude="K", k0_fixed=None) + "",
        y_min=y_min,
        y_max=y_max,
        xlog=True,
    )

    # (b) T
    ax = axes[0, 1]
    ax.plot(Ts, yT, marker="o", linestyle="-", linewidth=2.25, markersize=5.1, color=colors[1])
    _style_ax(
        ax,
        xlabel=r"T  (prompt tokens; log$_2$ scale)",
        title="(b) Sensitivity to T (prompt length)",
        subtitle=_make_subtitle(base, exclude="T", k0_fixed=base_k0) + "",
        y_min=y_min,
        y_max=y_max,
        xlog=True,
        xformatter=_k_formatter,
    )
    _stripe_at_value(ax, x_value=base["L"], y_max=y_max, label="T = L (token-level boundary)", xytext_mult=1.6)

    # (c) L
    ax = axes[1, 0]
    ax.plot(Ls, yL, marker="o", linestyle="-", linewidth=2.25, markersize=5.1, color=colors[2])
    _style_ax(
        ax,
        xlabel=r"L  (leaf features; log$_2$ scale)",
        title="(c) Sensitivity to L (feature resolution)",
        subtitle=_make_subtitle(base, exclude="L", k0_fixed=base_k0) + "",
        y_min=y_min,
        y_max=y_max,
        xlog=True,
    )
    _stripe_at_value(ax, x_value=base["T"], y_max=y_max, label="L = T (token-level limit)", xytext_mult=1.35)

    # (d) P
    ax = axes[1, 1]
    ax.plot(Ps, yP, marker="o", linestyle="-", linewidth=2.25, markersize=5.1, color=colors[3])
    _style_ax(
        ax,
        xlabel="P  (primary groups; linear scale)",
        title="(d) Sensitivity to P (root breadth)",
        subtitle=_make_subtitle(base, exclude="P", k0_fixed=base_k0),
        y_min=y_min,
        y_max=y_max,
        xlog=False,
    )

    #fig.suptitle(
    #    "aHFR-TokenSHAP vs Flat TokenSHAP: Percentage Reduction in Value-Function Evaluations",
    #    fontsize=14.6,
    #)

    out_png = out_dir / "complexity_reduction_percent_2x2.png"
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    print("Wrote outputs to:", out_dir)
    print(" -", out_dir / "plot_vs_K.png")
    print(" -", out_dir / "plot_vs_T.png")
    print(" -", out_dir / "plot_vs_L.png")
    print(" -", out_dir / "plot_vs_P.png")
    print(" -", out_dir / "complexity_reduction_percent_2x2.png")


if __name__ == "__main__":
    main()
