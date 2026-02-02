# visualize_prompt_with_feature_squares.py
from __future__ import annotations

import argparse
import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Set

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize, ListedColormap
from matplotlib.cm import ScalarMappable
from matplotlib.font_manager import FontProperties


# =============================================================================
# Prompt parsing
# =============================================================================
# Feature line: @@ FEAT_ID=... | text=... | value=... | vote=... @@
_FEAT_LINE_RE = re.compile(r"^@@\s+FEAT_ID=([^|\s]+)\s+\|.*?@@\s*$")
_TEXT_FIELD_RE = re.compile(r"(text=)([^|]*?)(\s*\|)")


def is_feature_line(line: str) -> bool:
    return _FEAT_LINE_RE.match(line.strip()) is not None


def extract_feat_id(line: str) -> Optional[str]:
    m = _FEAT_LINE_RE.match(line.strip())
    if not m:
        return None
    return m.group(1).strip()


def extract_text_field(line: str) -> Optional[str]:
    m = _TEXT_FIELD_RE.search(line)
    if not m:
        return None
    return m.group(2).strip()


def load_prompt_text(prompt_path: Optional[str], prompt_text: Optional[str]) -> str:
    if prompt_text and prompt_text.strip():
        return prompt_text
    if prompt_path:
        p = Path(prompt_path)
        if not p.exists():
            raise FileNotFoundError(f"Prompt file not found: {p}")
        return p.read_text(encoding="utf-8")
    raise ValueError("Provide either --prompt_path or --prompt_text.")


# =============================================================================
# Hierarchical feature tree (flatten to id -> {label, score})
# =============================================================================
@dataclass(frozen=True)
class NodeInfo:
    label: str
    score: float


def _as_float(x: Any, *, where: str) -> float:
    try:
        return float(x)
    except Exception as e:
        raise ValueError(f"Expected numeric score at {where}, got: {x!r}") from e


def _get_label(d: Dict[str, Any], default: str) -> str:
    for k in ("label", "name", "title"):
        if k in d and isinstance(d[k], str) and d[k].strip():
            return d[k].strip()
    return default


def _get_score(d: Dict[str, Any], *, where: str) -> Optional[float]:
    # Accept a few common keys; primary is "score"
    for k in ("score", "aHFR-TokenSHAP", "aHFR_TokenSHAP", "mean", "value"):
        if k in d and d[k] is not None:
            return _as_float(d[k], where=f"{where}.{k}")
    return None


TreeLike = Union[Dict[str, Any], List[Any]]


def flatten_feature_tree(tree: TreeLike) -> Dict[str, NodeInfo]:
    """
    Flattens a hierarchical feature structure into:
        out[node_id] = NodeInfo(label=<str>, score=<float>)

    Supported inputs (JSON / python literal):
      1) Node dict:
         {"id":"root","label":"Root","score":0.1,"children":[{...}, ...]}

      2) Mapping dict:
         {"childhood_trauma_exposure":{"label":"Childhood trauma","score":0.49,"children":[...]}, ...}

      3) List of nodes:
         [{"id":"x","label":"X","score":...}, {"id":"y",...}]
    """
    out: Dict[str, NodeInfo] = {}

    def add_node(node_id: str, node_dict: Dict[str, Any], *, where: str) -> None:
        node_id = str(node_id).strip()
        if not node_id:
            raise ValueError(f"Empty node id at {where}")

        label = _get_label(node_dict, default=node_id)
        score_opt = _get_score(node_dict, where=where)
        if score_opt is None:
            raise ValueError(f"Missing score for node '{node_id}' at {where}")
        score = float(score_opt)

        if node_id in out:
            # allow exact duplicates (same label/score), otherwise fail loudly
            prev = out[node_id]
            if prev.label != label or abs(prev.score - score) > 1e-12:
                raise ValueError(
                    f"Duplicate node id '{node_id}' with conflicting data.\n"
                    f"  prev: label={prev.label!r}, score={prev.score}\n"
                    f"  new : label={label!r}, score={score}\n"
                    f"  at  : {where}"
                )
        else:
            out[node_id] = NodeInfo(label=label, score=score)

        # Recurse
        kids = node_dict.get("children", None)
        if kids is None:
            return
        if not isinstance(kids, list):
            raise ValueError(f"'children' must be a list at {where}.children")
        for i, child in enumerate(kids):
            _walk(child, where=f"{where}.children[{i}]")

    def _walk(obj: Any, *, where: str) -> None:
        if isinstance(obj, list):
            for i, it in enumerate(obj):
                _walk(it, where=f"{where}[{i}]")
            return

        if not isinstance(obj, dict):
            raise ValueError(f"Tree element must be dict or list at {where}, got {type(obj).__name__}")

        # Case 1: explicit node dict with "id"
        if "id" in obj:
            add_node(obj["id"], obj, where=where)
            return

        # Case 2: mapping dict of id -> node-dict
        # (interpret keys as node ids, values as node dicts)
        for k, v in obj.items():
            if not isinstance(v, dict):
                raise ValueError(
                    f"Mapping tree requires dict values at {where}.{k}, got {type(v).__name__}"
                )
            # allow omitting "id" inside value
            node_dict = dict(v)
            node_dict.setdefault("id", k)
            add_node(k, node_dict, where=f"{where}.{k}")

    _walk(tree, where="tree")
    return out


def load_feature_tree(path: str) -> Dict[str, NodeInfo]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Feature tree file not found: {p}")

    raw = p.read_text(encoding="utf-8").strip()
    if not raw:
        raise ValueError(f"Feature tree file is empty: {p}")

    # JSON by default; python literal fallback
    try:
        tree_obj = json.loads(raw)
    except Exception:
        tree_obj = ast.literal_eval(raw)

    return flatten_feature_tree(tree_obj)


# =============================================================================
# Color utilities
# =============================================================================
def _lighten_rgba(rgba, amount: float) -> tuple:
    """
    Blend color with white to reduce saturation / make it lighter.
    amount in [0,1]; 0 = original, 1 = white.
    """
    amount = float(np.clip(amount, 0.0, 1.0))
    r, g, b = rgba[:3]
    a = rgba[3] if len(rgba) > 3 else 1.0
    r2 = (1.0 - amount) * r + amount * 1.0
    g2 = (1.0 - amount) * g + amount * 1.0
    b2 = (1.0 - amount) * b + amount * 1.0
    return (r2, g2, b2, a)


def make_light_cmap(base_cmap_name: str, lighten: float = 0.25, n: int = 256) -> ListedColormap:
    base = plt.get_cmap(base_cmap_name)
    cols = [_lighten_rgba(base(i / (n - 1)), lighten) for i in range(n)]
    return ListedColormap(cols, name=f"{base_cmap_name}_light_{lighten:.2f}")


# =============================================================================
# Layout structs
# =============================================================================
@dataclass
class RowPlain:
    x_axes: float
    text: str


@dataclass
class RowFeatureMain:
    fid: str
    ftext: str
    pre: str
    post_chunk: str
    # computed geometry (axes coords)
    sq_x: float
    sq_w: float
    sq_h: float
    post_x: float


@dataclass
class RowFeatureCont:
    x_axes: float
    text: str


Row = RowPlain | RowFeatureMain | RowFeatureCont


# =============================================================================
# Text measurement + wrapping
# =============================================================================
def _text_whd_px(renderer, s: str, fp: FontProperties) -> Tuple[float, float, float]:
    """width, height, descent in pixels"""
    w, h, d = renderer.get_text_width_height_descent(s, fp, ismath=False)
    return float(w), float(h), float(d)


def _wrap_by_px(renderer, s: str, fp: FontProperties, max_w_px: float) -> List[str]:
    """
    Wrap `s` into multiple lines so each line width <= max_w_px.
    Very simple greedy word-wrap.
    """
    s = s.rstrip()
    if not s:
        return [""]

    # Fast path
    w, _, _ = _text_whd_px(renderer, s, fp)
    if w <= max_w_px:
        return [s]

    words = s.split(" ")
    out: List[str] = []
    cur = ""

    for w0 in words:
        cand = w0 if cur == "" else (cur + " " + w0)
        w_cand, _, _ = _text_whd_px(renderer, cand, fp)
        if w_cand <= max_w_px:
            cur = cand
            continue

        # If single word itself is too long, hard-split by chars
        if cur == "":
            chunk = ""
            for ch in w0:
                cand2 = chunk + ch
                w2, _, _ = _text_whd_px(renderer, cand2, fp)
                if w2 <= max_w_px:
                    chunk = cand2
                else:
                    if chunk:
                        out.append(chunk)
                    chunk = ch
            if chunk:
                out.append(chunk)
            cur = ""
        else:
            out.append(cur)
            cur = w0

    if cur:
        out.append(cur)

    return out if out else [""]


# =============================================================================
# Rendering (same plotting logic; configurable inputs)
# =============================================================================
def render_prompt_png(
    *,
    prompt_text: str,
    node_map: Dict[str, NodeInfo],
    out_png: Path,
    label_source: str = "prompt",  # "prompt" or "tree"
    emphasize_ids: Optional[Set[str]] = None,
    cmap_name: str = "coolwarm",
    font_size: int = 11,
    lighten: float = 0.25,  # lower -> more saturated
    show_value_numbers: bool = False,
    border_pad_frac: float = 0.10,
    max_fit_attempts: int = 6,
    fig_size: Tuple[float, float] = (12, 16),
) -> None:
    if label_source not in ("prompt", "tree"):
        raise ValueError("--label_source must be 'prompt' or 'tree'")

    emphasize_ids = emphasize_ids or set()

    # Determine normalization range from actually-used features
    used_vals: List[float] = []
    for ln in prompt_text.splitlines():
        fid = extract_feat_id(ln)
        if fid and fid in node_map:
            used_vals.append(float(node_map[fid].score))
    vmax = float(max(used_vals)) if used_vals else 1.0
    norm = Normalize(vmin=0.0, vmax=vmax)
    cmap = make_light_cmap(cmap_name, lighten=lighten)

    # Fixed axes geometry (colorbar outside prompt border)
    AX_TEXT = [0.04, 0.04, 0.82, 0.92]  # left, bottom, width, height
    AX_CBAR = [0.90, 0.10, 0.03, 0.80]

    # Margins in ax_text (axes coords)
    left_margin = 0.02
    max_x = 0.985  # hard right limit for wrapping INSIDE prompt area

    # Square paddings (axes coords)
    pad_x_outer = 0.004
    pad_x_inner = 0.008

    mono = "DejaVu Sans Mono"

    # Try to ensure prompt content stays left of colorbar by reducing font size if needed.
    fs = int(font_size)
    last_overlap = None

    for attempt in range(max_fit_attempts):
        fig = plt.figure(figsize=fig_size)
        ax_text = fig.add_axes(AX_TEXT)
        cax = fig.add_axes(AX_CBAR)
        ax_text.set_axis_off()

        # Need renderer early for layout computations
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()

        fp = FontProperties(family=mono, size=fs)

        axw_px = float(ax_text.bbox.width)
        axh_px = float(ax_text.bbox.height)

        # Convert padding in axes->px
        pad_inner_px = pad_x_inner * axw_px

        # ---- Build layout rows (with wrapping)
        rows: List[Row] = []
        prompt_lines = prompt_text.splitlines()

        for ln in prompt_lines:
            if not is_feature_line(ln):
                max_w_px = (max_x - left_margin) * axw_px
                wrapped = _wrap_by_px(renderer, ln, fp, max_w_px=max_w_px)
                for wln in wrapped:
                    rows.append(RowPlain(x_axes=left_margin, text=wln))
                continue

            fid = extract_feat_id(ln) or ""

            # Choose label (prompt text field by default; optionally override with tree label)
            prompt_ftext = extract_text_field(ln) or fid.replace("_", " ")
            tree_label = node_map.get(fid, NodeInfo(label=prompt_ftext, score=0.0)).label
            ftext = tree_label if label_source == "tree" else prompt_ftext

            m_text = _TEXT_FIELD_RE.search(ln)
            if not m_text:
                max_w_px = (max_x - left_margin) * axw_px
                wrapped = _wrap_by_px(renderer, ln, fp, max_w_px=max_w_px)
                for wln in wrapped:
                    rows.append(RowPlain(x_axes=left_margin, text=wln))
                continue

            pre = ln[: m_text.start(2)]        # includes "text="
            post_full = ln[m_text.end(2) :]    # includes " | ..."

            pre_w_px, _, _ = _text_whd_px(renderer, pre, fp)
            ft_w_px, ft_h_px, _ = _text_whd_px(renderer, ftext, fp)

            # Square size in axes units (adaptive but stable)
            sq_w_px = ft_w_px + 2.0 * pad_inner_px
            sq_h_px = max(1.35 * ft_h_px, 0.82 * (axh_px / max(1, len(prompt_lines))))

            sq_w_axes = sq_w_px / axw_px
            sq_h_axes = sq_h_px / axh_px

            # Square x position (axes)
            sq_x_axes = left_margin + (pre_w_px / axw_px) + pad_x_outer

            # Where post starts (axes)
            post_x_axes = sq_x_axes + sq_w_axes + pad_x_outer

            # Wrap post to stay inside max_x
            max_post_w_px = max(10.0, (max_x - post_x_axes) * axw_px)
            post_wrapped = _wrap_by_px(renderer, post_full, fp, max_w_px=max_post_w_px)

            # Main row uses first chunk
            rows.append(
                RowFeatureMain(
                    fid=fid,
                    ftext=ftext,
                    pre=pre,
                    post_chunk=post_wrapped[0] if post_wrapped else "",
                    sq_x=sq_x_axes,
                    sq_w=sq_w_axes,
                    sq_h=sq_h_axes,
                    post_x=post_x_axes,
                )
            )
            # Continuation rows align at post_x_axes (so it visually continues)
            for cont in post_wrapped[1:]:
                rows.append(RowFeatureCont(x_axes=post_x_axes, text=cont))

        # ---- Vertical layout
        top = 0.975
        bottom = 0.03
        nrows = len(rows)
        dy = (top - bottom) / max(nrows, 1)

        text_artists = []
        patch_artists = []

        for i, row in enumerate(rows):
            y = top - (i + 0.5) * dy  # row center

            if isinstance(row, RowPlain):
                t = ax_text.text(
                    row.x_axes, y, row.text,
                    transform=ax_text.transAxes,
                    ha="left", va="center",
                    fontsize=fs,
                    fontproperties=fp,
                )
                text_artists.append(t)
                continue

            if isinstance(row, RowFeatureCont):
                t = ax_text.text(
                    row.x_axes, y, row.text,
                    transform=ax_text.transAxes,
                    ha="left", va="center",
                    fontsize=fs,
                    fontproperties=fp,
                )
                text_artists.append(t)
                continue

            # Feature main row
            score = float(node_map.get(row.fid, NodeInfo(label=row.ftext, score=0.0)).score)
            color = cmap(norm(score))

            # pre
            t_pre = ax_text.text(
                left_margin, y, row.pre,
                transform=ax_text.transAxes,
                ha="left", va="center",
                fontsize=fs,
                fontproperties=fp,
            )
            text_artists.append(t_pre)

            # square (vertically centered on row)
            sq_y = y - row.sq_h / 2.0

            rect = Rectangle(
                (row.sq_x, sq_y),
                row.sq_w,
                row.sq_h,
                transform=ax_text.transAxes,
                facecolor=color,
                edgecolor="black",
                linewidth=1.0,  # fixed thickness (per request)
            )
            ax_text.add_patch(rect)
            patch_artists.append(rect)

            # text inside square (vertically centered)
            t_sq = ax_text.text(
                row.sq_x + pad_x_inner, y, row.ftext,
                transform=ax_text.transAxes,
                ha="left", va="center",
                fontsize=fs,
                fontproperties=fp,
                color="black",
                fontweight="bold" if row.fid in emphasize_ids else "normal",
            )
            text_artists.append(t_sq)

            # post chunk (same row)
            t_post = ax_text.text(
                row.post_x, y, row.post_chunk,
                transform=ax_text.transAxes,
                ha="left", va="center",
                fontsize=fs,
                fontproperties=fp,
            )
            text_artists.append(t_post)

            if show_value_numbers:
                t_val = ax_text.text(
                    max_x, y, f"{score:.3f}",
                    transform=ax_text.transAxes,
                    ha="right", va="center",
                    fontsize=fs,
                    fontproperties=fp,
                    alpha=0.75,
                )
                text_artists.append(t_val)

        # Draw now to get final extents
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()

        # Union bbox for ALL prompt content (texts + colored squares) in DISPLAY coords
        bboxes = []
        for t in text_artists:
            try:
                bboxes.append(t.get_window_extent(renderer=renderer))
            except Exception:
                pass
        for p in patch_artists:
            try:
                bboxes.append(p.get_window_extent(renderer=renderer))
            except Exception:
                pass

        if not bboxes:
            raise RuntimeError("No artists were rendered; cannot compute border bbox.")

        x0 = min(bb.x0 for bb in bboxes)
        y0 = min(bb.y0 for bb in bboxes)
        x1 = max(bb.x1 for bb in bboxes)
        y1 = max(bb.y1 for bb in bboxes)

        # Padding in DISPLAY coords
        pad_x = border_pad_frac * (x1 - x0)
        pad_y = border_pad_frac * (y1 - y0)
        x0p, y0p, x1p, y1p = (x0 - pad_x, y0 - pad_y, x1 + pad_x, y1 + pad_y)

        # Verify: border must NOT include colorbar axis (keep colormap outside the big square)
        cax_bb = cax.get_window_extent(renderer=renderer)
        overlap = x1p >= (cax_bb.x0 - 2.0)  # 2px safety gap
        last_overlap = overlap

        if overlap and attempt < max_fit_attempts - 1:
            plt.close(fig)
            fs = max(7, fs - 1)  # reduce font size and retry
            continue

        # ---- Draw prompt border in FIGURE coordinates (black border)
        inv_fig = fig.transFigure.inverted()
        (fx0, fy0) = inv_fig.transform((x0p, y0p))
        (fx1, fy1) = inv_fig.transform((x1p, y1p))

        border = Rectangle(
            (fx0, fy0),
            (fx1 - fx0),
            (fy1 - fy0),
            transform=fig.transFigure,
            fill=False,
            edgecolor="black",
            linewidth=1.8,
            clip_on=False,
            zorder=10_000,
        )
        fig.add_artist(border)

        # ---- Colorbar (outside prompt border)
        sm = ScalarMappable(norm=norm, cmap=cmap)
        cbar = fig.colorbar(sm, cax=cax, orientation="vertical")
        cbar.set_label("aHFR-TokenSHAP", rotation=90)

        if last_overlap:
            print(
                "[WARN] Prompt border/content still overlaps the colorbar region after fitting attempts.\n"
                "       Consider: --font_size smaller, or increasing figure size, or moving AX_CBAR further right."
            )

        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=220)
        plt.close(fig)
        return

    raise RuntimeError("Failed to render without overlap after attempts.")


# =============================================================================
# Reporting
# =============================================================================
def feature_ids_in_prompt(prompt_text: str) -> List[str]:
    """Return feature ids in appearance order (unique)."""
    seen: Set[str] = set()
    order: List[str] = []
    for ln in prompt_text.splitlines():
        fid = extract_feat_id(ln)
        if fid and fid not in seen:
            seen.add(fid)
            order.append(fid)
    return order


def print_prompt_feature_table(prompt_text: str, node_map: Dict[str, NodeInfo]) -> None:
    order = feature_ids_in_prompt(prompt_text)
    print("\nPrompt feature table:")
    print(f"{'feat_id':28s} {'label':36s} {'aHFR-TokenSHAP':>14s}")
    print("-" * 82)
    for fid in order:
        info = node_map.get(fid, NodeInfo(label=fid, score=float("nan")))
        label = info.label if len(info.label) <= 36 else (info.label[:33] + "...")
        score = info.score
        print(f"{fid:28s} {label:36s} {score:14.9f}")


# =============================================================================
# CLI
# =============================================================================
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Render a structured prompt image with per-feature aHFR-TokenSHAP color squares.\n"
            "Inputs are a prompt text and a hierarchical feature tree (JSON or python literal)."
        )
    )

    ap.add_argument("--prompt_path", type=str, default=None, help="Path to a text file containing the prompt.")
    ap.add_argument("--prompt_text", type=str, default=None, help="Prompt text passed directly on the CLI.")
    ap.add_argument(
        "--feature_tree_path",
        type=str,
        required=True,
        help="Path to hierarchical feature tree (JSON recommended; python literal also supported).",
    )

    ap.add_argument("--out_png", type=str, required=True, help="Output PNG path.")
    ap.add_argument("--cmap", type=str, default="coolwarm", help="Matplotlib colormap name.")
    ap.add_argument("--font_size", type=int, default=11)
    ap.add_argument(
        "--lighten",
        type=float,
        default=0.25,
        help="Blend colormap with white (0=original, 1=white). Lower = more saturated.",
    )
    ap.add_argument("--show_values", action="store_true", help="Print numeric values at right side of rows.")
    ap.add_argument(
        "--border_pad_frac",
        type=float,
        default=0.10,
        help="Prompt border padding as fraction of content bbox size.",
    )
    ap.add_argument(
        "--label_source",
        type=str,
        default="prompt",
        choices=["prompt", "tree"],
        help="Text inside squares: 'prompt' uses prompt text field; 'tree' uses node label from feature tree.",
    )
    ap.add_argument(
        "--emphasize_ids",
        type=str,
        default="",
        help="Comma-separated FEAT_IDs to render in bold inside squares.",
    )
    ap.add_argument("--no_table", action="store_true", help="Disable printing the prompt feature table.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    prompt_text = load_prompt_text(args.prompt_path, args.prompt_text)
    node_map = load_feature_tree(args.feature_tree_path)

    emphasize_ids = {s.strip() for s in (args.emphasize_ids or "").split(",") if s.strip()}

    # sanity check: report any FEAT_IDs in prompt that are missing from tree
    missing = [fid for fid in feature_ids_in_prompt(prompt_text) if fid not in node_map]
    if missing:
        print("[WARN] The following FEAT_IDs occur in the prompt but are missing in the feature tree:")
        for fid in missing:
            print(f"  - {fid}")

    render_prompt_png(
        prompt_text=prompt_text,
        node_map=node_map,
        out_png=Path(args.out_png),
        label_source=args.label_source,
        emphasize_ids=emphasize_ids,
        cmap_name=args.cmap,
        font_size=args.font_size,
        lighten=args.lighten,
        show_value_numbers=args.show_values,
        border_pad_frac=args.border_pad_frac,
    )

    print(f"[OK] Wrote: {args.out_png}")

    if not args.no_table:
        print_prompt_feature_table(prompt_text, node_map)


if __name__ == "__main__":
    main()
