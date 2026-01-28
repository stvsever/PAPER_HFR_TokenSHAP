# ig_attribution.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Callable, Any
import time
import re
import warnings

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM


@dataclass
class LabelTokens:
    case_str: str = " CASE"
    control_str: str = " CONTROL"
    case_ids: List[int] = None
    control_ids: List[int] = None

    @property
    def single_token_labels(self) -> bool:
        return (len(self.case_ids) == 1) and (len(self.control_ids) == 1)


def _pick_device(device: Optional[str] = None, prefer_mps: bool = True) -> str:
    if device is not None:
        return device
    if prefer_mps and torch.backends.mps.is_available() and torch.backends.mps.is_built(): # for OS: macOS with MPS ; I have M2 chip
        return "mps"
    if torch.cuda.is_available(): # for OS: Linux/Windows with CUDA
        return "cuda"
    return "cpu"


def load_model(
    model_name: str = "distilgpt2",
    device: Optional[str] = None,
    prefer_mps: bool = True,
):
    device = _pick_device(device=device, prefer_mps=prefer_mps)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    model.to(device)
    model.eval()

    # Some tokenizers have no pad_token; set to eos to keep batching safe.
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer, device


def _token_ids(tokenizer, text: str) -> List[int]:
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def prepare_label_tokens(tokenizer, case_str: str = " CASE", control_str: str = " CONTROL") -> LabelTokens:
    lt = LabelTokens(case_str=case_str, control_str=control_str)
    lt.case_ids = _token_ids(tokenizer, case_str)
    lt.control_ids = _token_ids(tokenizer, control_str)
    if len(lt.case_ids) == 0 or len(lt.control_ids) == 0:
        raise ValueError("Label strings tokenized to empty sequence; choose different labels.")
    return lt


@torch.no_grad()
def generate_next_token(
    model,
    tokenizer,
    prompt: str,
    device: str,
    max_new_tokens: int = 1,
) -> str:
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"].to(device)

    # Provide attention_mask explicitly (avoids pad==eos ambiguity warnings).
    attention_mask = torch.ones_like(input_ids, device=device)

    out_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    gen_part = out_ids[0, input_ids.shape[1] :]
    return tokenizer.decode(gen_part, skip_special_tokens=True)


@torch.no_grad()
def score_case_control(
    model,
    tokenizer,
    prompt: str,
    labels: LabelTokens,
    device: str,
):
    """
    s = logP(CASE|prompt) - logP(CONTROL|prompt)
    """
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"].to(device)

    if input_ids.shape[1] < 1:
        raise ValueError("Prompt tokenized to empty. Add some text.")

    attention_mask = torch.ones_like(input_ids, device=device)

    if labels.single_token_labels:
        out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        logits_last = out.logits[0, -1, :]
        logprobs_last = F.log_softmax(logits_last, dim=-1)
        logp_case = float(logprobs_last[labels.case_ids[0]].item())
        logp_control = float(logprobs_last[labels.control_ids[0]].item())
        return logp_case - logp_control, logp_case, logp_control

    logp_case = float(_sequence_logprob(model, input_ids, attention_mask, labels.case_ids))
    logp_control = float(_sequence_logprob(model, input_ids, attention_mask, labels.control_ids))
    return logp_case - logp_control, logp_case, logp_control


@torch.no_grad()
def _sequence_logprob(
    model,
    prompt_ids: torch.Tensor,
    prompt_mask: torch.Tensor,
    label_ids: List[int],
) -> torch.Tensor:
    device = prompt_ids.device
    label = torch.tensor(label_ids, device=device).unsqueeze(0)
    full = torch.cat([prompt_ids, label], dim=1)

    # attention mask = 1 everywhere (since we do not pad here)
    full_mask = torch.ones_like(full, device=device)

    out = model(input_ids=full, attention_mask=full_mask, use_cache=False)
    logits = out.logits

    P = prompt_ids.shape[1]
    L = label.shape[1]

    logp = 0.0
    for j in range(L):
        pos = P + j - 1
        lp = F.log_softmax(logits[0, pos, :], dim=-1)[label[0, j]]
        logp = logp + lp
    return logp


def feature_token_indices_from_offsets(
    offsets: List[Tuple[int, int]],
    feature_spans: Dict[str, Tuple[int, int]],
) -> Dict[str, List[int]]:
    feat_to_tokens: Dict[str, List[int]] = {fid: [] for fid in feature_spans.keys()}
    for ti, (ts, te) in enumerate(offsets):
        if te <= ts:
            continue
        for fid, (fs, fe) in feature_spans.items():
            if te > fs and ts < fe:
                feat_to_tokens[fid].append(ti)
    return feat_to_tokens


_VALUE_RE = re.compile(r"value=([+-]?\d+(?:\.\d+)?)")


def _refine_feature_spans(prompt: str, feature_spans: Dict[str, Tuple[int, int]], span_mode: str) -> Dict[str, Tuple[int, int]]:
    """
    span_mode:
      - "line": use the full provided span (often the whole feature line).
      - "value": shrink each feature span to ONLY the numeric value after `value=...`.
               This avoids attributing label-relevant tokens inside FEAT_ID/modality text.
    """
    if span_mode not in ("line", "value"):
        raise ValueError("span_mode must be 'line' or 'value'.")

    if span_mode == "line":
        return dict(feature_spans)

    refined: Dict[str, Tuple[int, int]] = {}
    for fid, (s, e) in feature_spans.items():
        seg = prompt[s:e]
        m = _VALUE_RE.search(seg)
        if m is None:
            refined[fid] = (s, e)
            continue
        # group(1) is the numeric value
        vs = s + m.start(1)
        ve = s + m.end(1)
        refined[fid] = (vs, ve)

    return refined


@torch.no_grad()
def _score_from_input_ids_single_token_labels(
    model,
    input_ids: torch.Tensor,
    labels: LabelTokens,
) -> float:
    """
    Score s = logP(CASE)-logP(CONTROL) from input_ids (single-token labels only).
    """
    if not labels.single_token_labels:
        raise ValueError("This helper only supports single-token labels.")
    attention_mask = torch.ones_like(input_ids, device=input_ids.device)
    out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits_last = out.logits[0, -1, :]
    logprobs_last = F.log_softmax(logits_last, dim=-1)
    return float((logprobs_last[labels.case_ids[0]] - logprobs_last[labels.control_ids[0]]).item())


def integrated_gradients_feature_importance(
    model,
    tokenizer,
    prompt: str,
    feature_spans: Dict[str, Tuple[int, int]],
    labels: LabelTokens,
    device: str,
    steps: int = 8,
    baseline_mode: str = "eos",
    baseline_prompt: Optional[str] = None,
    span_mode: str = "value",
    check_completeness: bool = True,
    completeness_rtol: float = 5e-2,
    progress_cb: Optional[Callable[[int, int, float, float], Any]] = None,
    return_debug: bool = False,
) -> Dict[str, float] | Tuple[Dict[str, float], Dict[str, float]]:
    """
    Integrated Gradients on prompt embeddings for s = logP(CASE)-logP(CONTROL).

    IMPORTANT (for stable "feature importance"):
      - span_mode="value" (default) attributes ONLY the numeric value token(s)
        rather than the entire feature line (FEAT_ID/modality text can dominate).
      - baseline_mode="prompt" is usually the most meaningful baseline if you can
        pass a baseline_prompt with the SAME template but ablated values.

    Parameters
    ----------
    baseline_mode:
      - "zero": baseline embeddings are 0 (cannot compute baseline score for completeness)
      - "eos":  baseline embeddings are eos embedding repeated across positions
      - "prompt": baseline embeddings come from tokenizing baseline_prompt (recommended if available)

    return_debug:
      If True, returns (feat_scores, debug_dict) where debug_dict includes:
        - s_x, s_x0 (when available), delta_s (when available)
        - total_token_attr, completeness_abs_err (when available)
        - span_mode used, baseline_mode used
    """
    if steps < 1:
        raise ValueError("steps must be >= 1")

    if not labels.single_token_labels:
        raise ValueError(
            "Multi-token labels detected. For speed, use labels that tokenize to 1 token each "
            "(leading-space ' CASE'/' CONTROL' usually works for GPT2-family)."
        )

    # Tokenize prompt with offsets for span->token mapping.
    enc = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    input_ids = enc["input_ids"].to(device)
    offsets = enc["offset_mapping"][0].tolist()
    T = input_ids.shape[1]
    if T < 1:
        raise ValueError("Prompt tokenized to empty.")

    # Refine spans (default: numeric value only).
    spans_used = _refine_feature_spans(prompt, feature_spans, span_mode=span_mode)

    embed = model.get_input_embeddings()
    E = embed(input_ids)

    # Build baseline embeddings E0
    baseline_mode = baseline_mode.lower().strip()
    if baseline_mode == "zero":
        E0 = torch.zeros_like(E)
        baseline_ids_for_debug = None
    elif baseline_mode == "eos":
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has no eos_token_id; use baseline_mode='zero' or 'prompt'.")
        eos_id = torch.tensor([[tokenizer.eos_token_id]], device=device)
        eos_emb = embed(eos_id)
        E0 = eos_emb.repeat(1, T, 1)
        baseline_ids_for_debug = torch.full_like(input_ids, tokenizer.eos_token_id)
    elif baseline_mode == "prompt":
        if baseline_prompt is None:
            raise ValueError("baseline_mode='prompt' requires baseline_prompt=str.")
        enc0 = tokenizer(baseline_prompt, return_tensors="pt", add_special_tokens=False)
        ids0 = enc0["input_ids"].to(device)

        # Align length to T for a well-defined baseline in embedding space.
        # (Best practice is to provide a baseline_prompt with identical template/length.)
        if ids0.shape[1] < T:
            if tokenizer.eos_token_id is None:
                raise ValueError("Tokenizer has no eos_token_id to pad baseline_prompt; shorten prompt or use baseline_mode='zero'.")
            pad = torch.full((1, T - ids0.shape[1]), tokenizer.eos_token_id, device=device, dtype=ids0.dtype)
            ids0 = torch.cat([ids0, pad], dim=1)
            warnings.warn(
                f"[IG] baseline_prompt shorter than prompt; padded baseline ids to length {T} with eos_token_id.",
                RuntimeWarning,
            )
        elif ids0.shape[1] > T:
            ids0 = ids0[:, :T]
            warnings.warn(
                f"[IG] baseline_prompt longer than prompt; truncated baseline ids to length {T}.",
                RuntimeWarning,
            )

        E0 = embed(ids0)
        baseline_ids_for_debug = ids0
    else:
        raise ValueError("baseline_mode must be one of: 'eos', 'zero', 'prompt'.")

    delta = E - E0
    grad_sum = torch.zeros_like(E)

    t0 = time.perf_counter()

    for j in range(1, steps + 1):
        alpha = j / steps
        E_alpha = (E0 + alpha * delta).detach().requires_grad_(True)

        attention_mask = torch.ones((1, T), device=device, dtype=torch.long)

        out = model(inputs_embeds=E_alpha, attention_mask=attention_mask, use_cache=False)
        logits_last = out.logits[:, -1, :]
        logprobs_last = F.log_softmax(logits_last, dim=-1)

        s = logprobs_last[0, labels.case_ids[0]] - logprobs_last[0, labels.control_ids[0]]

        model.zero_grad(set_to_none=True)
        if E_alpha.grad is not None:
            E_alpha.grad.zero_()

        s.backward()

        if E_alpha.grad is None:
            raise RuntimeError("IG: E_alpha.grad is None after backward(). Check that inputs_embeds participates in the graph.")

        grad = E_alpha.grad.detach()
        if torch.isnan(grad).any() or torch.isinf(grad).any():
            raise RuntimeError("IG: NaN/Inf detected in gradients. Try fewer steps, different baseline, or run on CPU/CUDA.")

        grad_sum += grad

        if progress_cb is not None:
            elapsed = time.perf_counter() - t0
            rate = elapsed / j
            eta = rate * (steps - j)
            progress_cb(j, steps, elapsed, eta)

    avg_grad = grad_sum / steps
    IG = delta * avg_grad

    # Scalar attribution per token: sum over embedding dims.
    token_scores = IG.sum(dim=-1).squeeze(0)  # [T]

    # Map features -> token indices and aggregate.
    feat_to_tokens = feature_token_indices_from_offsets(offsets, spans_used)
    feat_scores: Dict[str, float] = {}
    for fid, tok_idxs in feat_to_tokens.items():
        feat_scores[fid] = float(token_scores[tok_idxs].sum().item()) if tok_idxs else 0.0

    debug: Dict[str, float] = {
        "steps": float(steps),
        "T": float(T),
        "span_mode": 0.0 if span_mode == "line" else 1.0,
        "baseline_mode_zero/eos/prompt": {"zero": 0.0, "eos": 1.0, "prompt": 2.0}.get(baseline_mode, -1.0),
    }

    # Optional completeness check: sum(token attributions) ≈ s(prompt) - s(baseline)
    if check_completeness and baseline_mode in ("eos", "prompt"):
        with torch.no_grad():
            s_x = _score_from_input_ids_single_token_labels(model, input_ids, labels)
            s_x0 = _score_from_input_ids_single_token_labels(model, baseline_ids_for_debug, labels)
            delta_s = s_x - s_x0
            total_attr = float(token_scores.sum().item())
            abs_err = abs(total_attr - delta_s)

            debug.update(
                {
                    "s_x": float(s_x),
                    "s_x0": float(s_x0),
                    "delta_s": float(delta_s),
                    "total_token_attr": float(total_attr),
                    "completeness_abs_err": float(abs_err),
                }
            )

            denom = abs(delta_s) + 1e-6
            if abs_err > completeness_rtol * denom:
                warnings.warn(
                    f"[IG] Completeness check failed beyond tolerance: "
                    f"sum(attr)={total_attr:+.6f} vs delta_s={delta_s:+.6f} "
                    f"(abs_err={abs_err:.6f}, rtol={completeness_rtol}).",
                    RuntimeWarning,
                )

    if return_debug:
        return feat_scores, debug

    return feat_scores
