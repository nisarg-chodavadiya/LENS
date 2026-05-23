from __future__ import annotations

from typing import Any

import numpy as np

from .filter_patterns import describe_filter
from .logger import ActivationLog, LayerRecord, _describe_region


def _bbox_region(box_xyxy: list[float], img_w: int, img_h: int) -> str:
    cx = (box_xyxy[0] + box_xyxy[2]) / 2
    cy = (box_xyxy[1] + box_xyxy[3]) / 2
    return _describe_region(int(cx), int(cy), img_w, img_h)


def _top_filters(layers: list[LayerRecord], k: int = 3) -> list[dict[str, Any]]:
    # Pick the strongest-firing layer in each meaningful stage (skip auxiliary).
    grouped: dict[str, LayerRecord] = {}
    for r in layers:
        if r.stage_role in ("auxiliary", "final_classification"):
            continue
        cur = grouped.get(r.stage_role)
        if cur is None or r.max_abs > cur.max_abs:
            grouped[r.stage_role] = r
    ranked = sorted(grouped.values(), key=lambda r: r.max_abs, reverse=True)[:k]
    return [
        {
            "layer": r.name,
            "pattern": describe_filter(r.class_name, r.stage_role, r.peak_channel),
            "strength": round(r.max_abs, 3),
            "peak_xy": list(r.peak_xy),
        }
        for r in ranked
    ]


def _layer_flow(layers: list[LayerRecord]) -> list[dict[str, Any]]:
    # Aggregate stats per stage; confidence_proxy is mean_abs normalized.
    stages: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for r in layers:
        role = r.stage_role
        if role == "auxiliary":
            continue
        if role not in stages:
            stages[role] = {"role": role, "indices": [], "mean_abs": []}
            order.append(role)
        stages[role]["indices"].append(r.index)
        stages[role]["mean_abs"].append(r.mean_abs)
    if not stages:
        return []
    all_means = [m for s in stages.values() for m in s["mean_abs"]]
    norm = max(all_means) if all_means else 1.0
    out = []
    for role in order:
        s = stages[role]
        lo, hi = min(s["indices"]), max(s["indices"])
        stage_label = f"layers_{lo}-{hi}" if lo != hi else f"layer_{lo}"
        proxy = (sum(s["mean_abs"]) / len(s["mean_abs"])) / norm if norm else 0.0
        out.append(
            {
                "stage": stage_label,
                "role": role,
                "confidence_proxy": round(float(proxy), 3),
            }
        )
    return out


def _class_logits(
    detect_raw: list[np.ndarray],
    nc: int,
    class_names: dict[int, str],
    winning_class: int,
    top_n: int = 4,
) -> dict[str, float]:
    # Each Detect head is [B, C, H, W] where C = (reg*4 + nc). Take the channels
    # corresponding to class scores, find max-scoring anchor for the winning
    # class, and report logits at that anchor location across top-N classes.
    if not detect_raw:
        return {}
    best = None  # (head_idx, c_y, c_x, score)
    for hi, head in enumerate(detect_raw):
        if head.ndim != 4:
            continue
        ch = head.shape[1]
        if ch <= nc:
            continue
        cls_slice = head[0, ch - nc :, :, :]  # [nc, H, W]
        win_map = cls_slice[winning_class]
        idx = int(np.argmax(win_map))
        y, x = divmod(idx, win_map.shape[-1])
        score = float(win_map[y, x])
        if best is None or score > best[3]:
            best = (hi, y, x, score)
    if best is None:
        return {}
    hi, y, x, _ = best
    head = detect_raw[hi]
    ch = head.shape[1]
    logits = head[0, ch - nc :, y, x]
    order = np.argsort(-logits)[:top_n]
    return {class_names.get(int(c), f"class_{int(c)}"): round(float(logits[c]), 3) for c in order}


def _fpn_activity(detect_raw: list[np.ndarray], nc: int, conf_thresh: float = 0.25) -> dict[str, Any]:
    out: dict[str, Any] = {}
    strides = (8, 16, 32)
    for hi, head in enumerate(detect_raw):
        if head.ndim != 4:
            continue
        ch = head.shape[1]
        if ch <= nc:
            continue
        cls = head[0, ch - nc :, :, :]
        # sigmoid for conf approximation
        prob = 1.0 / (1.0 + np.exp(-cls))
        fired = int(np.sum(prob.max(axis=0) > conf_thresh))
        stride = strides[hi] if hi < len(strides) else 2 ** (hi + 3)
        out[f"P{hi + 3}_stride{stride}"] = {
            "active": fired > 0,
            "anchors_fired": fired,
        }
    return out


def _pre_nms_stats(detect_raw: list[np.ndarray], nc: int, n_final: int) -> dict[str, int]:
    cands, high = 0, 0
    for head in detect_raw:
        if head.ndim != 4:
            continue
        ch = head.shape[1]
        if ch <= nc:
            continue
        cls = head[0, ch - nc :, :, :]
        prob = 1.0 / (1.0 + np.exp(-cls))
        per_anchor_max = prob.max(axis=0).reshape(-1)
        cands += int(np.sum(per_anchor_max > 0.25))
        high += int(np.sum(per_anchor_max > 0.5))
    return {
        "candidates": cands,
        "high_confidence": high,
        "final_detections": n_final,
    }


def build_inference_log(
    activation_log: ActivationLog,
    detection: dict[str, Any] | None,
    nc: int,
) -> dict[str, Any]:
    """Compose the full LENS inference log.

    `detection` is one row from Ultralytics results: {class, confidence, bbox}.
    Pass None if no detection survived NMS.
    """
    img_w, img_h = activation_log.image_size
    names = activation_log.class_names

    if detection is not None:
        region = _bbox_region(detection["bbox"], img_w, img_h)
        activation_log.input_region = region
        win_cls = detection["class_id"]
        n_final = 1
    else:
        win_cls = 0
        n_final = 0

    log = activation_log.to_dict()
    log["prediction"] = (
        {
            "class": detection["class"],
            "confidence": round(float(detection["confidence"]), 3),
            "bbox": [round(float(v), 1) for v in detection["bbox"]],
        }
        if detection is not None
        else {"class": None, "confidence": 0.0, "bbox": []}
    )
    log["class_logits"] = _class_logits(activation_log.detect_raw, nc, names, win_cls)
    log["top_filters"] = _top_filters(activation_log.layers)
    log["layer_flow"] = _layer_flow(activation_log.layers)
    log["fpn_activity"] = _fpn_activity(activation_log.detect_raw, nc)
    log["pre_nms"] = _pre_nms_stats(activation_log.detect_raw, nc, n_final)
    # Drop the raw per-layer dump from final JSON to keep prompts compact;
    # callers who want it can read activation_log.layers directly.
    log.pop("layers", None)
    return log
