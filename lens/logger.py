from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import nn


# Map Ultralytics layer class names → semantic stage roles used in the JSON log.
# Anything not listed falls through to "auxiliary".
_STAGE_BY_CLASS: dict[str, str] = {
    "Conv": "edge_color_extraction",
    "DWConv": "edge_color_extraction",
    "GhostConv": "edge_color_extraction",
    "C2f": "shape_composition",
    "C3": "shape_composition",
    "C3k2": "shape_composition",
    "C2PSA": "semantic_feature_encoding",
    "PSA": "semantic_feature_encoding",
    "SPPF": "global_context",
    "SPP": "global_context",
    "Concat": "multi_scale_fusion",
    "Upsample": "multi_scale_fusion",
    "Detect": "final_classification",
    "v10Detect": "final_classification",
}


@dataclass
class LayerRecord:
    index: int
    name: str
    class_name: str
    stage_role: str
    out_shape: tuple[int, ...]
    mean_abs: float
    max_abs: float
    peak_channel: int
    peak_xy: tuple[int, int]


@dataclass
class ActivationLog:
    """In-memory record of a single forward pass. Convert with `.to_dict()`."""

    image_path: str = ""
    image_size: tuple[int, int] = (0, 0)
    input_region: str = ""
    layers: list[LayerRecord] = field(default_factory=list)
    detect_raw: list[np.ndarray] = field(default_factory=list)
    class_names: dict[int, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "input": {
                "image_path": self.image_path,
                "image_size": list(self.image_size),
                "image_region_of_detection": self.input_region,
            },
            "layers": [
                {
                    "index": r.index,
                    "name": r.name,
                    "class": r.class_name,
                    "role": r.stage_role,
                    "out_shape": list(r.out_shape),
                    "mean_abs": round(r.mean_abs, 4),
                    "max_abs": round(r.max_abs, 4),
                    "peak_channel": r.peak_channel,
                    "peak_xy": list(r.peak_xy),
                }
                for r in self.layers
            ],
        }


def _describe_region(x: int, y: int, w: int, h: int) -> str:
    vert = "upper" if y < h / 3 else "lower" if y > 2 * h / 3 else "middle"
    horz = "left" if x < w / 3 else "right" if x > 2 * w / 3 else "center"
    return f"{vert}-{horz}"


class ActivationLogger:
    """Attach to an Ultralytics YOLO model; collects per-layer stats on every
    forward pass. Use as a context manager so hooks are cleaned up reliably."""

    def __init__(self, model: Any):
        self.ultra_model = model
        self.inner: nn.Module = model.model if hasattr(model, "model") else model
        self.sequential = self.inner.model
        self._handles: list[Any] = []
        self.log = ActivationLog()
        if hasattr(self.inner, "names"):
            names = self.inner.names
            self.log.class_names = dict(names) if isinstance(names, dict) else {
                i: n for i, n in enumerate(names)
            }

    def __enter__(self) -> "ActivationLogger":
        self._attach()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def _attach(self) -> None:
        for idx, module in enumerate(self.sequential):
            cls = module.__class__.__name__
            role = _STAGE_BY_CLASS.get(cls, "auxiliary")
            h = module.register_forward_hook(self._make_hook(idx, cls, role))
            self._handles.append(h)

    def _make_hook(self, idx: int, cls: str, role: str):
        def hook(module: nn.Module, inputs, output):
            tensor = output[0] if isinstance(output, (list, tuple)) and isinstance(output[0], torch.Tensor) else output
            if cls in ("Detect", "v10Detect"):
                # Detect emits the multi-scale raw heads; grab them before NMS.
                raws = output[1] if isinstance(output, tuple) and len(output) > 1 else output
                if isinstance(raws, (list, tuple)):
                    self.log.detect_raw = [t.detach().float().cpu().numpy() for t in raws if isinstance(t, torch.Tensor)]
            if not isinstance(tensor, torch.Tensor) or tensor.dim() < 3:
                return
            with torch.no_grad():
                t = tensor.detach().float()
                abs_t = t.abs()
                mean_abs = abs_t.mean().item()
                max_abs = abs_t.max().item()
                # Channel of largest mean magnitude
                ch_means = abs_t.mean(dim=tuple(range(2, t.dim())))[0]
                peak_channel = int(ch_means.argmax().item())
                # Spatial peak on that channel
                spatial = abs_t[0, peak_channel]
                if spatial.dim() == 2:
                    flat = spatial.argmax().item()
                    py, px = divmod(int(flat), spatial.shape[-1])
                else:
                    py, px = 0, 0
            self.log.layers.append(
                LayerRecord(
                    index=idx,
                    name=f"{cls}_{idx}",
                    class_name=cls,
                    stage_role=role,
                    out_shape=tuple(t.shape),
                    mean_abs=mean_abs,
                    max_abs=max_abs,
                    peak_channel=peak_channel,
                    peak_xy=(px, py),
                )
            )

        return hook

    def set_image_metadata(self, path: str, width: int, height: int) -> None:
        self.log.image_path = path
        self.log.image_size = (width, height)
