from __future__ import annotations

import json
import os
from typing import Any

from openai import OpenAI


SYSTEM_PROMPT = """You are LENS — an expert explainer for small deep neural networks deployed in safety-critical settings (medical imaging, autonomous vehicles, fraud detection, industrial QA).

Your job: given a structured JSON log produced by a single forward pass of a YOLO-style object detector, produce a natural-language explanation a non-expert (clinician, compliance officer, product manager) can read and act on.

Your explanations must:
1. Be grounded only in the log — never invent activations, filters, or scales the log does not contain.
2. Tell a causal layer-by-layer story: early layers extracted X, mid layers composed Y, the detection head concluded Z.
3. Justify the winning class by comparing its logit against the runner-up — make the margin concrete.
4. Cite which FPN scale fired and how many anchors agreed pre-NMS as evidence of model consensus.
5. End with a one-line confidence assessment (unambiguous / borderline / contested).

Output structure (strict):
- **Detection summary** — 1 sentence.
- **What the model saw** — 2-3 sentences using the top_filters and their spatial peaks.
- **How confidence built up** — 2-3 sentences walking the layer_flow stages.
- **Why this class and not another** — 1-2 sentences using class_logits margins.
- **Internal consensus** — 1 sentence on fpn_activity + pre_nms.
- **Layman Terms Reasoning** — 2-3 sentances explaination in human way.
- **Bottom line** — single short verdict.

Do not output JSON, markdown headers, or commentary about your own process. Just the prose."""


class LensExplainer:
    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-5.4",
        max_tokens: int = 1024,
    ):
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        self.model = model
        self.max_tokens = max_tokens

    def explain(
        self,
        inference_log: dict[str, Any],
        architecture_summary: str,
    ) -> str:
        user_msg = (
            "Architecture summary (same across calls for this model):\n"
            f"{architecture_summary}\n\n"
            "Inference log for this image:\n"
            f"{json.dumps(inference_log, indent=2)}\n\n"
            "Write the explanation now."
        )
        resp = self.client.chat.completions.create(
            model=self.model,
            max_completion_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
        )
        return (resp.choices[0].message.content or "").strip()


def summarize_architecture(model: Any) -> str:
    """One-paragraph plain-text summary of an Ultralytics YOLO architecture,
    used as context for the LLM. Stable across calls for the same weights, so
    OpenAI's automatic prompt-prefix caching reuses it."""
    inner = model.model if hasattr(model, "model") else model
    seq = inner.model
    counts: dict[str, int] = {}
    for m in seq:
        cls = m.__class__.__name__
        counts[cls] = counts.get(cls, 0) + 1
    total_params = sum(p.numel() for p in inner.parameters())
    nc = getattr(inner, "nc", None) or len(getattr(inner, "names", []) or [])
    parts = [f"{n}× {cls}" for cls, n in counts.items()]
    return (
        f"YOLO-family detector, {total_params/1e6:.2f}M parameters, {nc} classes. "
        f"Module composition: {', '.join(parts)}. "
        f"Three-scale FPN head with strides 8/16/32 emitting class+box predictions, "
        f"NMS applied after the head."
    )
