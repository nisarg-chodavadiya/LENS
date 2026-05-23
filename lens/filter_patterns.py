from __future__ import annotations

_ROLE_VOCAB: dict[str, list[str]] = {
    "edge_color_extraction": [
        "vertical_edge",
        "horizontal_edge",
        "diagonal_edge",
        "color_blob",
        "high_contrast_boundary",
        "low_frequency_gradient",
    ],
    "shape_composition": [
        "elongated_upright_shape",
        "rounded_curvature",
        "wheel-like_circle",
        "rectangular_corner",
        "limb-like_extension",
        "symmetric_pair",
    ],
    "semantic_feature_encoding": [
        "torso-like_region",
        "facial_feature_cluster",
        "vehicle_body_motif",
        "animal_fur_texture",
        "geometric_object_motif",
        "texture_uniformity",
    ],
    "global_context": [
        "scene_layout_summary",
        "foreground_background_separation",
        "multi-object_aggregation",
    ],
    "multi_scale_fusion": [
        "cross-scale_alignment",
        "fine_detail_injection",
        "context_to_detail_bridge",
    ],
    "final_classification": [
        "class_decision_signal",
    ],
}


def describe_filter(class_name: str, stage_role: str, channel_idx: int) -> str:
    vocab = _ROLE_VOCAB.get(stage_role)
    if not vocab:
        return f"{stage_role}_channel_{channel_idx}"
    return vocab[channel_idx % len(vocab)]
