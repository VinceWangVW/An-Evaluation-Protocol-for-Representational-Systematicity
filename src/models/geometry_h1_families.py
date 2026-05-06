from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GeometryH1FamilySpec:
    family_id: str
    description: str
    required_config_fields: tuple[str, ...]


GEOMETRY_H1_FAMILY_SPECS = {
    "cross_attention_access_block": GeometryH1FamilySpec(
        family_id="cross_attention_access_block",
        description="Preserve encoder geometry while blocking decoder access through a subset of cross-attention layers.",
        required_config_fields=(
            "decoder_cross_attention_layer_keep_ratio",
            "decoder_cross_attention_layer_seed",
        ),
    ),
    "decoder_relation_subspace_projection": GeometryH1FamilySpec(
        family_id="decoder_relation_subspace_projection",
        description="Preserve encoder geometry while projecting decoder readout states away from a seeded relation subspace.",
        required_config_fields=(
            "decoder_readout_subspace_keep_ratio",
            "decoder_readout_subspace_seed",
            "decoder_readout_subspace_mode",
        ),
    ),
    "cross_attention_bottleneck_projection": GeometryH1FamilySpec(
        family_id="cross_attention_bottleneck_projection",
        description="Preserve encoder geometry while forcing decoder cross-attention outputs through a low-rank bottleneck before decode.",
        required_config_fields=(
            "decoder_cross_attention_bottleneck_keep_ratio",
            "decoder_cross_attention_bottleneck_seed",
            "decoder_cross_attention_bottleneck_mode",
        ),
    ),
    "cross_attention_relation_barrier": GeometryH1FamilySpec(
        family_id="cross_attention_relation_barrier",
        description="Preserve encoder geometry while selectively attenuating a seeded relation-bearing subspace in decoder cross-attention outputs.",
        required_config_fields=(
            "decoder_cross_attention_relation_barrier_keep_ratio",
            "decoder_cross_attention_relation_barrier_seed",
            "decoder_cross_attention_relation_barrier_scale",
        ),
    ),
}


def geometry_h1_family_ids() -> tuple[str, ...]:
    return tuple(GEOMETRY_H1_FAMILY_SPECS.keys())


def get_geometry_h1_family_spec(family_id: str) -> GeometryH1FamilySpec:
    try:
        return GEOMETRY_H1_FAMILY_SPECS[family_id]
    except KeyError as exc:
        raise KeyError(f"Unknown geometry H1 family: {family_id}") from exc


__all__ = [
    "GEOMETRY_H1_FAMILY_SPECS",
    "GeometryH1FamilySpec",
    "geometry_h1_family_ids",
    "get_geometry_h1_family_spec",
]
