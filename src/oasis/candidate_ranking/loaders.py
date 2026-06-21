from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from oasis.candidate_ranking.types import (
    MlipAnomalyMetadata,
    MlipModelPrediction,
    ScreeningInputRecord,
    ValidatedReference,
)
from oasis.mlip.artifacts import (
    load_result_json,
    model_name_from_result_path,
)
from oasis.mlip.result_parsing import (
    detect_anomalies_from_result_dict,
    extract_adsorbate,
)

_SITE_METADATA_KEYS = (
    "initial_site_label",
    "initial_site_coordinate",
    "top_layer_motif",
    "surface_type",
    "supercell_size",
)


def _structure_metadata(reaction_payload: dict[str, Any], structure_name: str) -> dict[str, Any]:
    metadata = reaction_payload.get("metadata", {})
    structures = metadata.get("structures", {})
    structure_metadata = structures.get(structure_name, {})
    return dict(structure_metadata) if isinstance(structure_metadata, dict) else {}


def _reference_metadata(reaction_payload: dict[str, Any]) -> dict[str, Any]:
    metadata = reaction_payload.get("metadata", {})
    reference_metadata = metadata.get("reference", {})
    return dict(reference_metadata) if isinstance(reference_metadata, dict) else {}


def _site_metadata(
    reference_metadata: dict[str, Any],
    adslab_metadata: dict[str, Any],
) -> dict[str, Any]:
    source = adslab_metadata or reference_metadata
    return {
        key: source[key]
        for key in _SITE_METADATA_KEYS
        if key in source
    }


def _get_required_identifier(
    *,
    reaction: str,
    field_name: str,
    adslab_metadata: dict[str, Any],
    slab_metadata: dict[str, Any],
    reference_metadata: dict[str, Any],
    result_path: Path,
) -> str:
    for metadata in (adslab_metadata, slab_metadata, reference_metadata):
        value = metadata.get(field_name)
        if isinstance(value, str) and value:
            return value
    raise ValueError(
        f"Missing {field_name!r} for reaction {reaction!r} in {result_path}"
    )


def _validate_matching_record(
    existing: ScreeningInputRecord,
    incoming: ScreeningInputRecord,
    *,
    result_path: Path,
) -> None:
    comparable_fields = (
        "parent_slab_id",
        "adslab_id",
        "adsorbate",
        "reference_binding_energy",
        "site_metadata",
    )
    for field_name in comparable_fields:
        if getattr(existing, field_name) != getattr(incoming, field_name):
            raise ValueError(
                "Inconsistent screening metadata for reaction "
                f"{existing.reaction!r} in {result_path}: field {field_name!r} "
                f"expected {getattr(existing, field_name)!r}, "
                f"got {getattr(incoming, field_name)!r}"
            )


def load_screening_input_records(
    result_files: list[Path],
) -> tuple[ScreeningInputRecord, ...]:
    """Load method-agnostic candidate-screening inputs from MLIP result bundles."""

    if not result_files:
        raise ValueError("At least one MLIP result file is required.")

    merged_records: dict[str, ScreeningInputRecord] = {}
    predictions_by_reaction: dict[str, list[MlipModelPrediction]] = {}

    for result_path in result_files:
        model_name = model_name_from_result_path(result_path)
        payload = load_result_json(result_path)
        anomaly_payloads = detect_anomalies_from_result_dict(payload)

        for reaction, anomaly_payload in anomaly_payloads.items():
            reaction_payload = payload.get(reaction)
            if not isinstance(reaction_payload, dict):
                raise ValueError(
                    f"Missing raw reaction payload for {reaction!r} in {result_path}"
                )

            slab_metadata = _structure_metadata(reaction_payload, "slab")
            adslab_metadata = _structure_metadata(reaction_payload, "adslab")
            reference_metadata = _reference_metadata(reaction_payload)
            incoming_record = ScreeningInputRecord(
                reaction=reaction,
                parent_slab_id=_get_required_identifier(
                    reaction=reaction,
                    field_name="parent_slab_id",
                    adslab_metadata=adslab_metadata,
                    slab_metadata=slab_metadata,
                    reference_metadata=reference_metadata,
                    result_path=result_path,
                ),
                adslab_id=_get_required_identifier(
                    reaction=reaction,
                    field_name="adslab_id",
                    adslab_metadata=adslab_metadata,
                    slab_metadata=slab_metadata,
                    reference_metadata=reference_metadata,
                    result_path=result_path,
                ),
                adsorbate=(
                    adslab_metadata.get("adsorbate")
                    if isinstance(adslab_metadata.get("adsorbate"), str)
                    else extract_adsorbate(reaction)
                ),
                reference_binding_energy=anomaly_payload.get("dft_ads_eng"),
                site_metadata=_site_metadata(reference_metadata, adslab_metadata),
                slab_metadata=slab_metadata,
                adslab_metadata=adslab_metadata,
                shared_metadata=reference_metadata,
            )
            if reaction in merged_records:
                _validate_matching_record(
                    merged_records[reaction],
                    incoming_record,
                    result_path=result_path,
                )
            else:
                merged_records[reaction] = incoming_record

            predictions_by_reaction.setdefault(reaction, []).append(
                MlipModelPrediction(
                    model_name=model_name,
                    result_path=str(result_path),
                    predicted_binding_energy=anomaly_payload.get("mlip_ads_eng_median"),
                    single_point_binding_energy=anomaly_payload.get(
                        "mlip_ads_eng_single"
                    ),
                    anomaly=MlipAnomalyMetadata(
                        label=str(anomaly_payload.get("label", "unknown")),
                        labels=tuple(str(label) for label in anomaly_payload.get("labels", ())),
                        details={
                            str(detail_name): int(detail_value)
                            for detail_name, detail_value in dict(
                                anomaly_payload.get("details", {})
                            ).items()
                        },
                    ),
                )
            )

    records: list[ScreeningInputRecord] = []
    for reaction in sorted(merged_records):
        record = merged_records[reaction]
        model_predictions = tuple(
            sorted(
                predictions_by_reaction.get(reaction, []),
                key=lambda prediction: prediction.model_name,
            )
        )
        records.append(
            ScreeningInputRecord(
                reaction=record.reaction,
                parent_slab_id=record.parent_slab_id,
                adslab_id=record.adslab_id,
                adsorbate=record.adsorbate,
                reference_binding_energy=record.reference_binding_energy,
                site_metadata=dict(record.site_metadata),
                slab_metadata=dict(record.slab_metadata),
                adslab_metadata=dict(record.adslab_metadata),
                shared_metadata=dict(record.shared_metadata),
                model_predictions=model_predictions,
            )
        )
    return tuple(records)


def load_validated_references(
    path: Path,
) -> tuple[ValidatedReference, ...]:
    """Load validated DFT references from one JSON file."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_references = (
        payload.get("validated_references", ())
        if isinstance(payload, dict)
        else payload
    )
    if not isinstance(raw_references, list):
        raise ValueError(
            f"Validated references file {path} must contain a JSON list or "
            "an object with a 'validated_references' list."
        )
    return tuple(ValidatedReference(**dict(reference)) for reference in raw_references)
