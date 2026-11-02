"""Hardware-bounded dataset fingerprint matching and reconfiguration planning."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


FINGERPRINT_FEATURES = (
    "event_count_log_mean",
    "event_count_cv",
    "duration_mean",
    "duration_cv",
    "temporal_entropy",
    "channel_entropy",
    "temporal_occupancy",
    "channel_occupancy",
    "feature_rank",
    "ordered_channel_count",
    "spatial_axes",
    "polarity_channels",
    "class_count",
)

HARD_COMPATIBILITY_FIELDS = (
    "feature_rank",
    "ordered_channel_count",
    "spatial_axes",
    "polarity_channels",
    "class_count",
)


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


class DecisionEngine:
    """Select only profiles that the current RTL can actually execute."""

    def __init__(self, profile_catalog: dict[str, Any], prototypes: dict[str, Any]):
        self.profile_catalog = profile_catalog
        self.prototypes = prototypes
        self._profiles = {
            item["profile_id"]: item for item in profile_catalog["profiles"]
        }
        self._prototype_rows = prototypes["prototypes"]
        self._validate_catalogs()
        self._scales = self._build_feature_scales()

    @classmethod
    def from_files(
        cls, profile_catalog_path: str | Path, prototype_path: str | Path
    ) -> "DecisionEngine":
        return cls(load_json(profile_catalog_path), load_json(prototype_path))

    def decide(
        self,
        request: dict[str, Any],
        learning_request: str = "auto",
    ) -> dict[str, Any]:
        fingerprint = request.get("fingerprint", request)
        self._validate_fingerprint(fingerprint)
        if learning_request not in {"auto", "on", "off"}:
            raise ValueError("learning_request must be one of: auto, on, off")

        compatible: list[dict[str, Any]] = []
        incompatible: list[dict[str, Any]] = []
        all_distances: list[dict[str, Any]] = []

        for row in self._prototype_rows:
            profile = self._profiles[row["profile_id"]]
            distance = self._distance(fingerprint, row["fingerprint"])
            mismatches = self._compatibility_mismatches(fingerprint, row["fingerprint"])
            item = {
                "profile_id": profile["profile_id"],
                "profile_code": profile["profile_code"],
                "dispatch_contract_id": profile["dispatch_contract_id"],
                "distance": round(distance, 8),
                "mismatches": mismatches,
            }
            all_distances.append(item)
            if profile.get("hardware_selectable", False) and not mismatches:
                compatible.append(item)
            else:
                incompatible.append(item)

        compatible.sort(key=lambda item: item["distance"])
        all_distances.sort(key=lambda item: item["distance"])

        if not compatible:
            nearest = all_distances[0]
            return {
                "schema_version": "1.0",
                "status": "requires_accelerator_extension",
                "selected_profile": None,
                "hardware_plan": None,
                "nearest_architecture_advisory": nearest,
                "compatibility_rejections": incompatible,
                "ranking_note": (
                    "Distance is an engineering similarity score, not a probability or "
                    "model-accuracy confidence. No hardware opcode was emitted."
                ),
            }

        selected_rank = compatible[0]
        profile = self._profiles[selected_rank["profile_id"]]
        learning = self._resolve_learning(profile, request, learning_request)
        margin = None
        if len(compatible) > 1:
            margin = round(compatible[1]["distance"] - compatible[0]["distance"], 8)

        return {
            "schema_version": "1.0",
            "status": "hardware_profile_selected",
            "selected_profile": {
                "profile_id": profile["profile_id"],
                "profile_code": profile["profile_code"],
                "dispatch_contract_id": profile["dispatch_contract_id"],
                "topology": profile["topology"],
                "reference_workload": profile["reference_workload"],
                "input_contract": profile["input_contract"],
                "numeric_contract": profile["numeric_contract"],
                "deployment_assets": profile["deployment_assets"],
                "learning_scope": profile["learning_scope"],
            },
            "ranking": {
                "compatible_profiles": compatible,
                "selected_distance": compatible[0]["distance"],
                "distance_margin_to_next_compatible": margin,
                "note": (
                    "Normalized prototype distance is used only after hard hardware "
                    "compatibility filtering; it is not a probability."
                ),
            },
            "learning_decision": learning,
            "hardware_plan": self._hardware_plan(profile, learning["enabled"]),
            "claim_boundary": (
                "Autonomous selection is limited to the registered hardware profile "
                "library. Adding a topology, depth, precision, or neuron scale requires "
                "offline profile qualification and may require RTL changes."
            ),
        }

    def _validate_catalogs(self) -> None:
        if not self._profiles:
            raise ValueError("profile catalog is empty")
        for row in self._prototype_rows:
            if row["profile_id"] not in self._profiles:
                raise ValueError(f"prototype references unknown profile {row['profile_id']}")
            self._validate_fingerprint(row["fingerprint"])

    def _validate_fingerprint(self, fingerprint: dict[str, Any]) -> None:
        missing = [key for key in FINGERPRINT_FEATURES if key not in fingerprint]
        if missing:
            raise ValueError(f"fingerprint is missing required keys: {', '.join(missing)}")
        if int(fingerprint["spatial_axes"]) == 2 and "spatial_shape" not in fingerprint:
            raise ValueError("2D fingerprints require spatial_shape")
        for key in FINGERPRINT_FEATURES:
            value = fingerprint[key]
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"fingerprint field {key} must be a finite number")

    def _build_feature_scales(self) -> dict[str, float]:
        scales: dict[str, float] = {}
        for key in FINGERPRINT_FEATURES:
            values = [float(row["fingerprint"][key]) for row in self._prototype_rows]
            span = max(values) - min(values)
            mean_abs = sum(abs(value) for value in values) / len(values)
            scales[key] = max(span, mean_abs * 0.1, 1e-6)
        return scales

    def _distance(self, query: dict[str, Any], prototype: dict[str, Any]) -> float:
        squared = [
            ((float(query[key]) - float(prototype[key])) / self._scales[key]) ** 2
            for key in FINGERPRINT_FEATURES
        ]
        return math.sqrt(sum(squared) / len(squared))

    @staticmethod
    def _compatibility_mismatches(
        query: dict[str, Any], prototype: dict[str, Any]
    ) -> list[dict[str, Any]]:
        mismatches: list[dict[str, Any]] = []
        for key in HARD_COMPATIBILITY_FIELDS:
            if int(query[key]) != int(prototype[key]):
                mismatches.append(
                    {"field": key, "query": query[key], "required": prototype[key]}
                )
        if int(prototype["spatial_axes"]) == 2:
            query_shape = [int(value) for value in query.get("spatial_shape", [])]
            required_shape = [int(value) for value in prototype["spatial_shape"]]
            if query_shape != required_shape:
                mismatches.append(
                    {
                        "field": "spatial_shape",
                        "query": query_shape,
                        "required": required_shape,
                    }
                )
        return mismatches

    @staticmethod
    def _resolve_learning(
        profile: dict[str, Any], request: dict[str, Any], learning_request: str
    ) -> dict[str, Any]:
        supported = profile["learning_scope"].get("online_supported", False)
        if learning_request == "on":
            if not supported:
                raise ValueError(f"profile {profile['profile_id']} does not support online learning")
            return {
                "enabled": True,
                "source": "explicit_request",
                "policy_status": "not_applicable",
            }
        if learning_request == "off":
            return {
                "enabled": False,
                "source": "explicit_request",
                "policy_status": "not_applicable",
            }

        health = request.get("runtime_health")
        if not health:
            return {
                "enabled": False,
                "source": "auto_without_runtime_health",
                "policy_status": "engineering_policy_unvalidated",
                "reason": "No confidence or drift signal was supplied.",
            }
        baseline = float(health.get("baseline_confidence", 0.0))
        observed = float(health.get("observed_confidence", baseline))
        drift = float(health.get("drift_score", 0.0))
        confidence_drop = baseline - observed
        enabled = supported and (confidence_drop >= 0.10 or drift >= 0.25)
        return {
            "enabled": enabled,
            "source": "auto_runtime_health",
            "policy_status": "engineering_policy_unvalidated",
            "observed_confidence_drop": round(confidence_drop, 6),
            "observed_drift_score": round(drift, 6),
            "thresholds": {"confidence_drop": 0.10, "drift_score": 0.25},
            "reason": (
                "Threshold crossed and profile supports online learning."
                if enabled
                else "No threshold crossed or online learning is unsupported."
            ),
        }

    @staticmethod
    def _hardware_plan(profile: dict[str, Any], learning_enabled: bool) -> dict[str, Any]:
        return {
            "control_transport": "native_profile_descriptor_handshake",
            "native_interface_contract": "catalog/native_control_interface.json",
            "profile_dispatch_contract": profile["dispatch_contract_id"],
            "mmio_address_map_status": "not_assigned",
            "profile_opcode": profile["profile_code"],
            "learning_enable": learning_enabled,
            "steps": [
                "stop_accepting_new_work",
                "wait_for_owner_drain_and_quiescent",
                "load_qualified_weights_state_and_descriptors_if_needed",
                "request_profile_opcode_and_learning_enable",
                "wait_for_profile_alignment_and_version_visibility",
                "submit_runtime_descriptors",
                (
                    "wait_for_learning_commit_and_published_version_before_next_inference"
                    if learning_enabled
                    else "continue_inference_without_learning_commit"
                ),
            ],
        }
