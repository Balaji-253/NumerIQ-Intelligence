from __future__ import annotations

import math
from typing import Any


def _validate_top5(prediction: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = prediction.get("top5")

    if not isinstance(candidates, list) or not candidates:
        raise ValueError("Each part must contain a nonempty top5 list.")

    validated = []
    seen = set()

    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise TypeError("Each top5 candidate must be a dictionary.")

        try:
            digit = int(candidate["digit"])
            probability = float(candidate["probability"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Invalid top5 candidate.") from exc

        if not 0 <= digit <= 9:
            raise ValueError("Candidate digit must be between 0 and 9.")

        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("Candidate probability must be finite and in [0, 1].")

        if digit not in seen:
            validated.append({
                "digit": digit,
                "probability": probability,
            })
            seen.add(digit)

    return validated


def _expand_hypothesis(
    hypothesis: dict[str, Any],
    beam_width: int,
) -> list[dict[str, Any]]:
    parts = hypothesis.get("part_predictions", [])

    if not parts:
        return []

    beams = [{
        "sequence": "",
        "log_probability_sum": 0.0,
        "digit_probabilities": [],
    }]

    for part in parts:
        candidates = _validate_top5(part)
        expanded = []

        for beam in beams:
            for candidate in candidates:
                probability = max(candidate["probability"], 1e-12)

                expanded.append({
                    "sequence": beam["sequence"] + str(candidate["digit"]),
                    "log_probability_sum": (
                        beam["log_probability_sum"] + math.log(probability)
                    ),
                    "digit_probabilities": (
                        beam["digit_probabilities"] + [candidate["probability"]]
                    ),
                })

        expanded.sort(
            key=lambda item: item["log_probability_sum"],
            reverse=True,
        )
        beams = expanded[:beam_width]

    for beam in beams:
        count = len(beam["digit_probabilities"])
        beam["mean_log_probability"] = (
            beam["log_probability_sum"] / count
        )
        beam["geometric_mean_probability"] = math.exp(
            beam["mean_log_probability"]
        )

    return beams


def decode_v5(
    scored_components: list[dict[str, Any]],
    beam_width: int = 64,
    max_results: int = 20,
) -> dict[str, Any]:
    """
    Build ranked sequence hypotheses from V4 component alternatives.

    This is a diagnostic decoder, not a validated recognition decision.
    CNN scores and geometry scores remain separate.
    No digit count is imposed and no candidate is automatically accepted.
    """
    if not 1 <= beam_width <= 512:
        raise ValueError("beam_width must be between 1 and 512.")

    if not 1 <= max_results <= 100:
        raise ValueError("max_results must be between 1 and 100.")

    if not isinstance(scored_components, list):
        raise TypeError("scored_components must be a list.")

    if not scored_components:
        return {
            "engine": "numeriq-v5.2",
            "status": "no_components",
            "sequence": None,
            "digit_count": None,
            "candidates": [],
            "selection_performed": False,
            "automatic_acceptance": False,
            "warnings": ["No components were available for sequence decoding."],
        }

    ordered = sorted(
        scored_components,
        key=lambda item: (
            (item.get("image_bbox") or (item["original_bbox"]))[0]
            + (item.get("image_bbox") or (item["original_bbox"]))[2] / 2
        ),
    )

    ordering_ambiguity = False
    previous_right = None

    for component in ordered:
        bbox = component.get("image_bbox") or component.get("original_bbox")

        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            raise ValueError("Every component must contain a valid bounding box.")

        x, _, width, _ = bbox
        if previous_right is not None and x < previous_right:
            ordering_ambiguity = True
        previous_right = max(previous_right or x, x + width)

    component_options = []
    original_component_options = []

    for component in ordered:
        hypotheses = component.get("hypotheses", [])
        options = []

        original_options = []

        for hypothesis in hypotheses:
            expanded = _expand_hypothesis(hypothesis, beam_width)

            if hypothesis.get("is_original") is True:
                for candidate in expanded:
                    original_options.append({
                        **candidate,
                        "component_id": component.get("component_id"),
                        "hypothesis_id": hypothesis.get("hypothesis_id"),
                        "strategy": hypothesis.get("strategy"),
                        "geometry_score": hypothesis.get("geometry_score"),
                        "part_count": hypothesis.get("part_count"),
                    })

            for candidate in expanded:
                options.append({
                    **candidate,
                    "component_id": component.get("component_id"),
                    "hypothesis_id": hypothesis.get("hypothesis_id"),
                    "strategy": hypothesis.get("strategy"),
                    "geometry_score": hypothesis.get("geometry_score"),
                    "part_count": hypothesis.get("part_count"),
                })

        if not options:
            return {
                "engine": "numeriq-v5.2",
                "status": "incomplete_hypotheses",
                "sequence": None,
                "digit_count": None,
                "candidates": [],
                "selection_performed": False,
                "automatic_acceptance": False,
                "warnings": [
                    "At least one component had no usable segmentation hypotheses."
                ],
            }

        # V5.2: retain candidates from every segmentation hypothesis.
        # Do not globally truncate component options by CNN score here.
        component_options.append(options)

        original_options.sort(
            key=lambda item: item["mean_log_probability"],
            reverse=True,
        )
        if not original_options:
            original_component_options = []
        elif original_component_options is not None:
            original_component_options.append(original_options[:beam_width])

    global_beams = [{
        "sequence": "",
        "log_probability_sum": 0.0,
        "digit_probabilities": [],
        "component_paths": [],
        "geometry_scores": [],
    }]

    for options in component_options:
        expanded = []

        for beam in global_beams:
            for option in options:
                probabilities = beam["digit_probabilities"] + (
                    option["digit_probabilities"]
                )

                expanded.append({
                    "sequence": beam["sequence"] + option["sequence"],
                    "log_probability_sum": (
                        beam["log_probability_sum"]
                        + option["log_probability_sum"]
                    ),
                    "digit_probabilities": probabilities,
                    "component_paths": beam["component_paths"] + [{
                        "component_id": option["component_id"],
                        "hypothesis_id": option["hypothesis_id"],
                        "strategy": option["strategy"],
                        "sequence": option["sequence"],
                        "part_count": option["part_count"],
                    }],
                    "geometry_scores": beam["geometry_scores"] + [
                        option["geometry_score"]
                    ],
                })

        # Preserve a separate beam for each complete segmentation path.
        # This prevents one high-scoring segmentation from eliminating
        # every candidate belonging to a different segmentation path.
        path_groups = {}

        for item in expanded:
            signature = tuple(
                entry["hypothesis_id"]
                for entry in item["component_paths"]
            )
            path_groups.setdefault(signature, []).append(item)

        ranked_paths = []

        for signature, group in path_groups.items():
            group.sort(
                key=lambda item: (
                    item["log_probability_sum"]
                    / max(len(item["digit_probabilities"]), 1)
                ),
                reverse=True,
            )
            ranked_paths.append((signature, group[:beam_width]))

        # Bound memory and CPU use while retaining the strongest paths.
        ranked_paths.sort(
            key=lambda entry: (
                entry[1][0]["log_probability_sum"]
                / max(len(entry[1][0]["digit_probabilities"]), 1)
            ),
            reverse=True,
        )

        global_beams = [
            item
            for _, group in ranked_paths[:512]
            for item in group
        ]

    # Independently decode the all-original segmentation path.
    # This branch is protected from competition with split hypotheses.
    original_path_beams = [{
        "sequence": "",
        "log_probability_sum": 0.0,
        "digit_probabilities": [],
        "component_paths": [],
        "geometry_scores": [],
    }]

    if len(original_component_options) == len(ordered):
        for options in original_component_options:
            expanded = []

            for beam in original_path_beams:
                for option in options:
                    expanded.append({
                        "sequence": beam["sequence"] + option["sequence"],
                        "log_probability_sum": (
                            beam["log_probability_sum"]
                            + option["log_probability_sum"]
                        ),
                        "digit_probabilities": (
                            beam["digit_probabilities"]
                            + option["digit_probabilities"]
                        ),
                        "component_paths": beam["component_paths"] + [{
                            "component_id": option["component_id"],
                            "hypothesis_id": option["hypothesis_id"],
                            "strategy": option["strategy"],
                            "sequence": option["sequence"],
                            "part_count": option["part_count"],
                        }],
                        "geometry_scores": (
                            beam["geometry_scores"] + [option["geometry_score"]]
                        ),
                    })

            expanded.sort(
                key=lambda item: (
                    item["log_probability_sum"]
                    / max(len(item["digit_probabilities"]), 1)
                ),
                reverse=True,
            )
            original_path_beams = expanded[:beam_width]

    # Retain distinct strings; different segmentation paths remain traceable.
    unique = {}
    for beam in global_beams:
        count = len(beam["digit_probabilities"])
        mean_log = beam["log_probability_sum"] / count

        candidate = {
            "sequence": beam["sequence"],
            "digit_count": len(beam["sequence"]),
            "mean_log_probability": mean_log,
            "geometric_mean_probability": math.exp(mean_log),
            "component_paths": beam["component_paths"],
            "geometry_scores": beam["geometry_scores"],
        }

        existing = unique.get(beam["sequence"])
        if (
            existing is None
            or candidate["mean_log_probability"]
            > existing["mean_log_probability"]
        ):
            unique[beam["sequence"]] = candidate

    # Preserve up to five candidates from the all-original path.
    # These are diagnostic alternatives, not automatically accepted answers.
    protected_original_sequences = set()

    for beam in original_path_beams[:min(5, max_results)]:
        count = len(beam["digit_probabilities"])
        if count == 0:
            continue

        mean_log = beam["log_probability_sum"] / count
        candidate = {
            "sequence": beam["sequence"],
            "digit_count": len(beam["sequence"]),
            "mean_log_probability": mean_log,
            "geometric_mean_probability": math.exp(mean_log),
            "component_paths": beam["component_paths"],
            "geometry_scores": beam["geometry_scores"],
            "preserved_original_path": True,
        }

        protected_original_sequences.add(beam["sequence"])
        # The protected branch owns this sequence's original-path trace.
        # Keep its score and component paths together as one consistent record.
        unique[beam["sequence"]] = candidate

    ranked_all = sorted(
        unique.values(),
        key=lambda item: item["mean_log_probability"],
        reverse=True,
    )

    # V5.2: select candidates with segmentation-path diversity.
    # First retain the strongest candidate from each distinct path.
    # Then fill remaining slots using the overall probability ranking.
    path_groups = {}

    for item in ranked_all:
        signature = tuple(
            entry["hypothesis_id"]
            for entry in item["component_paths"]
        )
        path_groups.setdefault(signature, []).append(item)

    path_representatives = []

    for signature, group in path_groups.items():
        group.sort(
            key=lambda item: item["mean_log_probability"],
            reverse=True,
        )
        path_representatives.append(group[0])

    path_representatives.sort(
        key=lambda item: item["mean_log_probability"],
        reverse=True,
    )

    selected = path_representatives[:max_results]
    selected_sequences = {item["sequence"] for item in selected}

    # Fill unused slots without duplicating sequences.
    for item in ranked_all:
        if len(selected) >= max_results:
            break
        if item["sequence"] not in selected_sequences:
            selected.append(item)
            selected_sequences.add(item["sequence"])

    ranked = selected

    for rank, candidate in enumerate(ranked, start=1):
        candidate["rank"] = rank

    return {
        "engine": "numeriq-v5.2",
        "status": "sequence_hypotheses_generated",
        "sequence": None,
        "digit_count": None,
        "candidates": ranked,
        "candidate_count": len(ranked),
        "component_count": len(ordered),
        "ordering_ambiguity": ordering_ambiguity,
        "scoring": {
            "method": "mean_log_top5_cnn_probability",
            "geometry_scored_separately": True,
            "probabilities_calibrated": False,
        },
        "selection_performed": False,
        "automatic_acceptance": False,
        "warnings": [
            "Candidate ranking is experimental and is not a correctness probability.",
            "Mean log scoring can favor incorrect segmentation lengths.",
            "Connected components may contain multiple digits or partial digits.",
            "Left-to-right ordering may be unreliable for overlapping components.",
            "No candidate is automatically accepted.",
        ],
    }
