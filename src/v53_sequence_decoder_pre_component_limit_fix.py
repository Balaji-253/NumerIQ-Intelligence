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
    work_counter: dict[str, int] | None = None,
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

        exhausted = False

        for beam in beams:
            for candidate in candidates:
                if work_counter is not None:
                    if work_counter["remaining"] <= 0:
                        exhausted = True
                        break

                    work_counter["remaining"] -= 1
                    work_counter["generated"] += 1

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

        if exhausted:
            return []

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
            "engine": "numeriq-v5.3",
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

    MAX_TOTAL_EXPANSIONS = 30000
    HYPOTHESIS_BUDGET = 7500
    GLOBAL_BUDGET = 16500
    ORIGINAL_BUDGET = 6000

    total_work = {
        "remaining": MAX_TOTAL_EXPANSIONS,
        "generated": 0,
    }

    total_hypotheses = sum(
        len(component.get("hypotheses", []))
        for component in ordered
    )
    per_hypothesis_budget = max(
        1,
        HYPOTHESIS_BUDGET // max(total_hypotheses, 1),
    )
    hypothesis_stage_generated = 0

    component_options = []
    original_component_options = []

    for component in ordered:
        hypotheses = component.get("hypotheses", [])
        options = []

        original_options = []

        for hypothesis in hypotheses:
            available_hypothesis_budget = min(
                per_hypothesis_budget,
                HYPOTHESIS_BUDGET - hypothesis_stage_generated,
                max(0, total_work["remaining"] - ORIGINAL_BUDGET),
            )

            if available_hypothesis_budget <= 0:
                expanded = []
            else:
                local_counter = {
                    "remaining": available_hypothesis_budget,
                    "generated": 0,
                }

                expanded = _expand_hypothesis(
                    hypothesis,
                    beam_width,
                    local_counter,
                )

                consumed = local_counter["generated"]
                total_work["generated"] += consumed
                total_work["remaining"] -= consumed
                hypothesis_stage_generated += consumed

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
                "engine": "numeriq-v5.3",
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

    max_active_paths = 32
    max_beams_per_path = 64
    max_expansions_per_component = 8192
    global_stage_generated = 0
    global_search_complete = True
    search_stats = []

    for component_index, options in enumerate(component_options):
        path_groups = {}
        expansions_this_component = 0
        budget_exhausted = False

        for beam in global_beams:
            for option in options:
                if (
                    expansions_this_component >= max_expansions_per_component
                    or global_stage_generated >= GLOBAL_BUDGET
                    or total_work["remaining"] <= ORIGINAL_BUDGET
                ):
                    budget_exhausted = True
                    break

                expansions_this_component += 1
                global_stage_generated += 1
                total_work["generated"] += 1
                total_work["remaining"] -= 1

                probabilities = (
                    beam["digit_probabilities"]
                    + option["digit_probabilities"]
                )

                component_path = {
                    "component_id": option["component_id"],
                    "hypothesis_id": option["hypothesis_id"],
                    "strategy": option["strategy"],
                    "sequence": option["sequence"],
                    "part_count": option["part_count"],
                }

                component_paths = (
                    beam["component_paths"] + [component_path]
                )

                candidate = {
                    "sequence": beam["sequence"] + option["sequence"],
                    "log_probability_sum": (
                        beam["log_probability_sum"]
                        + option["log_probability_sum"]
                    ),
                    "digit_probabilities": probabilities,
                    "component_paths": component_paths,
                    "geometry_scores": (
                        beam["geometry_scores"] + [option["geometry_score"]]
                    ),
                }

                signature = tuple(
                    entry["hypothesis_id"] for entry in component_paths
                )

                score = (
                    candidate["log_probability_sum"]
                    / max(len(probabilities), 1)
                )

                group = path_groups.get(signature)

                if group is None:
                    if len(path_groups) >= max_active_paths:
                        weakest_signature = min(
                            path_groups,
                            key=lambda key: (
                                path_groups[key][0]["log_probability_sum"]
                                / max(
                                    len(
                                        path_groups[key][0]["digit_probabilities"]
                                    ),
                                    1,
                                )
                            ),
                        )

                        weakest_score = (
                            path_groups[weakest_signature][0][
                                "log_probability_sum"
                            ]
                            / max(
                                len(
                                    path_groups[weakest_signature][0][
                                        "digit_probabilities"
                                    ]
                                ),
                                1,
                            )
                        )

                        if score <= weakest_score:
                            continue

                        del path_groups[weakest_signature]

                    path_groups[signature] = [candidate]
                    continue

                group.append(candidate)
                group.sort(
                    key=lambda item: (
                        item["log_probability_sum"]
                        / max(len(item["digit_probabilities"]), 1)
                    ),
                    reverse=True,
                )
                del group[max_beams_per_path:]

            if budget_exhausted:
                break

        search_stats.append({
            "component_index": component_index,
            "available_options": len(options),
            "input_beams": len(global_beams),
            "generated_expansions": expansions_this_component,
            "budget": max_expansions_per_component,
            "budget_exhausted": budget_exhausted,
            "retained_paths": len(path_groups),
            "retained_beams": sum(len(group) for group in path_groups.values()),
        })

        global_beams = [
            item
            for group in path_groups.values()
            for item in group
        ]

        if budget_exhausted:
            global_search_complete = False
            break

    # Independently decode the all-original segmentation path.
    # This branch is protected from competition with split hypotheses.
    original_path_beams = [{
        "sequence": "",
        "log_probability_sum": 0.0,
        "digit_probabilities": [],
        "component_paths": [],
        "geometry_scores": [],
    }]

    original_path_complete = (
        len(original_component_options) == len(ordered)
    )
    original_stage_generated = 0

    if original_path_complete:
        for options in original_component_options:
            expanded = []
            original_budget_exhausted = False

            for beam in original_path_beams:
                for option in options:
                    if (
                        original_stage_generated >= ORIGINAL_BUDGET
                        or total_work["remaining"] <= 0
                    ):
                        original_budget_exhausted = True
                        break

                    original_stage_generated += 1
                    total_work["generated"] += 1
                    total_work["remaining"] -= 1

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
            if original_budget_exhausted:
                original_path_complete = False
                break

            original_path_beams = expanded[:beam_width]

    # Retain distinct strings; different segmentation paths remain traceable.
    unique = {}
    for beam in (global_beams if global_search_complete else []):
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

    for beam in (
        original_path_beams[:min(5, max_results)]
        if original_path_complete
        else []
    ):
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

    # V5.3: select candidates with segmentation-path diversity.
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

        # Prefer a protected original-path candidate within its own path.
        protected = [
            item for item in group
            if item.get("preserved_original_path") is True
        ]

        path_representatives.append(
            protected[0] if protected else group[0]
        )

    path_representatives.sort(
        key=lambda item: item["mean_log_probability"],
        reverse=True,
    )

    # Preserve path diversity without allowing original-path candidates
    # to consume multiple result slots unnecessarily.
    selected = path_representatives[:max_results]
    selected_sequences = {item["sequence"] for item in selected}

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
        "engine": "numeriq-v5.3",
        "status": "sequence_hypotheses_generated",
        "sequence": None,
        "digit_count": None,
        "candidates": ranked,
        "candidate_count": len(ranked),
        "component_count": len(ordered),
        "ordering_ambiguity": ordering_ambiguity,
        "search_diagnostics": {
            "max_active_paths": max_active_paths,
            "max_beams_per_path": max_beams_per_path,
            "max_expansions_per_component": max_expansions_per_component,
            "max_total_expansions": MAX_TOTAL_EXPANSIONS,
            "hypothesis_budget": HYPOTHESIS_BUDGET,
            "global_budget": GLOBAL_BUDGET,
            "original_path_budget": ORIGINAL_BUDGET,
            "total_generated_expansions": total_work["generated"],
            "remaining_expansions": total_work["remaining"],
            "hypothesis_expansions": hypothesis_stage_generated,
            "global_expansions": global_stage_generated,
            "original_path_expansions": original_stage_generated,
            "budget_exhaustions": (
                sum(item["budget_exhausted"] for item in search_stats)
                + int(not global_search_complete)
                + int(not original_path_complete)
            ),
            "global_search_complete": global_search_complete,
            "original_path_complete": original_path_complete,
            "per_component": search_stats,
        },
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
