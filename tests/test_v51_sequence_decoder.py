from src.v51_sequence_decoder import decode_v5


def make_hypothesis(hypothesis_id, parts, is_original):
    return {
        "hypothesis_id": hypothesis_id,
        "strategy": "original" if is_original else "vertical_projection",
        "is_original": is_original,
        "geometry_score": 1.0,
        "part_count": len(parts),
        "part_predictions": [
            {"top5": [
                {"digit": digit, "probability": probability}
                for digit, probability in candidates
            ]}
            for candidates in parts
        ],
    }


def make_component(component_id, x, hypotheses):
    return {
        "component_id": component_id,
        "original_bbox": (x, 0, 80, 100),
        "image_bbox": (x, 0, 80, 100),
        "hypotheses": hypotheses,
    }


def test_original_segmentation_path_preserves_3789():
    components = [
        make_component(4, 0, [
            make_hypothesis("c4_original", [[(3, .9999), (8, .00005)]], True),
        ]),
        make_component(3, 100, [
            make_hypothesis("c3_original", [[
                (9, .92356), (4, .05003), (7, .01212),
                (2, .01090), (8, .00133),
            ]], True),
            make_hypothesis("c3_split", [
                [(1, .9999), (7, .0001)],
                [(1, .9999), (7, .0001)],
            ], False),
        ]),
        make_component(2, 200, [
            make_hypothesis("c2_original", [[
                (8, .9454), (3, .03), (9, .02),
            ]], True),
        ]),
        make_component(1, 300, [
            make_hypothesis("c1_original", [[
                (9, .9986), (4, .001),
            ]], True),
        ]),
    ]

    result = decode_v5(components, beam_width=64, max_results=20)

    candidates = {
        item["sequence"]: item
        for item in result["candidates"]
    }

    assert "3789" in candidates
    assert candidates["3789"]["preserved_original_path"] is True
    assert [
        path["hypothesis_id"]
        for path in candidates["3789"]["component_paths"]
    ] == [
        "c4_original",
        "c3_original",
        "c2_original",
        "c1_original",
    ]


def test_no_automatic_acceptance():
    result = decode_v5([], beam_width=64, max_results=20)

    assert result["sequence"] is None
    assert result["automatic_acceptance"] is False
    assert result["selection_performed"] is False
