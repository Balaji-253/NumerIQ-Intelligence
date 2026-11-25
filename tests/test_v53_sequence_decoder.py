from src.v53_sequence_decoder import decode_v5


def hypothesis(hypothesis_id, parts, original=False):
    return {
        "hypothesis_id": hypothesis_id,
        "strategy": "original" if original else "single_cut",
        "is_original": original,
        "geometry_score": 1.0,
        "part_count": len(parts),
        "part_predictions": [
            {
                "top5": [
                    {"digit": digit, "probability": probability}
                    for digit, probability in candidates
                ]
            }
            for candidates in parts
        ],
    }


def component(component_id, x, hypotheses):
    return {
        "component_id": component_id,
        "original_bbox": (x, 0, 100, 100),
        "image_bbox": (x, 0, 100, 100),
        "hypotheses": hypotheses,
    }


def test_v52_preserves_0218_split_path():
    components = [
        component(1, 0, [
            hypothesis(
                "c1_original",
                [[(9, .6597), (8, .1922), (3, .1352),
                  (5, .0119), (0, .0006)]],
                True,
            ),
            hypothesis(
                "c1_30",
                [
                    [(6, .8115), (1, .11), (0, .0751),
                     (8, .0021), (2, .0011)],
                    [(3, .9518), (9, .0234), (8, .0118),
                     (2, .0061), (7, .003)],
                ],
            ),
        ]),
        component(2, 110, [
            hypothesis(
                "c2_original",
                [[(1, .9999), (7, .0001)]],
                True,
            ),
        ]),
        component(3, 220, [
            hypothesis(
                "c3_original",
                [[(8, .9993), (3, .0007)]],
                True,
            ),
        ]),
    ]

    result = decode_v5(components, beam_width=64, max_results=20)

    candidates = {
        item["sequence"]: item
        for item in result["candidates"]
    }

    assert "0218" in candidates

    path_ids = [
        item["hypothesis_id"]
        for item in candidates["0218"]["component_paths"]
    ]

    assert path_ids == [
        "c1_30",
        "c2_original",
        "c3_original",
    ]


def test_v52_does_not_automatically_accept():
    result = decode_v5([], beam_width=64, max_results=20)

    assert result["sequence"] is None
    assert result["automatic_acceptance"] is False
    assert result["selection_performed"] is False
