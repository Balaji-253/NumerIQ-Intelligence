import csv

import pytest

from src.sequence_evaluation import (
    levenshtein_distance,
    load_manifest,
    percentile,
)


def test_identical_sequences_have_zero_edit_distance():
    assert levenshtein_distance("2026", "2026") == 0


def test_substitution_distance():
    assert levenshtein_distance("1234", "1284") == 1


def test_insertion_distance():
    assert levenshtein_distance("123", "1234") == 1


def test_deletion_distance():
    assert levenshtein_distance("1234", "124") == 1


def test_empty_sequence_distance():
    assert levenshtein_distance("", "123") == 3
    assert levenshtein_distance("123", "") == 3


def test_percentile_values():
    values = [10.0, 20.0, 30.0, 40.0, 50.0]

    assert percentile(values, 50) == 30.0
    assert percentile(values, 95) == 50.0


def test_percentile_empty_values():
    assert percentile([], 95) is None


def test_manifest_loading(tmp_path):
    image = tmp_path / "sample.png"
    image.touch()

    manifest = tmp_path / "manifest.csv"

    with manifest.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["image_path", "ground_truth"])
        writer.writerow(["sample.png", "2026"])

    rows = load_manifest(manifest)

    assert len(rows) == 1
    assert rows[0]["image_path"] == "sample.png"
    assert rows[0]["ground_truth"] == "2026"


def test_manifest_rejects_non_digit_ground_truth(tmp_path):
    manifest = tmp_path / "manifest.csv"

    with manifest.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["image_path", "ground_truth"])
        writer.writerow(["sample.png", "20A6"])

    with pytest.raises(ValueError):
        load_manifest(manifest)


def test_processing_failure_counts_as_incorrect_prediction():
    from src.sequence_evaluation import summarize_predictions

    records = [
        {
            "ground_truth": "123",
            "prediction": "123",
            "status": "completed",
            "digit_count": 3,
            "low_confidence": False,
            "latency_ms": 10.0,
        },
        {
            "ground_truth": "456",
            "prediction": "",
            "status": "processing_error",
            "digit_count": 0,
            "low_confidence": False,
            "latency_ms": 20.0,
        },
    ]

    metrics = summarize_predictions(records)

    assert metrics["total_samples"] == 2
    assert metrics["processing_failures"] == 1
    assert metrics["processing_failure_rate"] == 0.5
    assert metrics["exact_sequence_accuracy"] == 0.5
    assert metrics["character_error_rate"] == 0.5
    assert metrics["latency_ms"]["mean"] == 15.0


def test_all_processing_failures_produce_zero_accuracy():
    from src.sequence_evaluation import summarize_predictions

    records = [
        {
            "ground_truth": "2026",
            "prediction": "",
            "status": "processing_error",
            "digit_count": 0,
            "low_confidence": False,
            "latency_ms": 12.0,
        }
    ]

    metrics = summarize_predictions(records)

    assert metrics["exact_sequence_accuracy"] == 0.0
    assert metrics["character_error_rate"] == 1.0
    assert metrics["processing_failure_rate"] == 1.0
    assert metrics["no_detection_rate"] is None


def test_percentile_rejects_invalid_range():
    with pytest.raises(ValueError):
        percentile([1.0, 2.0], 120)


def test_digit_count_metrics_include_processing_failures():
    from src.sequence_evaluation import summarize_predictions

    records = [
        {
            "ground_truth": "123",
            "prediction": "123",
            "status": "completed",
            "digit_count": 3,
            "low_confidence": False,
            "latency_ms": 10.0,
        },
        {
            "ground_truth": "456",
            "prediction": "",
            "status": "processing_error",
            "digit_count": 0,
            "low_confidence": False,
            "latency_ms": 20.0,
        },
    ]

    metrics = summarize_predictions(records)

    assert metrics["digit_count_mismatch_rate_all"] == 0.5
    assert metrics["digit_count_mismatch_rate_successful"] == 0.0
    assert metrics["digit_count_accuracy_all"] == 0.5


def test_digit_count_metrics_when_all_samples_fail():
    from src.sequence_evaluation import summarize_predictions

    records = [
        {
            "ground_truth": "1234",
            "prediction": "",
            "status": "processing_error",
            "digit_count": 0,
            "low_confidence": False,
            "latency_ms": 12.0,
        }
    ]

    metrics = summarize_predictions(records)

    assert metrics["digit_count_mismatch_rate_all"] == 1.0
    assert metrics["digit_count_mismatch_rate_successful"] is None
    assert metrics["digit_count_accuracy_all"] == 0.0
