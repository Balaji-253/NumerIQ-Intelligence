from PIL import Image, ImageDraw

from src.v3_sequence_recognizer import V3SequenceRecognizer


class MockPredictor:
    model_metadata = {
        "model_name": "MockDigitCNN",
        "model_version": "test",
        "device": "cpu",
    }

    def predict_crops(self, crops):
        results = []

        for index, _ in enumerate(crops):
            digit = index % 10
            results.append({
                "digit": digit,
                "confidence": 0.95,
                "is_uncertain": False,
                "top3": [
                    {"digit": digit, "probability": 0.95},
                    {"digit": (digit + 1) % 10, "probability": 0.03},
                    {"digit": (digit + 2) % 10, "probability": 0.02},
                ],
            })

        return results


def test_blank_image_returns_no_digits():
    recognizer = V3SequenceRecognizer(MockPredictor())
    image = Image.new("RGB", (200, 100), "white")

    result = recognizer.recognize(image)

    assert result["status"] == "no_digits_detected"
    assert result["sequence"] == ""
    assert result["digit_count"] == 0


def test_separated_components_are_preserved():
    recognizer = V3SequenceRecognizer(MockPredictor())

    image = Image.new("RGB", (180, 100), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 20, 40, 75), fill="black")
    draw.rectangle((100, 20, 120, 75), fill="black")

    result = recognizer.recognize(image)

    assert result["digit_count"] == 2
    assert len(result["digits"]) == 2
    assert result["component_count"] == 2
    assert result["sequence"] == result["provisional_sequence"]


def test_candidate_split_is_not_automatically_accepted():
    recognizer = V3SequenceRecognizer(MockPredictor())

    image = Image.new("RGB", (120, 100), "white")
    draw = ImageDraw.Draw(image)

    # Two wide regions connected by a narrow horizontal bridge.
    draw.rectangle((15, 25, 45, 75), fill="black")
    draw.rectangle((65, 25, 95, 75), fill="black")
    draw.rectangle((40, 25, 70, 27), fill="black")

    result = recognizer.recognize(image)

    assert result["component_count"] == 1
    assert result["digit_count"] == 1

    digit = result["digits"][0]

    assert digit["possible_multi_digit_component"] is True
    assert digit["is_uncertain"] is True
    assert digit["candidate_splits"]

    assert all(
        candidate["accepted"] is False
        for candidate in digit["candidate_splits"]
    )

    assert result["status"] == "completed_with_uncertainty"
    assert result["digit_count_is_reliable"] is False
    assert result["sequence"] == ""
    assert result["provisional_sequence"] == "0"


def test_original_component_prediction_is_retained():
    recognizer = V3SequenceRecognizer(MockPredictor())

    image = Image.new("RGB", (100, 100), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((30, 20, 55, 75), fill="black")

    result = recognizer.recognize(image)

    assert result["digit_count"] == 1
    assert result["digits"][0]["digit"] == "0"
    assert result["digits"][0]["confidence"] == 0.95
    assert result["sequence"] == "0"
    assert result["provisional_sequence"] == "0"


class LowConfidencePredictor(MockPredictor):
    def predict_crops(self, crops):
        results = super().predict_crops(crops)
        for result in results:
            result["is_uncertain"] = True
        return results


def test_low_confidence_prediction_is_not_definitive():
    recognizer = V3SequenceRecognizer(LowConfidencePredictor())

    image = Image.new("RGB", (100, 100), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((30, 20, 55, 75), fill="black")

    result = recognizer.recognize(image)

    assert result["status"] == "completed_with_uncertainty"
    assert result["sequence"] == ""
    assert result["provisional_sequence"] == "0"
    assert result["digits"][0]["is_uncertain"] is True


def test_component_count_is_not_claimed_as_verified_digit_count():
    recognizer = V3SequenceRecognizer(MockPredictor())

    image = Image.new("RGB", (180, 100), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 20, 40, 75), fill="black")
    draw.rectangle((100, 20, 120, 75), fill="black")

    result = recognizer.recognize(image)

    assert result["component_count"] == 2
    assert result["digit_count_is_reliable"] is False
