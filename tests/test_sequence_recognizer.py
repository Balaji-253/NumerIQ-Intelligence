from PIL import Image, ImageDraw

from src.sequence_recognizer import SequenceRecognizer


class MockPredictor:
    def __init__(self, digits):
        self.digits = iter(digits)
        self.calls = 0

    def predict(self, image):
        self.calls += 1
        digit, uncertain = next(self.digits)

        return {
            "digit": digit,
            "confidence": 0.95 if not uncertain else 0.45,
            "is_uncertain": uncertain,
            "top3": [
                {"digit": digit, "confidence": 0.95}
            ],
        }


def create_two_component_image():
    image = Image.new("RGB", (200, 100), "white")
    draw = ImageDraw.Draw(image)

    draw.rectangle((20, 20, 45, 75), fill="black")
    draw.rectangle((110, 20, 135, 75), fill="black")

    return image


def test_recognizes_multiple_separated_components():
    predictor = MockPredictor([
        (2, False),
        (6, False),
    ])

    recognizer = SequenceRecognizer(predictor)
    result = recognizer.recognize(create_two_component_image())

    assert result["status"] == "completed"
    assert result["sequence"] == "26"
    assert result["digit_count"] == 2
    assert len(result["digits"]) == 2
    assert predictor.calls == 2


def test_preserves_leading_zero():
    predictor = MockPredictor([
        (0, False),
        (7, False),
    ])

    recognizer = SequenceRecognizer(predictor)
    result = recognizer.recognize(create_two_component_image())

    assert result["sequence"] == "07"


def test_blank_image_returns_no_digits():
    image = Image.new("RGB", (200, 100), "white")

    recognizer = SequenceRecognizer(MockPredictor([]))
    result = recognizer.recognize(image)

    assert result["status"] == "no_digits_detected"
    assert result["sequence"] == ""
    assert result["digit_count"] == 0


def test_uncertain_prediction_is_reported():
    predictor = MockPredictor([
        (3, False),
        (8, True),
    ])

    recognizer = SequenceRecognizer(predictor)
    result = recognizer.recognize(create_two_component_image())

    assert result["status"] == "completed_with_uncertainty"
    assert result["sequence"] == "38"
    assert len(result["warnings"]) >= 3
