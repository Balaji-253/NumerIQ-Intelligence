from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from src.api import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def create_digit_image():
    image = Image.new("RGB", (280, 280), "black")
    draw = ImageDraw.Draw(image)
    draw.text((100, 80), "8", fill="white")

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_root_endpoint(client):
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["application"] == "NumerIQ Intelligence API"


def test_health_endpoint(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_model_readiness(client):
    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json()["status"] == "ready"
    assert response.json()["model_loaded"] is True


def test_valid_digit_prediction(client):
    image_data = create_digit_image()

    response = client.post(
        "/predict",
        files={
            "file": (
                "digit.png",
                image_data,
                "image/png",
            )
        },
    )

    assert response.status_code == 200

    result = response.json()

    assert isinstance(result["digit"], int)
    assert 0 <= result["digit"] <= 9

    assert 0.0 <= result["confidence"] <= 1.0
    assert isinstance(result["is_uncertain"], bool)

    assert len(result["top3"]) == 3
    assert len(result["top5"]) == 5

    assert result["latency_ms"] >= 0


def test_reject_unsupported_content_type(client):
    response = client.post(
        "/predict",
        files={
            "file": (
                "document.pdf",
                b"not an image",
                "application/pdf",
            )
        },
    )

    assert response.status_code == 415


def test_reject_invalid_image(client):
    response = client.post(
        "/predict",
        files={
            "file": (
                "invalid.png",
                b"invalid image bytes",
                "image/png",
            )
        },
    )

    assert response.status_code == 400


def test_reject_empty_upload(client):
    response = client.post(
        "/predict",
        files={
            "file": (
                "empty.png",
                b"",
                "image/png",
            )
        },
    )

    assert response.status_code == 400


def test_reject_oversized_upload(client):
    oversized_data = b"x" * (5 * 1024 * 1024 + 1)

    response = client.post(
        "/predict",
        files={
            "file": (
                "large.png",
                oversized_data,
                "image/png",
            )
        },
    )

    assert response.status_code == 413


def create_sequence_image():
    image = Image.new("RGB", (240, 100), "white")
    draw = ImageDraw.Draw(image)

    draw.rectangle((20, 20, 45, 75), fill="black")
    draw.rectangle((130, 20, 155, 75), fill="black")

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def create_blank_image():
    image = Image.new("RGB", (200, 100), "white")

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_sequence_prediction_endpoint(client):
    response = client.post(
        "/predict/sequence",
        files={
            "file": (
                "sequence.png",
                create_sequence_image(),
                "image/png",
            )
        },
    )

    assert response.status_code == 200

    result = response.json()

    assert result["status"] in {
        "completed",
        "completed_with_uncertainty",
    }
    assert isinstance(result["sequence"], str)
    assert result["digit_count"] == 2
    assert len(result["digits"]) == 2

    for digit in result["digits"]:
        assert digit["digit"] in "0123456789"
        assert 0.0 <= digit["confidence"] <= 1.0
        assert len(digit["bbox"]) == 4


def test_sequence_prediction_blank_image(client):
    response = client.post(
        "/predict/sequence",
        files={
            "file": (
                "blank.png",
                create_blank_image(),
                "image/png",
            )
        },
    )

    assert response.status_code == 200

    result = response.json()

    assert result["status"] == "no_digits_detected"
    assert result["sequence"] == ""
    assert result["digit_count"] == 0


def test_sequence_prediction_rejects_pdf(client):
    response = client.post(
        "/predict/sequence",
        files={
            "file": (
                "document.pdf",
                b"not an image",
                "application/pdf",
            )
        },
    )

    assert response.status_code == 415
