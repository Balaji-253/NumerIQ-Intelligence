from __future__ import annotations

import logging
import warnings
from contextlib import asynccontextmanager
from io import BytesIO
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from PIL import Image, ImageOps, UnidentifiedImageError

from src.inference import Predictor
from src.sequence_recognizer import SequenceRecognizer


APP_NAME = "NumerIQ Intelligence API"
APP_VERSION = "2.1.0"

MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_IMAGE_WIDTH = 3000
MAX_IMAGE_HEIGHT = 3000

ALLOWED_CONTENT_TYPES = {
    "image/png",
    "image/jpeg",
    "image/webp",
    "image/bmp",
}

ALLOWED_FORMATS = {"PNG", "JPEG", "WEBP", "BMP"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger("numeriq.api")

predictor: Predictor | None = None
sequence_recognizer: SequenceRecognizer | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor, sequence_recognizer

    logger.info("Starting %s version %s", APP_NAME, APP_VERSION)

    try:
        predictor = Predictor()
        sequence_recognizer = SequenceRecognizer(predictor)
        logger.info("AI model and sequence recognizer loaded successfully")
    except FileNotFoundError:
        predictor = None
        sequence_recognizer = None
        logger.warning("Model checkpoint not found. Prediction unavailable.")

    yield

    predictor = None
    sequence_recognizer = None
    logger.info("Application shutdown completed")


app = FastAPI(
    title=APP_NAME,
    description=(
        "Handwritten digit recognition API with computer vision "
        "and deep learning."
    ),
    version=APP_VERSION,
    lifespan=lifespan,
)


@app.middleware("http")
async def request_observability(request: Request, call_next):
    request_id = str(uuid4())
    start_time = perf_counter()
    request.state.request_id = request_id

    try:
        response = await call_next(request)
    except Exception:
        logger.exception(
            "Unhandled request error | request_id=%s",
            request_id,
        )
        response = JSONResponse(
            status_code=500,
            content={"detail": "Internal server error."},
        )

    duration_ms = (perf_counter() - start_time) * 1000

    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-MS"] = f"{duration_ms:.2f}"

    logger.info(
        "request_id=%s method=%s path=%s status=%s duration_ms=%.2f",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )

    return response


async def read_limited_upload(file: UploadFile) -> bytes:
    """Read uploads in bounded chunks to enforce the size limit."""
    chunks = []
    total_size = 0

    while True:
        chunk = await file.read(64 * 1024)

        if not chunk:
            break

        total_size += len(chunk)

        if total_size > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail="Image must be 5 MB or smaller.",
            )

        chunks.append(chunk)

    if total_size == 0:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )

    return b"".join(chunks)


def decode_image(raw: bytes) -> Image.Image:
    """Validate and safely decode an uploaded image."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter(
                "error",
                Image.DecompressionBombWarning,
            )

            with Image.open(BytesIO(raw)) as verification_image:
                if verification_image.format not in ALLOWED_FORMATS:
                    raise HTTPException(
                        status_code=415,
                        detail="Unsupported image format.",
                    )

                width, height = verification_image.size

                if (
                    width <= 0
                    or height <= 0
                    or width > MAX_IMAGE_WIDTH
                    or height > MAX_IMAGE_HEIGHT
                ):
                    raise HTTPException(
                        status_code=400,
                        detail="Image dimensions must be at most 3000x3000.",
                    )

                if getattr(verification_image, "n_frames", 1) != 1:
                    raise HTTPException(
                        status_code=400,
                        detail="Animated or multi-frame images are not supported.",
                    )

                verification_image.verify()

            with Image.open(BytesIO(raw)) as image:
                image.load()
                image = ImageOps.exif_transpose(image)
                return image.convert("RGB").copy()

    except HTTPException:
        raise

    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as error:
        raise HTTPException(
            status_code=400,
            detail="Invalid, corrupted, or unsafe image.",
        ) from error


def validate_content_type(file: UploadFile) -> None:
    content_type = (
        file.content_type or ""
    ).split(";")[0].strip().lower()

    if content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail="Upload PNG, JPEG, WEBP, or BMP images.",
        )


async def prepare_upload(file: UploadFile) -> Image.Image:
    validate_content_type(file)
    raw = await read_limited_upload(file)
    return await run_in_threadpool(decode_image, raw)


@app.get("/")
async def root():
    return {
        "application": APP_NAME,
        "version": APP_VERSION,
        "documentation": "/docs",
        "health": "/health",
        "readiness": "/ready",
        "single_digit_endpoint": "/predict",
        "sequence_endpoint": "/predict/sequence",
    }


@app.get("/health")
async def health():
    """Liveness check."""
    return {
        "status": "ok",
        "application": APP_NAME,
        "version": APP_VERSION,
    }


@app.get("/ready")
async def readiness():
    """Readiness check."""
    if predictor is None or sequence_recognizer is None:
        raise HTTPException(
            status_code=503,
            detail="AI model is not loaded.",
        )

    return {
        "status": "ready",
        "model_loaded": True,
        "sequence_recognition_loaded": True,
        "model": predictor.model_metadata,
    }


@app.post("/predict")
async def predict(
    request: Request,
    file: UploadFile = File(...),
):
    """Recognize one handwritten digit."""

    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="AI model unavailable. Train or load a model first.",
        )

    try:
        image = await prepare_upload(file)

        result = await run_in_threadpool(
            predictor.predict,
            image,
        )

        logger.info(
            "Single-digit prediction | request_id=%s digit=%s confidence=%.4f",
            request.state.request_id,
            result["digit"],
            result["confidence"],
        )

        return result

    finally:
        await file.close()


@app.post("/predict/sequence")
async def predict_sequence(
    request: Request,
    file: UploadFile = File(...),
):
    """
    Recognize a sequence of separated handwritten digits.

    Connected or touching digits are not guaranteed to segment correctly.
    """

    if sequence_recognizer is None:
        raise HTTPException(
            status_code=503,
            detail="Sequence recognition model unavailable.",
        )

    try:
        image = await prepare_upload(file)

        result = await run_in_threadpool(
            sequence_recognizer.recognize,
            image,
        )

        logger.info(
            (
                "Sequence prediction | request_id=%s "
                "sequence=%s digit_count=%s status=%s"
            ),
            request.state.request_id,
            result["sequence"],
            result["digit_count"],
            result["status"],
        )

        return result

    finally:
        await file.close()


@app.exception_handler(HTTPException)
async def http_exception_handler(
    request: Request,
    exc: HTTPException,
):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": exc.detail,
            "request_id": getattr(
                request.state,
                "request_id",
                None,
            ),
        },
        headers=exc.headers,
    )
