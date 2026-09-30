import io

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from PIL import Image
from pydantic import BaseModel

from app.core.config import ALLOWED_FORMATS, MAX_UPLOAD_SIZE, META
from app.services.detector import detector

router = APIRouter(tags=["predict"])


class PredictionResponse(BaseModel):
    label: str
    confidence: float
    probabilities: dict
    prediction_time_ms: float


@router.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    data = await file.read()

    if not data:
        raise HTTPException(400, "Empty file")
    if len(data) > MAX_UPLOAD_SIZE:
        raise HTTPException(413, "File too large (max 10 MB)")

    try:
        img = Image.open(io.BytesIO(data))
        fmt = (img.format or "").upper()
    except Exception:
        raise HTTPException(400, "Invalid or corrupted image file")

    if fmt not in ALLOWED_FORMATS:
        raise HTTPException(400, f"Unsupported format '{fmt}'. Allowed: JPEG, PNG, WEBP, BMP")

    # run TF inference in a thread so we don't block the event loop
    result = await run_in_threadpool(detector.predict, img)
    return result


@router.get("/model-info")
def model_info():
    """Exposes meta.json — nice for the frontend 'About' section."""
    return META