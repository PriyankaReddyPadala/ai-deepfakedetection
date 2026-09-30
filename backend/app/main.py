from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.predict import router as predict_router
from app.core.config import CORS_ORIGINS
from app.services.detector import detector


@asynccontextmanager
async def lifespan(app: FastAPI):
    detector.load()      # load model at startup, not per-request
    yield


app = FastAPI(
    title="AI Image Detector API",
    description="Detects whether an uploaded image is AI-generated or real",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,     # React dev server
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict_router, prefix="/api")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": detector.model is not None}