import time

import numpy as np
import tensorflow as tf
from PIL import Image

from app.core.config import INPUT_SIZE, LABELS, MODEL_PATH


class Detector:
    def __init__(self):
        self.model = None
        self.labels = LABELS

    def load(self):
        """Load the model once at startup + warmup (first predict is slow otherwise)."""
        self.model = tf.keras.models.load_model(MODEL_PATH)
        dummy = np.zeros((1, *INPUT_SIZE, 3), dtype=np.float32)
        self.model.predict(dummy, verbose=0)

    def predict(self, image: Image.Image) -> dict:
        """Preprocess must EXACTLY match training: RGB, tf.image.resize to INPUT_SIZE, values 0-255."""
        start = time.perf_counter()

        # Resize EXACTLY like training: tf.image.resize (bilinear, no antialiasing),
        # then round to uint8 like the training pipeline. Pillow's resize smooths
        # (antialiases) large images, which the model reads as "AI-like" texture.
        img = np.asarray(image.convert("RGB"), dtype=np.uint8)
        t = tf.image.resize(img, INPUT_SIZE)                       # float32, values 0-255
        t = tf.cast(tf.clip_by_value(t, 0, 255), tf.uint8)
        arr = tf.cast(t, tf.float32)[tf.newaxis, ...].numpy()       # 0-255, NO rescaling

        p = float(self.model.predict(arr, verbose=0).ravel()[0])   # P(REAL)

        elapsed_ms = (time.perf_counter() - start) * 1000
        pred_class = 1 if p >= 0.5 else 0
        label = self.labels[str(pred_class)]

        return {
            "label": label,
            "confidence": p if pred_class == 1 else 1 - p,
            "probabilities": {
                self.labels["0"]: round(1 - p, 4),
                self.labels["1"]: round(p, 4),
            },
            "prediction_time_ms": round(elapsed_ms, 1),
        }


detector = Detector()   # singleton used by the API