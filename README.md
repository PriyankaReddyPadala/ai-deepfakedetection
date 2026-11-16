# AI-Based Deepfake Image Detection using Deep Learning

A web app that tells whether an image is **real** or **AI-generated**.
Upload a photo and the model returns **REAL** or **FAKE** with a confidence score in about 0.1 seconds.

The detector is an **EfficientNetB0** convolutional neural network, fine-tuned with transfer learning on
~17,000 real and AI-generated images (faces and general images). It reaches **90.4% accuracy on unseen test images**.

---

## Features

- Upload an image (JPG, PNG, WEBP or BMP, up to 10 MB) and get a REAL / FAKE prediction
- Confidence score and probability for each class
- Works on **faces** and **general images** (animals, objects, scenes)
- REST API (FastAPI) + web interface (React)

---

## Results (final model, v6)

Measured on **1,700 test images** the model never saw during training:

| Metric | Value |
|---|---|
| **Test accuracy** | **90.4%** |
| AUC | 0.968 |
| Faces (978 images) | 92.3% |
| General images (722 images) | 87.8% |
| AI images correctly detected | 92.6% (785 / 848) |
| Real images correctly detected | 88.3% (752 / 852) |

### Model development

| Version | Dataset | Test accuracy | Notes |
|---|---|---|---|
| v1 (`model_training.ipynb`) | CIFAKE (32×32) | 71.2% | EfficientNetB0 on 32×32 input, mostly frozen backbone |
| v2 | CIFAKE (32×32) | 88.7% | Upscaling to 224 inside the model, full fine-tuning |
| v3 | CIFAKE (32×32) | 92.5% | Faster training at 128 px, duplicate removal |
| v4 | CIFAKE (32×32) | 93.7% | Shorter, fully decayed learning-rate schedule |
| **v6 (final)** | **Faces + general images (224×224)** | **90.4%** | Clear, realistic images, 8+ AI generators |
| v7 | Faces + general images | 87.9% | Added blur/noise augmentation, lower accuracy, so v6 kept |

**Why the final model has a lower number than v4:** v4's 93.7% was measured on blurry 32×32 images.
When tested on real-world photos, it failed (for example, it called a modern AI-generated portrait 100% real).
v6 is trained and tested on clear, varied images from many AI generators, so its 90.4% reflects real-world use much better.

---

## Datasets

The datasets are **not included in this repository** (about 3 GB). Download them and place them as shown below.

| Dataset | Content | Source |
|---|---|---|
| **Real vs Fake Faces (RVF10K)** | 10,000 faces: real (NVIDIA FFHQ, from Flickr) vs fake (StyleGAN) | [Zenodo](https://zenodo.org/records/14532968) |
| **Tiny-GenImage** (validation split) | ~7,000 general images: real (ImageNet) vs fake (Midjourney, Stable Diffusion 1.4 / 1.5, ADM, GLIDE, VQDM, Wukong, BigGAN) | [Hugging Face](https://huggingface.co/datasets/TheKernel01/Tiny-GenImage) |

### Dataset setup

1. Download `Real vs Fake Faces.zip` from Zenodo and unzip it.
2. From Hugging Face, download the 4 files `data/validation-0000X-of-00004.parquet`.
3. Arrange them like this:

```
backend/train/dataset/
├── faces/
│   └── Real vs Fake Faces/      ← unzipped face dataset
└── general/
    ├── validation-00000-of-00004.parquet
    ├── validation-00001-of-00004.parquet
    ├── validation-00002-of-00004.parquet
    └── validation-00003-of-00004.parquet
```

The training notebook labels face images by their `real` / `fake` folder, reads labels from the parquet files,
removes duplicates, and makes an 80 / 10 / 10 train / validation / test split.

---

## Model

| Item | Details |
|---|---|
| Architecture | EfficientNetB0 (pretrained on ImageNet) + global average pooling + dropout (0.2) + sigmoid output |
| Input | 224 × 224 RGB, pixel values 0–255 (normalisation is built into the model) |
| Output | Probability that the image is real (label 0 = FAKE, 1 = REAL) |
| Training | Phase 1: 1 epoch, backbone frozen (Adam, lr 1e-3). Phase 2: 3 epochs, full fine-tuning (AdamW, cosine decay from 2e-4, weight decay 1e-4) |
| Regularisation | Dropout, early stopping, weight decay, BatchNorm frozen during fine-tuning |
| Augmentation | Horizontal flip, brightness ±10%, contrast ±10%, random JPEG compression (quality 60–100) |
| Framework | TensorFlow / Keras 2.16 |

---

## Tech stack

- **Model / training:** Python, TensorFlow / Keras, scikit-learn, NumPy, PyArrow
- **Backend:** FastAPI, Uvicorn, Pillow
- **Frontend:** React, Vite

---

## Project structure

```
ai-deepfakedetection/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI app (loads the model at startup)
│   │   ├── api/predict.py       # /api/predict and /api/model-info endpoints
│   │   ├── services/detector.py # preprocessing + prediction
│   │   └── core/config.py       # paths, upload limits, CORS
│   ├── saved_models/
│   │   ├── model.keras          # final model used by the app (v6)
│   │   ├── meta.json            # input size, labels, test accuracy
│   │   ├── model_cifake.keras   # earlier CIFAKE model (v4), kept for reference
│   │   └── meta_cifake.json
│   ├── train/
│   │   ├── model_training.ipynb               # v1
│   │   ├── model_training_v2.ipynb            # v2
│   │   ├── model_training_v3.ipynb            # v3
│   │   ├── model_training_v4.ipynb            # v4
│   │   ├── model_training_v6_faces_general.ipynb  # v6 (final)
│   │   ├── model_training_v7.ipynb            # v7 experiment
│   │   └── dataset/                           # not in git (see Dataset setup)
│   └── requirements.txt
└── frontend/
    └── src/                     # React app (upload zone, result card, API calls)
```

---

## How to run

### 1. Backend

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install tensorflow==2.16.2     # on Apple Silicon Macs also: pip install tensorflow-metal
uvicorn app.main:app --reload --port 8000
```

Wait for `Application startup complete`. The API runs at http://localhost:8000 (interactive docs at http://localhost:8000/docs).

### 2. Frontend (in a second terminal)

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173 and upload an image.

### 3. Training (optional)

1. Set up the datasets as described above.
2. Install the training extras in the same environment:
   ```bash
   pip install scikit-learn "pyarrow==17.0.0" jupyter
   ```
   (Keep NumPy below 2.0 for TensorFlow 2.16; newer PyArrow versions require NumPy 2.)
3. Open `backend/train/model_training_v6_faces_general.ipynb` and run all cells (about 20–30 minutes on an Apple M-series MacBook).
4. The notebook backs up the previous model and saves the new `model.keras` and `meta.json` to `backend/saved_models/`. Restart the backend to use it.

---

## API

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/predict` | Upload an image (`file` form field). Returns label, confidence, probabilities and prediction time. |
| `GET` | `/api/model-info` | Returns the model metadata from `meta.json` |
| `GET` | `/health` | Health check |

Example response from `/api/predict`:

```json
{
  "label": "FAKE",
  "confidence": 0.98,
  "probabilities": { "FAKE": 0.98, "REAL": 0.02 },
  "prediction_time_ms": 78.0
}
```

---

## Limitations

- **Newest AI generators:** very photo-realistic images from recent tools (e.g. modern portrait generators) can still be classified as real, because these generators are not in the training data.
- **Heavily processed phone photos:** real photos taken in low light or with beauty mode / strong noise reduction can be classified as fake, because the smoothing resembles AI-generated texture.
- **Low-resolution or blurry images:** very small images (e.g. 32×32) tend to be classified as fake.
- **Non-photographic images** (drawings, cartoons, screenshots of text) are outside the training data.
- Accuracy depends on the data: 90.4% is measured on the test split of the datasets above; results on other kinds of images may differ.

---

## Future work

- Train on larger and newer datasets (full GenImage, faces from modern generators such as Flux, SDXL and Midjourney v6) on a GPU
- Add real-world phone photos to the training data
- Try larger backbones (EfficientNetB2–B4, ConvNeXt, ViT) and higher input resolution
- Show which image regions influenced the prediction (Grad-CAM)

---

## Acknowledgements

- Real vs Fake Faces (RVF10K): real faces from NVIDIA FFHQ, fake faces from the "1 Million Fake Faces" (StyleGAN) collection
- Tiny-GenImage, a scaled-down version of the GenImage benchmark (Zhu et al., NeurIPS 2023)
- CIFAKE dataset (Bird & Lotfi), used in the early model versions
- EfficientNet (Tan & Le, 2019), via Keras Applications
