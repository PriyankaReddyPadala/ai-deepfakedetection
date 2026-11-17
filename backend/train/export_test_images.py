"""
Export the v6 TEST images (the 10% the model never trained on) as normal image files,
so they can be used as fair demo images.

It rebuilds the exact same split as model_training_v6_faces_general.ipynb
(same folders, same order, same duplicate removal, same seed), then saves up to
PER_GROUP images per class for faces and general into:

    ~/Desktop/demo_test_images/real/   and   ~/Desktop/demo_test_images/fake/

Run from the backend folder with the project's environment:
    source .venv/bin/activate
    python train/export_test_images.py

Note: the split only matches v6 if dataset/faces and dataset/general contain
exactly the same files as when v6 was trained (faces + the 4 validation parquet files).
"""
import hashlib, io, shutil
from pathlib import Path

import numpy as np
from PIL import Image

PER_GROUP = 25          # per class per group -> 25 real + 25 fake faces, 25 real + 25 fake general
SEED = 42
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

ROOT = Path(__file__).resolve()
while not (ROOT / "backend" / "train").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
FACES_DIR   = ROOT / "backend" / "train" / "dataset" / "faces"
GENERAL_DIR = ROOT / "backend" / "train" / "dataset" / "general"
OUT_DIR     = Path.home() / "Desktop" / "demo_test_images"


def label_of(p):
    # identical rule to v6: nearest folder mentioning only one of real/fake decides
    for part in reversed(p.relative_to(FACES_DIR).parts[:-1]):
        x = part.lower()
        has_f, has_r = "fake" in x, "real" in x
        if has_f != has_r:
            return 0 if has_f else 1          # 0 = FAKE, 1 = REAL
    return None


# ---- rebuild the item list exactly like v6 ----
items, seen = [], set()
if FACES_DIR.exists():
    for p in sorted(FACES_DIR.rglob("*")):
        if p.suffix.lower() not in IMG_EXT or p.name.startswith("."):
            continue
        lab = label_of(p)
        if lab is None:
            continue
        h = hashlib.md5(p.read_bytes()).hexdigest()
        if h not in seen:
            seen.add(h); items.append(("faces", "file", str(p), lab))

parquets = sorted(GENERAL_DIR.rglob("*.parquet")) if GENERAL_DIR.exists() else []
if parquets:
    import pyarrow.parquet as pq
    for f in parquets:
        t = pq.read_table(f, columns=["image", "label"])
        for img, lab in zip(t.column("image").to_pylist(), t.column("label").to_pylist()):
            b = img["bytes"] if isinstance(img, dict) else img
            h = hashlib.md5(b).hexdigest()
            if h not in seen:
                seen.add(h); items.append(("general", "bytes", b, 1 - int(lab)))
        print("read", f.name)

print(f"total unique images: {len(items):,}  (parquet files used: {[f.name for f in parquets]})")
if any("train-" in f.name for f in parquets):
    print("WARNING: train-*.parquet files are present, so this split will NOT match v6. "
          "Move them out of dataset/general and run again.")

idx = np.random.default_rng(SEED).permutation(len(items))
n_test = len(idx) // 10
test_items = [items[i] for i in idx[:n_test]]
print(f"test images (never used for training): {len(test_items):,}")

# ---- save a sample of them as image files ----
if OUT_DIR.exists():
    shutil.rmtree(OUT_DIR)
for sub in ["real", "fake"]:
    (OUT_DIR / sub).mkdir(parents=True)

rng = np.random.default_rng(0)
counts = {}
for group in ["faces", "general"]:
    for lab, name in [(1, "real"), (0, "fake")]:
        pool = [it for it in test_items if it[0] == group and it[3] == lab]
        pick = rng.choice(len(pool), size=min(PER_GROUP, len(pool)), replace=False) if pool else []
        for k, j in enumerate(pick, 1):
            _, kind, ref, _ = pool[j]
            raw = Path(ref).read_bytes() if kind == "file" else ref
            with Image.open(io.BytesIO(raw)) as im:
                im = im.convert("RGB")
                im.save(OUT_DIR / name / f"{group}_{name}_{k:02d}.jpg", quality=95)
        counts[f"{group}/{name}"] = len(pick)

print("saved:", counts)
print("folder:", OUT_DIR)
