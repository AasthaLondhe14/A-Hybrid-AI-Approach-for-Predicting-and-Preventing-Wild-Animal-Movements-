import hashlib
import os
import random
import sys
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter, ImageOps


TARGET_ROOT = Path(r"E:\Large Animal DatasetF\raw-img")
CLASSES = ["lion", "tiger", "leopard", "bear"]
TARGET_COUNT = 1700
MIN_SIDE = 300
MIN_FILE_BYTES = 50_000
PROGRESS_EVERY = 50


def hash_image(img: Image.Image) -> str:
    return hashlib.md5(img.tobytes()).hexdigest()


def load_existing_hashes(out_dir: Path) -> set[str]:
    hashes: set[str] = set()
    for path in out_dir.glob("*.jpeg"):
        try:
            with Image.open(path) as img:
                img = ImageOps.exif_transpose(img)
                if img.mode != "RGB":
                    img = img.convert("RGB")
                hashes.add(hash_image(img))
        except Exception:
            continue
    return hashes


def augment_image(img: Image.Image) -> Image.Image:
    # Keep size; apply varied transforms while preserving content.
    w, h = img.size
    out = img

    if random.random() < 0.5:
        out = ImageOps.mirror(out)

    # Rotation (wider range)
    angle = random.uniform(-45, 45)
    out = out.rotate(angle, resample=Image.BICUBIC, expand=False)

    # Random crop/zoom then fit back to original size
    crop_scale = random.uniform(0.70, 0.98)
    cw = int(w * crop_scale)
    ch = int(h * crop_scale)
    if cw > 0 and ch > 0 and (cw != w or ch != h):
        left = random.randint(0, max(0, w - cw))
        top = random.randint(0, max(0, h - ch))
        out = out.crop((left, top, left + cw, top + ch))
        out = ImageOps.fit(out, (w, h), method=Image.BICUBIC)

    # Color jitter
    out = ImageEnhance.Brightness(out).enhance(random.uniform(0.85, 1.15))
    out = ImageEnhance.Contrast(out).enhance(random.uniform(0.85, 1.15))
    out = ImageEnhance.Color(out).enhance(random.uniform(0.80, 1.20))
    out = ImageEnhance.Sharpness(out).enhance(random.uniform(0.8, 1.3))

    # Mild Gaussian blur or sharpen (rare)
    if random.random() < 0.2:
        out = out.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 0.8)))

    # Add light noise to avoid duplicates (keeps detail)
    if random.random() < 0.9:
        pixels = out.load()
        for _ in range(int((w * h) * 0.005)):  # 0.5% pixels
            x = random.randint(0, w - 1)
            y = random.randint(0, h - 1)
            r, g, b = pixels[x, y]
            noise = random.randint(-8, 8)
            pixels[x, y] = (
                max(0, min(255, r + noise)),
                max(0, min(255, g + noise)),
                max(0, min(255, b + noise)),
            )

    return out


def augment_class(label: str) -> None:
    out_dir = TARGET_ROOT / label
    out_dir.mkdir(parents=True, exist_ok=True)
    images = list(out_dir.glob("*.jpeg"))
    if not images:
        print(f"[{label}] No images to augment.")
        return

    existing_hashes = load_existing_hashes(out_dir)
    count = len(images)
    if count >= TARGET_COUNT:
        print(f"[{label}] Already at {count}.")
        return

    attempts = 0
    max_attempts = (TARGET_COUNT - count) * 200

    while count < TARGET_COUNT and attempts < max_attempts:
        attempts += 1
        src_path = random.choice(images)
        try:
            with Image.open(src_path) as img:
                img = ImageOps.exif_transpose(img)
                if img.mode != "RGB":
                    img = img.convert("RGB")
                if min(img.size) < MIN_SIDE:
                    continue

                aug = augment_image(img)
                if min(aug.size) < MIN_SIDE:
                    continue
                h = hash_image(aug)
                if h in existing_hashes:
                    continue

                out_path = out_dir / f"{label}_aug_{count + 1:04d}.jpeg"
                aug.save(out_path, "JPEG", quality=92, dpi=(96, 96))
                if out_path.stat().st_size < MIN_FILE_BYTES:
                    out_path.unlink()
                    continue

                existing_hashes.add(h)
                count += 1
                if count % PROGRESS_EVERY == 0:
                    print(f"[{label}] {count}/{TARGET_COUNT}", flush=True)
        except Exception:
            continue

    print(f"[{label}] Final count: {count}")


def main() -> int:
    random.seed(42)
    selected = os.environ.get("ANIMAL_CLASSES", "").strip().lower()
    selected_set = {s.strip() for s in selected.split(",") if s.strip()} if selected else None
    for label in CLASSES:
        if selected_set and label not in selected_set:
            continue
        augment_class(label)
    return 0


if __name__ == "__main__":
    sys.exit(main())
