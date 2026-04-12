import hashlib
import os
import random
import sys
import time
from pathlib import Path
from typing import List

from icrawler.builtin import BingImageCrawler, GoogleImageCrawler, BaiduImageCrawler
from PIL import Image, ImageFilter, ImageOps, ImageStat


TARGET_ROOT = Path(r"E:\Large Animal DatasetF\raw-img")
MIN_SIDE = 300
MIN_FILE_BYTES = 80_000
MIN_PER_CLASS = 1700
MAX_PER_CLASS = 1700
SHARPNESS_THRESHOLD = 100.0
MIN_RAW_SHORT = 300
MIN_RAW_LONG = 300
RAW_TARGET = 30000
PER_QUERY = 1000
KEEP_RAW = True

QUERIES = {
    "lion": [
        "lion animal wildlife",
        "lion in savannah",
        "male lion close up",
        "lion face",
        "lion standing",
        "lion resting",
        "lion cub",
        "lion portrait",
        "lion closeup",
        "lion profile",
        "lion wildlife photography",
        "lion big cat",
        "lion head",
        "lion safari",
        "lion in grassland",
        "lion close-up photography",
        "lion in africa",
        "panthera leo",
        "lion zoo",
        "lion close up face",
        "lion full body",
        "lion side view",
        "lion wildlife portrait",
    ],
    "tiger": [
        "tiger animal wildlife",
        "bengal tiger",
        "siberian tiger",
        "tiger close up",
        "tiger face",
        "tiger standing",
        "tiger in forest",
        "tiger portrait",
        "tiger closeup",
        "tiger profile",
        "tiger wildlife photography",
        "tiger big cat",
        "tiger head",
        "tiger safari",
        "tiger in jungle",
        "tiger close-up photography",
        "tiger in india",
        "panthera tigris",
        "tiger zoo",
        "tiger close up face",
        "tiger full body",
        "tiger side view",
        "tiger wildlife portrait",
    ],
    "leopard": [
        "leopard animal wildlife",
        "african leopard",
        "snow leopard",
        "leopard close up",
        "leopard face",
        "leopard in tree",
        "leopard walking",
        "leopard portrait",
        "leopard closeup",
        "leopard profile",
        "leopard wildlife photography",
        "leopard big cat",
        "leopard head",
        "leopard safari",
        "leopard in jungle",
        "leopard close-up photography",
        "leopard in africa",
        "panthera pardus",
        "leopard zoo",
        "leopard close up face",
        "leopard full body",
        "leopard side view",
        "leopard wildlife portrait",
    ],
    "bear": [
        "bear animal wildlife",
        "brown bear",
        "black bear",
        "polar bear",
        "bear close up",
        "bear standing",
        "bear in forest",
        "bear portrait",
        "bear closeup",
        "bear profile",
        "bear wildlife photography",
        "bear head",
        "bear face",
        "bear in wild",
        "bear close-up photography",
        "ursus arctos",
        "ursus americanus",
        "ursus maritimus",
        "bear zoo",
        "bear close up face",
        "bear full body",
        "bear side view",
        "bear wildlife portrait",
    ],
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def clear_dir(path: Path) -> None:
    if path.exists():
        for child in path.iterdir():
            if child.is_file():
                child.unlink()
            elif child.is_dir():
                for sub in child.rglob("*"):
                    if sub.is_file():
                        sub.unlink()
                for sub in sorted(child.rglob("*"), reverse=True):
                    if sub.is_dir():
                        sub.rmdir()
                child.rmdir()
    else:
        path.mkdir(parents=True, exist_ok=True)


def hash_image(img: Image.Image) -> str:
    return hashlib.md5(img.tobytes()).hexdigest()


def is_sharp_enough(img: Image.Image) -> bool:
    gray = img.convert("L")
    edges = gray.filter(ImageFilter.FIND_EDGES)
    variance = ImageStat.Stat(edges).var[0]
    return variance >= SHARPNESS_THRESHOLD


def _crawl_source(crawler_cls, query: str, raw_dir: Path, max_num: int) -> None:
    crawler = crawler_cls(
        storage={"root_dir": str(raw_dir)},
        feeder_threads=6,
        parser_threads=6,
        downloader_threads=24,
    )
    crawler.crawl(keyword=query, filters={"size": "large"}, max_num=max_num)


def download_images(label: str, raw_dir: Path, target: int) -> None:
    ensure_dir(raw_dir)
    queries = QUERIES.get(label, [label])
    per_query = max(PER_QUERY, target // max(1, len(queries)))
    sources = [BingImageCrawler, GoogleImageCrawler, BaiduImageCrawler]

    for source in sources:
        for query in queries:
            _crawl_source(source, query, raw_dir, per_query)
            filter_raw_dir(raw_dir)
            if len(list(raw_dir.glob("*"))) >= target:
                return
        if len(list(raw_dir.glob("*"))) >= target:
            return


def filter_raw_dir(raw_dir: Path) -> None:
    for path in raw_dir.glob("*"):
        if not path.is_file():
            continue
        try:
            with Image.open(path) as img:
                width, height = img.size
                short = min(width, height)
                long = max(width, height)
                if short < MIN_RAW_SHORT or long < MIN_RAW_LONG:
                    path.unlink()
        except Exception:
            try:
                path.unlink()
            except Exception:
                pass


def load_existing_hashes(out_dir: Path) -> Set[str]:
    hashes: Set[str] = set()
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


def process_images(label: str, raw_dir: Path, out_dir: Path, target: int) -> int:
    hashes = load_existing_hashes(out_dir)
    existing = sorted(out_dir.glob(f"{label}_*.jpeg"))
    count = len(existing)
    if count >= target:
        return count
    paths = list(raw_dir.glob("*"))
    random.shuffle(paths)
    for path in paths:
        try:
            with Image.open(path) as img:
                img = ImageOps.exif_transpose(img)
                if img.mode != "RGB":
                    img = img.convert("RGB")
                width, height = img.size
                if min(width, height) < MIN_SIDE:
                    continue
                if path.stat().st_size < MIN_FILE_BYTES:
                    continue
                if not is_sharp_enough(img):
                    continue
                h = hash_image(img)
                if h in hashes:
                    continue
                hashes.add(h)
                out_path = out_dir / f"{label}_{count + 1:04d}.jpeg"
                img.save(out_path, "JPEG", quality=92, dpi=(96, 96))
                count += 1
                if count >= target:
                    break
        except Exception:
            continue
    return count


def purge_bad_outputs(out_dir: Path) -> int:
    removed = 0
    for path in out_dir.glob("*.jpeg"):
        try:
            if path.stat().st_size < MIN_FILE_BYTES:
                path.unlink()
                removed += 1
                continue
            with Image.open(path) as img:
                width, height = img.size
                if min(width, height) < MIN_SIDE:
                    path.unlink()
                    removed += 1
        except Exception:
            try:
                path.unlink()
                removed += 1
            except Exception:
                pass
    return removed


def main() -> int:
    random.seed(42)
    print("Starting Bing image download...", flush=True)

    selected = os.environ.get("ANIMAL_CLASSES", "").strip().lower()
    selected_set = {s.strip() for s in selected.split(",") if s.strip()} if selected else None

    for label in QUERIES.keys():
        if selected_set and label not in selected_set:
            continue
        out_dir = TARGET_ROOT / label
        raw_dir = TARGET_ROOT / label / "_raw"
        ensure_dir(out_dir)
        ensure_dir(raw_dir)

        try:
            existing_count = len(list(out_dir.glob("*.jpeg")))
            if existing_count < MAX_PER_CLASS:
                print(f"[{label}] Downloading images from web sources...", flush=True)
                download_images(label, raw_dir, RAW_TARGET)
                filter_raw_dir(raw_dir)
            else:
                print(f"[{label}] Already complete with {existing_count} images.", flush=True)

            print(
                f"[{label}] Filtering and saving JPEGs (min {MIN_SIDE}px, size >= {MIN_FILE_BYTES} bytes, sharpness >= {SHARPNESS_THRESHOLD})...",
                flush=True,
            )
            count = process_images(label, raw_dir, out_dir, MAX_PER_CLASS)

            print(f"[{label}] Final count: {count}", flush=True)
            if count < MIN_PER_CLASS:
                print(f"[{label}] Below minimum; fetching more raw images...", flush=True)
                download_images(label, raw_dir, RAW_TARGET * 2)
                filter_raw_dir(raw_dir)
                count = process_images(label, raw_dir, out_dir, MAX_PER_CLASS)
                print(f"[{label}] Final count after retry: {count}", flush=True)
                if count < MIN_PER_CLASS:
                    print(f"[{label}] WARNING: only {count} valid images (below {MIN_PER_CLASS}).", flush=True)
            removed = purge_bad_outputs(out_dir)
            if removed:
                print(f"[{label}] Removed {removed} low-quality outputs.", flush=True)
            if not KEEP_RAW:
                clear_dir(raw_dir)
        except Exception as exc:
            print(f"[{label}] ERROR: {exc}", flush=True)

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
