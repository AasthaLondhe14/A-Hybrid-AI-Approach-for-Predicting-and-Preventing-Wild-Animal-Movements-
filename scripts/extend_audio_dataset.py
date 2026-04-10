import csv
import json
import random
import shutil
import subprocess
import urllib.request
from collections import defaultdict
from pathlib import Path


BASE = Path(r"E:\datasets\audio_datasets")
AUDIOSET_ROOT = BASE / "audioset"
PREPARED_ROOT = BASE / "prepared"
SPLITS_ROOT = PREPARED_ROOT / "_splits"
EXPORT_ROOT = BASE / "additional_upload_wildlife_audio"
CLASS_LABELS_URL = "https://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv"
SEGMENT_URLS = [
    "https://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/balanced_train_segments.csv",
    "https://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/unbalanced_train_segments.csv",
]
RANDOM_SEED = 42
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
MIN_BYTES = 32_000

# Keep these classes close to the smallest current wild-animal class instead of
# trying to rebalance the whole historical dataset.
REFERENCE_CLASS = "lion"
TARGETS = {
    "bear": ["Bear", "Growling", "Wild animals"],
    "elephant": ["Elephant", "Wild animals"],
    "lion": ["Lion", "Roaring cats (lions, tigers)", "Roar", "Growling"],
}


def read_csv_from_url(url: str, skiprows: int = 0):
    with urllib.request.urlopen(url) as response:
        text = response.read().decode("utf-8")
    lines = text.splitlines()
    if skiprows:
        lines = lines[skiprows:]
    return list(csv.reader(lines))


def build_label_map():
    rows = read_csv_from_url(CLASS_LABELS_URL)
    return {display_name: mid for _, mid, display_name in rows[1:]}


def load_segments():
    segments = []
    for url in SEGMENT_URLS:
        rows = read_csv_from_url(url, skiprows=3)
        for row in rows:
            if len(row) < 4:
                continue
            ytid = row[0].strip()
            start = float(row[1].strip())
            end = float(row[2].strip())
            labels = [label.strip() for label in row[3].replace('"', "").split(",") if label.strip()]
            segments.append(
                {
                    "ytid": ytid,
                    "start": start,
                    "end": end,
                    "labels": labels,
                }
            )
    return segments


def count_wavs(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(1 for _ in root.rglob("*.wav"))


def ensure_clean_wav(path: Path) -> bool:
    return path.exists() and path.stat().st_size >= MIN_BYTES


def download_segment(class_name: str, item: dict) -> Path | None:
    out_dir = AUDIOSET_ROOT / class_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{item['ytid']}_{item['start']:.0f}-{item['end']:.0f}.wav"
    if ensure_clean_wav(out_path):
        return out_path
    if out_path.exists():
        out_path.unlink()

    cmd = [
        "yt-dlp",
        "-x",
        "--audio-format",
        "wav",
        "--audio-quality",
        "5",
        "--output",
        str(out_path),
        "--postprocessor-args",
        f"ffmpeg:-ss {item['start']} -to {item['end']}",
        f"https://www.youtube.com/watch?v={item['ytid']}",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not ensure_clean_wav(out_path):
        if out_path.exists():
            out_path.unlink()
        return None
    return out_path


def select_candidates(label_map: dict[str, str], segments: list[dict], class_name: str):
    mids = [label_map[label] for label in TARGETS[class_name] if label in label_map]
    matches = []
    seen = set()
    for item in segments:
        if not any(mid in item["labels"] for mid in mids):
            continue
        key = (item["ytid"], int(item["start"]), int(item["end"]))
        if key in seen:
            continue
        seen.add(key)
        matches.append(item)
    return matches


def sync_prepared(class_name: str):
    src_root = AUDIOSET_ROOT / class_name
    dst_root = PREPARED_ROOT / class_name / "audioset"
    dst_root.mkdir(parents=True, exist_ok=True)
    new_files = []
    for wav in sorted(src_root.glob("*.wav")):
        dst = dst_root / wav.name
        if not dst.exists():
            shutil.copy2(wav, dst)
            new_files.append(dst)
    return new_files


def rebuild_class_splits(class_name: str):
    files = sorted((PREPARED_ROOT / class_name).rglob("*.wav"))
    if not files:
        return {"train": 0, "val": 0, "test": 0}

    rng = random.Random(RANDOM_SEED)
    rng.shuffle(files)
    total = len(files)
    n_train = int(total * TRAIN_RATIO)
    n_val = int(total * VAL_RATIO)
    split_map = {
        "train": files[:n_train],
        "val": files[n_train:n_train + n_val],
        "test": files[n_train + n_val:],
    }

    for split in ("train", "val", "test"):
        split_dir = SPLITS_ROOT / split / class_name
        if split_dir.exists():
            shutil.rmtree(split_dir)
        split_dir.mkdir(parents=True, exist_ok=True)
        for wav in split_map[split]:
            shutil.copy2(wav, split_dir / wav.name)

    return {split: len(items) for split, items in split_map.items()}


def export_upload_bundle(manifest: dict, new_prepared_by_class: dict[str, list[Path]], new_split_names_by_class: dict[str, dict[str, list[str]]]):
    if EXPORT_ROOT.exists():
        shutil.rmtree(EXPORT_ROOT)
    export_prepared = EXPORT_ROOT / "prepared"
    export_splits = EXPORT_ROOT / "prepared" / "_splits"

    for class_name in TARGETS:
        class_prepared_root = export_prepared / class_name / "audioset"
        class_prepared_root.mkdir(parents=True, exist_ok=True)
        for src in new_prepared_by_class.get(class_name, []):
            shutil.copy2(src, class_prepared_root / src.name)
        for split in ("train", "val", "test"):
            split_dir = export_splits / split / class_name
            split_dir.mkdir(parents=True, exist_ok=True)
            src_split_root = SPLITS_ROOT / split / class_name
            for file_name in new_split_names_by_class.get(class_name, {}).get(split, []):
                src = src_split_root / file_name
                if src.exists():
                    shutil.copy2(src, split_dir / file_name)

    (EXPORT_ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main():
    label_map = build_label_map()
    segments = load_segments()

    target_total = count_wavs(PREPARED_ROOT / REFERENCE_CLASS)
    if target_total == 0:
        target_total = 180

    manifest = {
        "reference_class": REFERENCE_CLASS,
        "target_total_per_new_class": target_total,
        "classes": {},
    }

    new_prepared_by_class = {}
    new_split_names_by_class = {}

    for class_name in TARGETS:
        existing_total = count_wavs(PREPARED_ROOT / class_name)
        needed = max(0, target_total - existing_total)
        candidates = select_candidates(label_map, segments, class_name)
        downloaded = 0

        for item in candidates:
            if count_wavs(AUDIOSET_ROOT / class_name) >= target_total:
                break
            if download_segment(class_name, item):
                downloaded += 1

        new_prepared = sync_prepared(class_name)
        split_counts = rebuild_class_splits(class_name)
        new_prepared_by_class[class_name] = new_prepared
        new_split_names_by_class[class_name] = {
            split: [
                src.name for src in new_prepared
                if (SPLITS_ROOT / split / class_name / src.name).exists()
            ]
            for split in ("train", "val", "test")
        }
        manifest["classes"][class_name] = {
            "existing_before": existing_total,
            "requested_new": needed,
            "downloaded_or_reused_in_audioset": count_wavs(AUDIOSET_ROOT / class_name),
            "copied_into_prepared_now": len(new_prepared),
            "prepared_total": count_wavs(PREPARED_ROOT / class_name),
            "splits": split_counts,
            "new_split_files": {k: len(v) for k, v in new_split_names_by_class[class_name].items()},
            "candidate_count_seen": len(candidates),
            "download_attempts_that_succeeded_now": downloaded,
        }

    export_upload_bundle(manifest, new_prepared_by_class, new_split_names_by_class)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
