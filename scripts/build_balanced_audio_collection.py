import csv
import io
import json
import random
import shutil
import subprocess
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


BASE = Path(r"E:\balanced dataset collection")
RAW_ROOT = BASE / "raw_pool"
FINAL_ROOT = BASE / "final_300"
SPLITS_ROOT = FINAL_ROOT / "_splits"
CACHE_ROOT = BASE / "_cache"
LEGACY_AUDIO_ROOT = Path(r"E:\datasets\audio_datasets")
CLASS_LABELS_URL = "https://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv"
SEGMENT_URLS = [
    "https://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/balanced_train_segments.csv",
    "https://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/unbalanced_train_segments.csv",
]
RANDOM_SEED = 42
MIN_BYTES = 32_000
TARGET_FINAL = 300
TRAIN_COUNT = 240
VAL_COUNT = 30
TEST_COUNT = 30
MAX_DOWNLOAD_WORKERS = 4

# Over-collect the confusing classes and later keep the cleanest 300.
TARGETS = {
    "cat": {
        "raw_target": 350,
        "labels": ["Cat", "Meow", "Purr"],
    },
    "dog": {
        "raw_target": 350,
        "labels": ["Dog", "Bark", "Bow-wow", "Howl", "Growling"],
    },
    "cow": {
        "raw_target": 350,
        "labels": ["Cattle, bovinae", "Moo", "Livestock, farm animals, working animals"],
    },
    "horse": {
        "raw_target": 500,
        "labels": ["Horse", "Neigh, whinny", "Clip-clop"],
    },
    "lion": {
        "raw_target": 500,
        "labels": ["Lion", "Roar", "Roaring cats (lions, tigers)"],
    },
    "elephant": {
        "raw_target": 500,
        "labels": ["Elephant"],
        "search_queries": [
            "elephant trumpet sound",
            "elephant rumble sound",
            "wild elephant call audio",
            "elephant vocalization sound",
        ],
    },
    "bear": {
        "raw_target": 500,
        "labels": ["Bear", "Growling"],
        "search_queries": [
            "bear growl sound",
            "bear roar sound",
            "wild bear vocalization audio",
            "bear grunting sound",
        ],
    },
    "monkey": {
        "raw_target": 350,
        "labels": ["Monkey", "Baboon", "Chimpanzee", "Animal"],
    },
}


def iter_csv_rows(url: str, skiprows: int = 0):
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    cache_name = url.rsplit("/", 1)[-1]
    cache_path = CACHE_ROOT / cache_name

    if cache_path.exists():
        with cache_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            for _ in range(skiprows):
                next(reader, None)
            for row in reader:
                yield row
    else:
        with urllib.request.urlopen(url) as response:
            text = response.read().decode("utf-8")
        cache_path.write_text(text, encoding="utf-8")
        reader = csv.reader(io.StringIO(text))
        for _ in range(skiprows):
            next(reader, None)
        for row in reader:
            yield row


def build_label_map():
    rows = iter_csv_rows(CLASS_LABELS_URL)
    next(rows, None)
    return {display_name: mid for _, mid, display_name in rows}


def count_wavs(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(1 for _ in root.rglob("*.wav"))


def ensure_clean_wav(path: Path) -> bool:
    return path.exists() and path.stat().st_size >= MIN_BYTES


def import_existing_sources(class_name: str):
    out_dir = RAW_ROOT / class_name
    out_dir.mkdir(parents=True, exist_ok=True)
    imported = 0
    seen = {p.name for p in out_dir.glob("*.wav")}

    candidate_roots = [
        LEGACY_AUDIO_ROOT / "prepared" / class_name,
        LEGACY_AUDIO_ROOT / "audioset" / class_name,
    ]

    for root in candidate_roots:
        if not root.exists():
            continue
        for src in root.rglob("*.wav"):
            if src.name in seen:
                continue
            if not ensure_clean_wav(src):
                continue
            dst = out_dir / src.name
            try:
                shutil.copy2(src, dst)
            except PermissionError:
                continue
            seen.add(src.name)
            imported += 1
    return imported


def build_candidates_by_class(label_map: dict[str, str]):
    target_mids = {
        class_name: {label_map[label] for label in info["labels"] if label in label_map}
        for class_name, info in TARGETS.items()
    }
    candidates = {class_name: [] for class_name in TARGETS}
    seen = {class_name: set() for class_name in TARGETS}

    for url in SEGMENT_URLS:
        for row in iter_csv_rows(url, skiprows=3):
            if len(row) < 4:
                continue
            ytid = row[0].strip()
            start = float(row[1].strip())
            end = float(row[2].strip())
            labels = {label.strip() for label in row[3].replace('"', "").split(",") if label.strip()}
            item = {
                "ytid": ytid,
                "start": start,
                "end": end,
                "labels": labels,
            }
            key = (ytid, int(start), int(end))
            for class_name, mids in target_mids.items():
                if not mids or not (labels & mids) or key in seen[class_name]:
                    continue
                seen[class_name].add(key)
                candidates[class_name].append(item)
    return candidates


def search_youtube_candidates(query: str, limit: int = 40):
    cmd = [
        "yt-dlp",
        "--flat-playlist",
        "--dump-single-json",
        f"ytsearch{limit}:{query}",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not result.stdout.strip():
        return []
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        return []

    candidates = []
    for entry in data.get("entries", []) or []:
        video_id = entry.get("id")
        duration = entry.get("duration") or 0
        if not video_id or duration < 8:
            continue
        clip_len = min(10, int(duration))
        start = max(0, int(duration // 3))
        end = start + clip_len
        candidates.append(
            {
                "ytid": video_id,
                "start": float(start),
                "end": float(end),
                "labels": set(),
            }
        )
    return candidates


def append_search_candidates(candidates_by_class: dict[str, list[dict]]):
    for class_name, info in TARGETS.items():
        queries = info.get("search_queries", [])
        if not queries:
            continue
        seen = {(item["ytid"], int(item["start"]), int(item["end"])) for item in candidates_by_class[class_name]}
        for query in queries:
            for item in search_youtube_candidates(query):
                key = (item["ytid"], int(item["start"]), int(item["end"]))
                if key in seen:
                    continue
                seen.add(key)
                candidates_by_class[class_name].append(item)
    return candidates_by_class


def download_segment(class_name: str, item: dict) -> Path | None:
    out_dir = RAW_ROOT / class_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{item['ytid']}_{item['start']:.0f}-{item['end']:.0f}.wav"
    if ensure_clean_wav(out_path):
        return out_path
    if out_path.exists():
        try:
            out_path.unlink()
        except PermissionError:
            return None

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
            try:
                out_path.unlink()
            except PermissionError:
                return None
        return None
    return out_path


def download_until_target(class_name: str, candidates: list[dict], raw_target: int):
    existing = count_wavs(RAW_ROOT / class_name)
    if existing >= raw_target:
        return 0

    downloaded_now = 0
    idx = 0
    while count_wavs(RAW_ROOT / class_name) < raw_target and idx < len(candidates):
        remaining = raw_target - count_wavs(RAW_ROOT / class_name)
        batch = candidates[idx:idx + max(remaining * 2, MAX_DOWNLOAD_WORKERS)]
        if not batch:
            break
        idx += len(batch)
        with ThreadPoolExecutor(max_workers=MAX_DOWNLOAD_WORKERS) as pool:
            futures = [pool.submit(download_segment, class_name, item) for item in batch]
            for future in as_completed(futures):
                if future.result() is not None:
                    downloaded_now += 1
                if count_wavs(RAW_ROOT / class_name) >= raw_target:
                    break
    return downloaded_now


def create_final_dataset():
    rng = random.Random(RANDOM_SEED)
    manifest = {"classes": {}}

    if FINAL_ROOT.exists():
        shutil.rmtree(FINAL_ROOT)

    for class_name, info in TARGETS.items():
        raw_dir = RAW_ROOT / class_name
        wavs = sorted(raw_dir.glob("*.wav"))
        rng.shuffle(wavs)
        selected = wavs[:TARGET_FINAL]

        class_final_dir = FINAL_ROOT / class_name
        class_final_dir.mkdir(parents=True, exist_ok=True)
        for src in selected:
            shutil.copy2(src, class_final_dir / src.name)

        split_map = {
            "train": selected[:TRAIN_COUNT],
            "val": selected[TRAIN_COUNT:TRAIN_COUNT + VAL_COUNT],
            "test": selected[TRAIN_COUNT + VAL_COUNT:TRAIN_COUNT + VAL_COUNT + TEST_COUNT],
        }
        for split, split_files in split_map.items():
            split_dir = SPLITS_ROOT / split / class_name
            split_dir.mkdir(parents=True, exist_ok=True)
            for src in split_files:
                shutil.copy2(src, split_dir / src.name)

        manifest["classes"][class_name] = {
            "raw_count": len(wavs),
            "selected_final": len(selected),
            "splits": {split: len(files) for split, files in split_map.items()},
        }

    (BASE / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main():
    label_map = build_label_map()
    candidates_by_class = build_candidates_by_class(label_map)
    candidates_by_class = append_search_candidates(candidates_by_class)
    summary = {"download": {}, "selection": {}}

    for class_name, info in TARGETS.items():
        raw_target = info["raw_target"]
        imported_now = import_existing_sources(class_name)
        candidates = candidates_by_class[class_name]
        downloaded_now = download_until_target(class_name, candidates, raw_target)

        summary["download"][class_name] = {
            "raw_target": raw_target,
            "raw_count": count_wavs(RAW_ROOT / class_name),
            "imported_now": imported_now,
            "downloaded_now": downloaded_now,
            "candidate_count": len(candidates),
        }

    summary["selection"] = create_final_dataset()["classes"]
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
