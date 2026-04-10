from __future__ import annotations

import hashlib
import shutil
import subprocess
import tempfile
from pathlib import Path


DATASET_ROOT = Path(r"E:\balanced dataset collection\final_300_merged_clean_v1")
COOKIES_PATH = Path(r"E:\models\www.youtube.com_cookies.txt")

LINKS = {
    "dog": [
        "https://youtube.com/shorts/LUThEQymCHQ?si=27EavljgeKs-s0l6",
        "https://youtube.com/shorts/hWlh8JHElmE?si=iVf9fXrCWGzQZoxc",
        "https://youtube.com/shorts/EiWUfn2QZSM?si=d2YONv8Hf3mzLXJ5",
        "https://youtube.com/shorts/kinHvTtDiWk?si=H1EZgByVMhK0-buQ",
        "https://youtube.com/shorts/-zWEWpQ260A?si=E6TF7A4dlFicr6QN",
        "https://youtube.com/shorts/B5F9d1i3s6M?si=XTp10RRGDZFvh5OL",
        "https://youtube.com/shorts/YJSmB9LiC-Y?si=ktlWsIYH_UNXvqBX",
        "https://youtube.com/shorts/1AmOFDv45Oc?si=2d4N6ujgIdcC2rCM",
        "https://youtube.com/shorts/LhqeXKtsJLo?si=PECxX2wItecE-gdt",
        "https://youtube.com/shorts/8KJeJeu1iwY?si=BB63N0oj_9ntwoVx",
    ],
    "bear": [
        "https://youtube.com/shorts/P49C2NXlNIM?si=GO-n6Z1B87u1IKmT",
        "https://youtube.com/shorts/Dsp90nzEzgo?si=hrxN-MGvhmPmOU_7",
        "https://youtube.com/shorts/e4eimAVHdxY?si=dR_b9L_EmxsoZH0I",
        "https://youtube.com/shorts/Z86uYOWcHc8?si=ovj6q3l4Z8_LEnWS",
    ],
    "elephant": [
        "https://youtube.com/shorts/Toy0qx06VQM?si=1mu9TQT4WMqr8ufR",
        "https://youtube.com/shorts/YZKy5cv4E7Y?si=yjtwj_HICrEQn14l",
        "https://youtube.com/shorts/2eGFdnvIUbo?si=f09Rr-EZU9pFcmMC",
        "https://youtube.com/shorts/3VmysBGewME?si=XvxA70a7_yE76fvw",
        "https://youtube.com/shorts/q-sfkL2D7p4?si=vXpqTHMhuoexfUNZ",
        "https://youtube.com/shorts/WCcPd57ozk8?si=EWOFk45KARD6C16m",
        "https://youtube.com/shorts/An44Lb1reJM?si=LC7gej9k4E8_Jc_x",
    ],
    "horse": [
        "https://youtube.com/shorts/erYhEM4qLpU?si=uUVwikqccIXennJr",
        "https://youtube.com/shorts/loEjllEcWgE?si=srL8kmXpfYGG1Vcf",
        "https://youtube.com/shorts/3BW1lBgtbbs?si=5yiryPYygbM9DNSH",
        "https://youtube.com/shorts/7m21ZBUnLFw?si=wkdbQb6UiAa38Oyd",
    ],
    "lion": [
        "https://youtube.com/shorts/r_VU1bGTGqY?si=1nzTOWy_c8TagnDC",
        "https://youtube.com/shorts/Z6qhi-PaZvU?si=JlAcH6FVytwUvLdA",
        "https://youtube.com/shorts/x6FeFwnOT-Q?si=CeKKe7OAgK6Rb9cH",
        "https://youtube.com/shorts/JmFpSyMHuSA?si=8bnAm17R1iDtmit_",
        "https://youtube.com/shorts/0dbu5Q3l-yM?si=-Wz7zIoprUnZeQOL",
    ],
    "monkey": [
        "https://youtube.com/shorts/_khrPd-qh54?si=N6PZGHvKRE4JuBpC",
        "https://youtube.com/shorts/8OJQ4sQrMbA?si=oxsr-66r_RXwgZOs",
        "https://youtube.com/shorts/RU01wauDHfs?si=vTAg3ABKAQV5gMoM",
        "https://youtube.com/shorts/tG-m6NVisYc?si=UbiMgMsMvJWoyyZX",
        "https://youtube.com/shorts/sBEQOpa96CE?si=ynQKOyqCdrrja1le",
    ],
}


def run(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        text=True,
        capture_output=True,
        check=False,
    )


def sanitized_name(url: str) -> str:
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]
    return f"youtube_url_{digest}.wav"


def ensure_dirs(label: str) -> tuple[Path, Path]:
    class_dir = DATASET_ROOT / label
    train_dir = DATASET_ROOT / "_splits" / "train" / label
    class_dir.mkdir(parents=True, exist_ok=True)
    train_dir.mkdir(parents=True, exist_ok=True)
    return class_dir, train_dir


def download_audio(url: str, out_dir: Path) -> Path:
    target = out_dir / "%(id)s.%(ext)s"
    cmd = [
        "yt-dlp",
        "--cookies",
        str(COOKIES_PATH),
        "--extract-audio",
        "--audio-format",
        "wav",
        "--audio-quality",
        "0",
        "-o",
        str(target),
        url,
    ]
    result = run(cmd)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "yt-dlp failed")
    wavs = sorted(out_dir.glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not wavs:
        raise RuntimeError("No wav file produced")
    return wavs[0]


def convert_clean_wav(src: Path, dst: Path) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-af",
        "loudnorm",
        str(dst),
    ]
    result = run(cmd)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "ffmpeg conversion failed")


def import_links() -> None:
    imported = []
    skipped = []
    failed = []

    with tempfile.TemporaryDirectory(prefix="yt_audio_import_") as tmpdir:
        tmp = Path(tmpdir)
        for label, urls in LINKS.items():
            class_dir, train_dir = ensure_dirs(label)
            for url in urls:
                file_name = sanitized_name(url)
                class_target = class_dir / file_name
                train_target = train_dir / file_name

                if class_target.exists() and train_target.exists():
                    skipped.append((label, url, "already exists"))
                    continue

                work_dir = tmp / label
                if work_dir.exists():
                    shutil.rmtree(work_dir)
                work_dir.mkdir(parents=True, exist_ok=True)

                try:
                    downloaded = download_audio(url, work_dir)
                    convert_clean_wav(downloaded, class_target)
                    shutil.copy2(class_target, train_target)
                    imported.append((label, url, str(class_target)))
                except Exception as exc:
                    failed.append((label, url, str(exc)))

    print("Imported:")
    for item in imported:
        print(item)
    print("\nSkipped:")
    for item in skipped:
        print(item)
    print("\nFailed:")
    for item in failed:
        print(item)


if __name__ == "__main__":
    import_links()
