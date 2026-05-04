#!/usr/bin/env python3
"""
Download and prepare Open STT audiobook_2 subset for kokoro-ruslan training.

Downloads the audiobook_2 archive (~25.8 GB OPUS), filters to ~8,000 clean
utterances, converts to 22kHz WAV, and generates a metadata CSV compatible
with the kokoro-ruslan pipeline.

Usage:
    python scripts/download_open_stt_audiobooks.py [--output-dir ./open_stt_corpus] [--target-count 8000]

Prerequisites:
    pip install pandas tqdm
    brew install ffmpeg   # or apt install ffmpeg
"""

import argparse
import csv
import os
import re
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# Azure-hosted direct download links for Open STT OPUS data
BASE_URL = "https://azureopendatastorage.blob.core.windows.net/openstt/ru_open_stt_opus"
ARCHIVE_NAME = "archives/private_buriy_audiobooks_2.tar.gz"
MANIFEST_NAME = "manifests/private_buriy_audiobooks_2.csv"

# Filtering parameters
MIN_DURATION = 2.0       # seconds — skip very short clips
MAX_DURATION = 15.0      # seconds — skip overly long clips
MIN_TEXT_LEN = 5         # characters — skip near-empty transcripts
MAX_TEXT_LEN = 200       # characters — skip extremely long ones
TARGET_SAMPLE_RATE = 22050


def download_file(url: str, dest: Path, desc: str = "") -> bool:
    """Download a file using aria2c (fast, resumable) or fall back to wget/curl."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        print(f"  [skip] {dest.name} already exists ({dest.stat().st_size / 1e9:.2f} GB)")
        return True

    # Try aria2c first (multi-threaded, resumable)
    if _cmd_exists("aria2c"):
        print(f"  Downloading with aria2c: {desc or url}")
        ret = subprocess.run(
            ["aria2c", "-c", "-x", "8", "-s", "8", "--file-allocation=none",
             "-d", str(dest.parent), "-o", dest.name, url],
            check=False,
        )
        if ret.returncode == 0:
            return True

    # Fall back to wget
    if _cmd_exists("wget"):
        print(f"  Downloading with wget: {desc or url}")
        ret = subprocess.run(
            ["wget", "-c", "-O", str(dest), url],
            check=False,
        )
        if ret.returncode == 0:
            return True

    # Fall back to curl
    if _cmd_exists("curl"):
        print(f"  Downloading with curl: {desc or url}")
        ret = subprocess.run(
            ["curl", "-L", "-C", "-", "-o", str(dest), url],
            check=False,
        )
        if ret.returncode == 0:
            return True

    print(f"  [ERROR] No download tool available. Install aria2c, wget, or curl.")
    return False


def _cmd_exists(cmd: str) -> bool:
    return subprocess.run(
        ["which", cmd], capture_output=True, check=False
    ).returncode == 0


def extract_archive(archive_path: Path, extract_dir: Path) -> None:
    """Extract tar.gz archive."""
    marker = extract_dir / ".extraction_complete"
    if marker.exists():
        print(f"  [skip] Already extracted to {extract_dir}")
        return

    print(f"  Extracting {archive_path.name} ...")
    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=extract_dir, filter="data")
    marker.touch()
    print(f"  Extraction complete.")


def read_manifest(manifest_path: Path) -> pd.DataFrame:
    """Read Open STT manifest CSV (no header: wav_path, text_path, duration)."""
    df = pd.read_csv(
        manifest_path,
        names=["wav_path", "text_path", "duration"],
        dtype={"wav_path": str, "text_path": str, "duration": float},
    )
    print(f"  Manifest loaded: {len(df)} entries, {df['duration'].sum() / 3600:.1f} hours total")
    return df


def read_text_file(text_path: str, base_dir: Path) -> str:
    """Read transcription from a .txt file."""
    full_path = base_dir / text_path
    if not full_path.exists():
        return ""
    try:
        return full_path.read_text(encoding="utf-8").strip()
    except Exception:
        return ""


def clean_text(text: str) -> str:
    """Basic text cleaning for Russian TTS training."""
    # Remove multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    # Remove non-Russian characters except basic punctuation
    text = re.sub(r"[^\u0400-\u04FF\s.,!?;:\-—–\"'()«»]", "", text)
    # Collapse whitespace again after removals
    text = re.sub(r"\s+", " ", text).strip()
    return text


def filter_manifest(df: pd.DataFrame, base_dir: Path, target_count: int) -> pd.DataFrame:
    """Filter manifest entries for clean, single-reader audiobook utterances."""
    print(f"\n  Filtering {len(df)} entries...")

    # Duration filter
    df_filtered = df[
        (df["duration"] >= MIN_DURATION) & (df["duration"] <= MAX_DURATION)
    ].copy()
    print(f"    After duration filter ({MIN_DURATION}–{MAX_DURATION}s): {len(df_filtered)}")

    # Read texts and filter by length
    print("    Reading transcription files...")
    texts = []
    for _, row in tqdm(df_filtered.iterrows(), total=len(df_filtered), desc="    Reading texts"):
        texts.append(read_text_file(row["text_path"], base_dir))
    df_filtered["text"] = texts

    df_filtered["text_clean"] = df_filtered["text"].apply(clean_text)
    df_filtered["text_len"] = df_filtered["text_clean"].str.len()

    df_filtered = df_filtered[
        (df_filtered["text_len"] >= MIN_TEXT_LEN) &
        (df_filtered["text_len"] <= MAX_TEXT_LEN)
    ]
    print(f"    After text length filter ({MIN_TEXT_LEN}–{MAX_TEXT_LEN} chars): {len(df_filtered)}")

    # Remove entries with empty/whitespace-only text
    df_filtered = df_filtered[df_filtered["text_clean"].str.strip().astype(bool)]
    print(f"    After removing empty texts: {len(df_filtered)}")

    # Prefer entries with cleaner text (no numbers, few special chars)
    # Score: penalize digits and rare punctuation
    df_filtered["quality_score"] = (
        df_filtered["text_clean"].str.count(r"[а-яА-ЯёЁ]") /
        df_filtered["text_len"].clip(lower=1)
    )

    # Sort by quality (highest Cyrillic ratio first), then by duration (prefer 3-10s)
    df_filtered["duration_score"] = 1.0 - abs(df_filtered["duration"] - 6.0) / 10.0
    df_filtered["combined_score"] = df_filtered["quality_score"] * 0.7 + df_filtered["duration_score"] * 0.3
    df_filtered = df_filtered.sort_values("combined_score", ascending=False)

    # Select top N entries
    if len(df_filtered) > target_count:
        df_filtered = df_filtered.head(target_count)
        print(f"    Selected top {target_count} by quality score")
    else:
        print(f"    Available: {len(df_filtered)} (less than target {target_count})")

    total_hours = df_filtered["duration"].sum() / 3600
    print(f"    Final selection: {len(df_filtered)} utterances, {total_hours:.1f} hours")

    return df_filtered.reset_index(drop=True)


def convert_opus_to_wav(opus_path: Path, wav_path: Path) -> bool:
    """Convert OPUS file to 22kHz mono WAV using ffmpeg."""
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    if wav_path.exists():
        return True

    ret = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-i", str(opus_path),
            "-ar", str(TARGET_SAMPLE_RATE),
            "-ac", "1",
            "-sample_fmt", "s16",
            str(wav_path),
        ],
        capture_output=True,
        check=False,
    )
    return ret.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description="Download and prepare Open STT audiobook_2 for kokoro-ruslan"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./open_stt_corpus",
        help="Output directory for processed corpus (default: ./open_stt_corpus)"
    )
    parser.add_argument(
        "--download-dir", type=str, default="./open_stt_downloads",
        help="Directory for raw downloads (default: ./open_stt_downloads)"
    )
    parser.add_argument(
        "--target-count", type=int, default=8000,
        help="Target number of utterances to select (default: 8000)"
    )
    parser.add_argument(
        "--id-offset", type=int, default=100000,
        help="Starting ID number to avoid collisions with RUSLAN (default: 100000)"
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Skip download step (use existing files in download-dir)"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    download_dir = Path(args.download_dir)
    wavs_dir = output_dir / "wavs"
    wavs_dir.mkdir(parents=True, exist_ok=True)

    # ─── Step 1: Download ───────────────────────────────────────────────
    print("\n═══ Step 1: Download Open STT audiobook_2 ═══")

    archive_path = download_dir / "private_buriy_audiobooks_2.tar.gz"
    manifest_path = download_dir / "private_buriy_audiobooks_2.csv"

    if not args.skip_download:
        print(f"\n  Downloading manifest ({MANIFEST_NAME})...")
        if not download_file(
            f"{BASE_URL}/{MANIFEST_NAME}", manifest_path, "audiobook_2 manifest"
        ):
            sys.exit(1)

        print(f"\n  Downloading archive ({ARCHIVE_NAME}, ~25.8 GB)...")
        print("  This will take a while depending on your connection speed.")
        if not download_file(
            f"{BASE_URL}/{ARCHIVE_NAME}", archive_path, "audiobook_2 audio archive"
        ):
            sys.exit(1)
    else:
        if not manifest_path.exists():
            print(f"  [ERROR] Manifest not found at {manifest_path}")
            sys.exit(1)
        if not archive_path.exists():
            print(f"  [ERROR] Archive not found at {archive_path}")
            sys.exit(1)

    # ─── Step 2: Extract ────────────────────────────────────────────────
    print("\n═══ Step 2: Extract archive ═══")
    extract_dir = download_dir / "extracted"
    extract_archive(archive_path, extract_dir)

    # ─── Step 3: Read and filter manifest ───────────────────────────────
    print("\n═══ Step 3: Read and filter manifest ═══")
    df = read_manifest(manifest_path)

    # The manifest paths are relative — find the actual base directory
    # by checking where the files actually are after extraction.
    # Open STT uses hash-based paths like: a/b1/b2c3d4e5f6g7.opus
    sample_path = df.iloc[0]["wav_path"]
    possible_bases = [
        extract_dir,
        extract_dir / "private_buriy_audiobooks_2",
        extract_dir / "data",
    ]

    base_dir = None
    for candidate in possible_bases:
        if (candidate / sample_path).exists():
            base_dir = candidate
            break

    if base_dir is None:
        # Try to find it by searching
        print("  Searching for extracted audio files...")
        for root, dirs, files in os.walk(extract_dir):
            if any(f.endswith(".opus") for f in files):
                # Check if this matches the manifest structure
                rel = Path(root).relative_to(extract_dir)
                test_path = extract_dir / sample_path
                if test_path.exists():
                    base_dir = extract_dir
                    break
                # Try parent directories
                parts = sample_path.split("/")
                for depth in range(len(parts)):
                    candidate = Path(root)
                    for _ in range(depth):
                        candidate = candidate.parent
                    if (candidate / sample_path).exists():
                        base_dir = candidate
                        break
                if base_dir:
                    break

    if base_dir is None:
        print("  [WARNING] Could not auto-detect base directory for manifest paths.")
        print(f"  Trying extract_dir as base: {extract_dir}")
        print(f"  Sample manifest path: {sample_path}")
        print(f"  Listing top-level extracted contents:")
        for item in sorted(extract_dir.iterdir())[:20]:
            print(f"    {item.name}{'/' if item.is_dir() else ''}")
        print("\n  You may need to adjust base_dir manually and re-run with --skip-download")
        base_dir = extract_dir

    print(f"  Using base directory: {base_dir}")

    df_filtered = filter_manifest(df, base_dir, args.target_count)

    # ─── Step 4: Convert OPUS → WAV (22kHz mono) ───────────────────────
    print("\n═══ Step 4: Convert OPUS to 22kHz WAV ═══")

    # Check ffmpeg availability
    if not _cmd_exists("ffmpeg"):
        print("  [ERROR] ffmpeg not found. Install with: brew install ffmpeg")
        sys.exit(1)

    metadata_rows = []
    failed_conversions = 0

    for idx, row in tqdm(
        df_filtered.iterrows(), total=len(df_filtered), desc="  Converting"
    ):
        utterance_id = f"{args.id_offset + idx:06d}_OPENSTT"
        opus_path = base_dir / row["wav_path"]
        wav_path = wavs_dir / f"{utterance_id}.wav"

        if not opus_path.exists():
            failed_conversions += 1
            continue

        if convert_opus_to_wav(opus_path, wav_path):
            metadata_rows.append((utterance_id, row["text_clean"]))
        else:
            failed_conversions += 1

    print(f"\n  Converted: {len(metadata_rows)}")
    if failed_conversions > 0:
        print(f"  Failed: {failed_conversions}")

    # ─── Step 5: Generate metadata CSV ──────────────────────────────────
    print("\n═══ Step 5: Generate metadata CSV ═══")

    csv_path = output_dir / f"metadata_OPENSTT_{len(metadata_rows)}.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        for uid, text in metadata_rows:
            f.write(f"{uid}|{text}\n")

    print(f"  Written: {csv_path}")
    print(f"  Entries: {len(metadata_rows)}")
    print(f"  WAVs directory: {wavs_dir}")

    # ─── Summary ────────────────────────────────────────────────────────
    print("\n═══ Done! ═══")
    print(f"  Corpus location: {output_dir}")
    print(f"  Metadata CSV:    {csv_path}")
    print(f"  WAV files:       {wavs_dir}/ ({len(metadata_rows)} files)")
    print(f"\n  Next steps:")
    print(f"    1. Merge with RUSLAN metadata:")
    print(f"       cat ruslan_corpus/metadata_RUSLAN_22200.csv {csv_path} > merged_metadata.csv")
    print(f"    2. Symlink/copy WAVs into ruslan_corpus/wavs/ or update data_dir config")
    print(f"    3. Run MFA alignment on new utterances")
    print(f"    4. Clear feature cache (rm -rf data/processed/) and retrain")


if __name__ == "__main__":
    main()
