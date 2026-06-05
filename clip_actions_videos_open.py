"""
Create 3-second action clips from whole-length surgical videos.

Expected folder layout:

  Cholec80_Action/
  |-- clip_actions_videos.py
  |-- metadata_actions.csv
  |-- whole_length_videos/
  |   |-- video01.mp4
  |   |-- video02.mp4
  |   |-- ...
  |   `-- video80.mp4
  `-- videos/
      |-- video01/
      |   |-- 00_22_Dissecting.mp4
      |   `-- ...
      `-- video80/

Usage:

  python clip_actions_videos.py --num_workers 40

The script always:
- reads videos from ./whole_length_videos
- reads annotations from ./metadata_actions.csv
- writes clips to ./videos/videoXX/
- exports only 3-second clips
- names video ids as split-style ids, e.g. video03
"""

import argparse
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Optional

import cv2
import pandas as pd
from tqdm import tqdm


VIDEOS_DIR = "./whole_length_videos"
META_CSV = "./metadata_actions.csv"
OUT_VIDEOS_DIR = "./videos"
CLIP_LEN = 3

ACTION_TO_LABEL = {
    "Dissecting": 0,
    "Exposing": 1,
    "Cutting": 2,
    "Suctioning/Irrigating": 3,
    "Suctioning": 3,
    "Irrigating": 3,
    "Coagulating": 4,
    "Clipping/Unclipping": 5,
    "Clipping": 5,
    "Unclipping": 5,
    "Idle": 6,
}

LABEL_TO_ACTION = {
    0: "Dissecting",
    1: "Exposing",
    2: "Cutting",
    3: "Suctioning_Irrigating",
    4: "Coagulating",
    5: "Clipping_Unclipping",
    6: "Idle",
}


def log(msg: str) -> None:
    print(msg, flush=True)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def seconds(minutes: int, seconds_: int) -> int:
    return int(minutes) * 60 + int(seconds_)


def video_id_from_name(name: str) -> str:
    stem = os.path.splitext(os.path.basename(str(name).strip()))[0]
    match = re.search(r"(\d+)$", stem)
    if match is None:
        raise ValueError(
            f"Cannot infer video id from '{name}'. "
            "Video names must end with a number, e.g. video03."
        )
    return f"video{int(match.group(1)):02d}"


def norm_stem(name: str) -> str:
    return os.path.splitext(os.path.basename(str(name).strip()))[0].lower()


def build_video_file_index(videos_dir: str) -> Dict[str, str]:
    if not os.path.isdir(videos_dir):
        raise FileNotFoundError(f"Missing input video folder: {videos_dir}")

    index = {}
    for filename in os.listdir(videos_dir):
        if filename.lower().endswith(".mp4"):
            index[norm_stem(filename)] = os.path.join(videos_dir, filename)
    return index


def resolve_video_path(video_name: str, video_file_index: Dict[str, str]) -> Optional[str]:
    split_id = video_id_from_name(video_name)
    return video_file_index.get(norm_stem(video_name)) or video_file_index.get(split_id)


def have_ffmpeg() -> bool:
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return True
    except Exception:
        return False


def get_video_duration(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if fps <= 0 or frame_count <= 0:
        raise RuntimeError(f"Cannot read duration from video: {video_path}")
    return frame_count / fps


def load_metadata(meta_csv: str) -> pd.DataFrame:
    if not os.path.isfile(meta_csv):
        raise FileNotFoundError(f"Missing metadata CSV: {meta_csv}")

    df = pd.read_csv(meta_csv)
    required = {
        "video",
        "action",
        "action_initial_minute",
        "action_initial_second",
        "action_final_minute",
        "action_final_second",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"metadata_actions.csv is missing columns: {missing}")

    df["video"] = df["video"].astype(str).str.strip()
    df["action"] = df["action"].astype(str).str.strip()
    df["start_s"] = df.apply(
        lambda row: seconds(row["action_initial_minute"], row["action_initial_second"]),
        axis=1,
    )
    df["end_s"] = df.apply(
        lambda row: seconds(row["action_final_minute"], row["action_final_second"]),
        axis=1,
    )
    return df


def build_second_level_labels(df_video: pd.DataFrame) -> Dict[int, int]:
    sec_to_label: Dict[int, int] = {}

    for _, row in df_video.iterrows():
        action = row["action"]
        start_s = int(row["start_s"])
        end_s = int(row["end_s"])

        if action not in ACTION_TO_LABEL:
            log(
                f"[WARN] {row['video']}: unknown action '{action}' "
                f"in [{start_s},{end_s}) -> skipped"
            )
            continue

        for sec in range(start_s, end_s):
            sec_to_label[sec] = ACTION_TO_LABEL[action]

    return sec_to_label


def cut_clip(video_path: str, center_s: int, label: int, out_dir: str, duration: float) -> Optional[str]:
    start_s = max(0.0, float(center_s - 1))
    end_s = start_s + float(CLIP_LEN)
    if end_s > duration:
        return None

    minute = center_s // 60
    second = center_s % 60
    action = LABEL_TO_ACTION[label]
    output_path = os.path.join(out_dir, f"{minute:02d}_{second:02d}_{action}.mp4")

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start_s:.3f}",
        "-i",
        video_path,
        "-t",
        f"{CLIP_LEN:.3f}",
        "-fflags",
        "+genpts",
        "-an",
        "-vsync",
        "cfr",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-movflags",
        "+faststart",
        "-y",
        output_path,
    ]
    subprocess.run(cmd, check=True)
    return output_path


def export_video_clips(video_id: str, video_path: str, sec_to_label: Dict[int, int], num_workers: int) -> int:
    out_dir = os.path.join(OUT_VIDEOS_DIR, video_id)
    ensure_dir(out_dir)

    duration = get_video_duration(video_path)
    centers = sorted(sec_to_label.keys())
    log(f"[STAGE] {video_id}: cutting {len(centers)} clips -> {out_dir}")

    exported = 0
    if num_workers <= 1:
        for center_s in tqdm(centers, desc=video_id, unit="clip"):
            if cut_clip(video_path, center_s, sec_to_label[center_s], out_dir, duration):
                exported += 1
        return exported

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(cut_clip, video_path, center_s, sec_to_label[center_s], out_dir, duration): center_s
            for center_s in centers
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc=video_id, unit="clip"):
            if future.result():
                exported += 1

    return exported


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cut 3-second action clips into ./videos/videoXX/."
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="number of parallel ffmpeg workers",
    )
    args = parser.parse_args()

    if args.num_workers < 1:
        raise ValueError("--num_workers must be >= 1")
    if not have_ffmpeg():
        raise RuntimeError("ffmpeg was not found. Please install ffmpeg and make it available in PATH.")

    df = load_metadata(META_CSV)
    video_file_index = build_video_file_index(VIDEOS_DIR)
    ensure_dir(OUT_VIDEOS_DIR)

    total_exported = 0
    video_names = sorted(df["video"].unique(), key=video_id_from_name)
    log(f"[INFO] Found {len(video_names)} annotated videos in {META_CSV}")

    for video_name in video_names:
        video_id = video_id_from_name(video_name)
        video_path = resolve_video_path(video_name, video_file_index)
        if video_path is None:
            log(f"[WARN] {video_id}: missing source mp4 in {VIDEOS_DIR}; skipped")
            continue

        df_video = df[df["video"] == video_name]
        sec_to_label = build_second_level_labels(df_video)
        if not sec_to_label:
            log(f"[WARN] {video_id}: no known action labels; skipped")
            continue

        exported = export_video_clips(video_id, video_path, sec_to_label, args.num_workers)
        total_exported += exported
        log(f"[OK] {video_id}: exported {exported} clips")

    log(f"[DONE] Exported {total_exported} clips under {OUT_VIDEOS_DIR}")


if __name__ == "__main__":
    main()
