from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    import imageio.v3 as iio
except Exception:  # pragma: no cover
    iio = None


PLAYER_KEYS = ("tl", "tr", "bl", "br")
TOP_KEYS = {"tl", "tr"}
BOTTOM_KEYS = {"bl", "br"}
PLAYER_COLORS = {
    "tl": (255, 99, 71),
    "tr": (30, 144, 255),
    "bl": (50, 205, 50),
    "br": (255, 215, 0),
}
BALL_COLOR = (255, 255, 255)
SMOOTHING_ALPHA = 0.65
BALL_SMOOTHING_ALPHA = 0.5
BALL_MAX_HOLD_FRAMES = 2


@dataclass
class FrameDetections:
    frame_idx: int
    ball_center: tuple[float, float] | None
    players: dict[str, tuple[float, float] | None]


@dataclass
class ActionRow:
    start_frame: int
    end_frame: int
    position: str
    action_name: str
    score: str
    raw: dict[str, str]


def euclidean(p1: tuple[float, float] | None, p2: tuple[float, float] | None) -> float | None:
    if p1 is None or p2 is None:
        return None
    return float(math.hypot(p1[0] - p2[0], p1[1] - p2[1]))


def safe_mean(values: Iterable[float | None]) -> float:
    valid = [float(v) for v in values if v is not None and np.isfinite(v)]
    if not valid:
        return 0.0
    return float(np.mean(valid))


def read_action_csv(csv_path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    return rows, fieldnames


def write_action_csv(csv_path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def normalize_action_rows(rows: list[dict[str, str]]) -> list[ActionRow]:
    normalized: list[ActionRow] = []
    for row in rows:
        action_name = row.get("action_name", row.get("name", ""))
        normalized.append(
            ActionRow(
                start_frame=int(float(row["start_frame"])),
                end_frame=int(float(row["end_frame"])),
                position=str(row.get("position", "")).strip(),
                action_name=str(action_name).strip(),
                score=str(row.get("score", "")).strip(),
                raw=row,
            )
        )
    return normalized


def build_target_frame_indices(
    action_rows: list[ActionRow],
    frame_step: int = 1,
    max_frames: int | None = None,
    context_padding: int = 3,
) -> set[int]:
    target_frames: set[int] = set()
    for row in action_rows:
        start_frame = max(0, row.start_frame - context_padding)
        end_frame = row.end_frame + context_padding
        for frame_idx in range(start_frame, end_frame + 1, max(frame_step, 1)):
            target_frames.add(frame_idx)
            if max_frames is not None and len(target_frames) >= max_frames:
                return target_frames
    return target_frames


def fill_missing_points(points: list[tuple[float, float] | None]) -> list[tuple[float, float] | None]:
    if not points:
        return points

    filled = list(points)
    valid_indices = [idx for idx, p in enumerate(filled) if p is not None]
    if not valid_indices:
        return filled

    first_valid = valid_indices[0]
    for idx in range(0, first_valid):
        filled[idx] = filled[first_valid]

    last_valid = valid_indices[-1]
    for idx in range(last_valid + 1, len(filled)):
        filled[idx] = filled[last_valid]

    for left_idx, right_idx in zip(valid_indices, valid_indices[1:]):
        left_point = filled[left_idx]
        right_point = filled[right_idx]
        if left_point is None or right_point is None or right_idx - left_idx <= 1:
            continue
        gap = right_idx - left_idx
        for inner_idx in range(left_idx + 1, right_idx):
            ratio = (inner_idx - left_idx) / gap
            filled[inner_idx] = (
                float(left_point[0] + ratio * (right_point[0] - left_point[0])),
                float(left_point[1] + ratio * (right_point[1] - left_point[1])),
            )

    return filled


def midpoint(
    p1: tuple[float, float] | None,
    p2: tuple[float, float] | None,
    require_both: bool = False,
) -> tuple[float, float] | None:
    if require_both and (p1 is None or p2 is None):
        return None
    if p1 is None and p2 is None:
        return None
    if p1 is None:
        return p2
    if p2 is None:
        return p1
    return (float((p1[0] + p2[0]) / 2.0), float((p1[1] + p2[1]) / 2.0))


def keypoint_to_point(person_keypoints: np.ndarray, idx: int) -> tuple[float, float] | None:
    if idx < 0 or idx >= len(person_keypoints):
        return None
    point = person_keypoints[idx]
    if not np.isfinite(point).all():
        return None
    return (float(point[0]), float(point[1]))


def extract_player_anchor_from_pose(
    person_keypoints: np.ndarray,
    bbox: np.ndarray | None = None,
) -> tuple[float, float] | None:
    # COCO keypoint order: 13/14 knees, 15/16 ankles
    left_ankle = keypoint_to_point(person_keypoints, 15)
    right_ankle = keypoint_to_point(person_keypoints, 16)
    ankle_mid = midpoint(left_ankle, right_ankle)
    if ankle_mid is not None:
        return ankle_mid

    left_knee = keypoint_to_point(person_keypoints, 13)
    right_knee = keypoint_to_point(person_keypoints, 14)
    knee_mid = midpoint(left_knee, right_knee)
    if knee_mid is not None:
        return knee_mid

    if bbox is not None and len(bbox) >= 4 and np.isfinite(bbox).all():
        x1, y1, x2, y2 = bbox[:4]
        return (float((x1 + x2) / 2.0), float(y2))

    return None


def read_video_metadata(video_path: Path) -> tuple[float, int]:
    if iio is None:
        raise RuntimeError("imageio is not installed. Please install imageio to read video metadata.")

    fps = None
    nframes = -1

    try:
        meta = iio.immeta(video_path)
        if isinstance(meta, dict):
            for key in ("fps", "FPS", "framerate", "frame_rate"):
                value = meta.get(key)
                if value:
                    fps = float(value)
                    break
            if "duration" in meta and "nframes" in meta and meta["duration"]:
                try:
                    nframes = int(meta["nframes"])
                except Exception:
                    nframes = -1
    except Exception:
        meta = None

    try:
        props = iio.improps(video_path)
        if fps is None:
            for attr in ("fps", "frame_rate"):
                value = getattr(props, attr, None)
                if value:
                    fps = float(value)
                    break
        prop_nframes = getattr(props, "n_images", None)
        if prop_nframes is not None:
            nframes = int(prop_nframes)
    except Exception:
        props = None

    if fps is None:
        fps = 25.0
    return fps, nframes


def assign_players_to_quadrants(player_centers: list[tuple[float, float]]) -> dict[str, tuple[float, float] | None]:
    assigned = {k: None for k in PLAYER_KEYS}
    if not player_centers:
        return assigned

    centers = sorted(player_centers, key=lambda p: (p[1], p[0]))
    if len(centers) >= 4:
        top_two = sorted(centers[:2], key=lambda p: p[0])
        bottom_two = sorted(centers[-2:], key=lambda p: p[0])
        assigned["tl"], assigned["tr"] = top_two[0], top_two[1]
        assigned["bl"], assigned["br"] = bottom_two[0], bottom_two[1]
        return assigned

    median_y = float(np.median([p[1] for p in centers]))
    top = sorted([p for p in centers if p[1] <= median_y], key=lambda p: p[0])
    bottom = sorted([p for p in centers if p[1] > median_y], key=lambda p: p[0])

    if len(top) >= 1:
        assigned["tl"] = top[0]
    if len(top) >= 2:
        assigned["tr"] = top[1]
    if len(bottom) >= 1:
        assigned["bl"] = bottom[0]
    if len(bottom) >= 2:
        assigned["br"] = bottom[1]
    return assigned


def split_player_candidates(player_centers: list[tuple[float, float]]) -> dict[str, list[tuple[float, float]]]:
    if not player_centers:
        return {"top": [], "bottom": []}

    if len(player_centers) >= 4:
        centers = sorted(player_centers, key=lambda p: (p[1], p[0]))
        return {
            "top": centers[:2],
            "bottom": centers[-2:],
        }

    median_y = float(np.median([p[1] for p in player_centers]))
    return {
        "top": [p for p in player_centers if p[1] <= median_y],
        "bottom": [p for p in player_centers if p[1] > median_y],
    }


def smooth_point(
    previous: tuple[float, float] | None,
    current: tuple[float, float] | None,
    alpha: float = SMOOTHING_ALPHA,
) -> tuple[float, float] | None:
    if current is None:
        return previous
    if previous is None:
        return current
    return (
        float(alpha * previous[0] + (1.0 - alpha) * current[0]),
        float(alpha * previous[1] + (1.0 - alpha) * current[1]),
    )


def smooth_ball_center(
    previous: tuple[float, float] | None,
    current: tuple[float, float] | None,
    alpha: float = BALL_SMOOTHING_ALPHA,
) -> tuple[float, float] | None:
    return smooth_point(previous, current, alpha=alpha)


def assign_side_with_tracking(
    candidates: list[tuple[float, float]],
    labels: tuple[str, str],
    prev_players: dict[str, tuple[float, float] | None] | None,
) -> dict[str, tuple[float, float] | None]:
    result = {labels[0]: None, labels[1]: None}
    if not candidates:
        return result

    remaining = list(candidates)
    if prev_players is None or all(prev_players.get(label) is None for label in labels):
        ordered = sorted(remaining, key=lambda p: p[0])
        if len(ordered) >= 1:
            result[labels[0]] = ordered[0]
        if len(ordered) >= 2:
            result[labels[1]] = ordered[1]
        return result

    for label in labels:
        prev_point = prev_players.get(label)
        if prev_point is None or not remaining:
            continue
        best_idx = min(range(len(remaining)), key=lambda idx: euclidean(prev_point, remaining[idx]) or float("inf"))
        result[label] = remaining.pop(best_idx)

    if remaining:
        unassigned = [label for label in labels if result[label] is None]
        ordered = sorted(remaining, key=lambda p: p[0])
        for label, candidate in zip(unassigned, ordered):
            result[label] = candidate
    return result


def assign_players_with_tracking(
    player_centers: list[tuple[float, float]],
    prev_players: dict[str, tuple[float, float] | None] | None,
) -> dict[str, tuple[float, float] | None]:
    raw_quadrant = assign_players_to_quadrants(player_centers)
    if prev_players is None:
        return raw_quadrant

    split = split_player_candidates(player_centers)
    tracked_top = assign_side_with_tracking(split["top"], ("tl", "tr"), prev_players)
    tracked_bottom = assign_side_with_tracking(split["bottom"], ("bl", "br"), prev_players)

    tracked = {**tracked_top, **tracked_bottom}
    final_players: dict[str, tuple[float, float] | None] = {}
    for key in PLAYER_KEYS:
        candidate = tracked.get(key)
        if candidate is None:
            candidate = raw_quadrant.get(key)
        final_players[key] = smooth_point(prev_players.get(key), candidate)
    return final_players


def detect_from_video(
    video_path: Path,
    ball_model_path: str,
    pose_model_path: str,
    target_frame_indices: set[int] | None = None,
    frame_step: int = 1,
    max_frames: int | None = None,
    progress_interval: int = 100,
    ball_imgsz: int = 640,
    pose_imgsz: int = 640,
    device: str | None = None,
    half: bool = False,
) -> list[FrameDetections]:
    try:
        from ultralytics import YOLO
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "ultralytics is required for direct video detection. "
            "Install it with `pip install ultralytics opencv-python imageio`."
        ) from exc

    if iio is None:
        raise RuntimeError("imageio is required for frame iteration. Install it with `pip install imageio`.")

    ball_model = YOLO(ball_model_path)
    pose_model = YOLO(pose_model_path)

    detections: list[FrameDetections] = []
    prev_players: dict[str, tuple[float, float] | None] | None = None
    prev_ball_center: tuple[float, float] | None = None
    missing_ball_frames = 0
    total_targets = len(target_frame_indices) if target_frame_indices is not None else None
    start_time = time.time()
    print(
        f"Starting detection: video={video_path.name}, "
        f"frame_step={frame_step}, max_frames={max_frames}, "
        f"target_frames={'all' if target_frame_indices is None else len(target_frame_indices)}, "
        f"ball_model={ball_model_path}, pose_model={pose_model_path}, "
        f"ball_imgsz={ball_imgsz}, pose_imgsz={pose_imgsz}, device={device or 'auto'}, half={half}"
    )
    for frame_idx, frame in enumerate(iio.imiter(video_path)):
        if target_frame_indices is not None:
            if frame_idx not in target_frame_indices:
                continue
        elif frame_idx % frame_step != 0:
            continue
        if max_frames is not None and len(detections) >= max_frames:
            break
        if progress_interval > 0 and len(detections) % progress_interval == 0:
            elapsed = max(time.time() - start_time, 1e-6)
            processed = len(detections)
            rate = processed / elapsed
            if total_targets:
                remaining = max(total_targets - processed, 0)
                eta = remaining / max(rate, 1e-6)
                print(
                    f"[detect] processed={processed}/{total_targets} "
                    f"current_frame_idx={frame_idx} fps={rate:.2f} eta={eta/60:.1f}m"
                )
            else:
                print(f"[detect] processed={processed} current_frame_idx={frame_idx} fps={rate:.2f}")

        ball_result = ball_model.predict(frame, verbose=False, imgsz=ball_imgsz, device=device, half=half)[0]
        pose_result = pose_model.predict(frame, verbose=False, imgsz=pose_imgsz, device=device, half=half)[0]

        ball_center = None
        if getattr(ball_result, "boxes", None) is not None and len(ball_result.boxes) > 0:
            boxes = ball_result.boxes.xyxy.cpu().numpy()
            confs = ball_result.boxes.conf.cpu().numpy()
            best_idx = int(np.argmax(confs))
            x1, y1, x2, y2 = boxes[best_idx]
            ball_center = (float((x1 + x2) / 2.0), float((y1 + y2) / 2.0))
            ball_center = smooth_ball_center(prev_ball_center, ball_center)
            prev_ball_center = ball_center
            missing_ball_frames = 0
        else:
            missing_ball_frames += 1
            if prev_ball_center is not None and missing_ball_frames <= BALL_MAX_HOLD_FRAMES:
                ball_center = prev_ball_center
            else:
                prev_ball_center = None

        player_centers: list[tuple[float, float]] = []
        if getattr(pose_result, "keypoints", None) is not None and pose_result.keypoints.xy is not None:
            keypoints = pose_result.keypoints.xy.cpu().numpy()
            for person in keypoints:
                valid = person[np.isfinite(person).all(axis=1)]
                if len(valid) == 0:
                    continue
                cx = float(np.mean(valid[:, 0]))
                cy = float(np.mean(valid[:, 1]))
                player_centers.append((cx, cy))
        elif getattr(pose_result, "boxes", None) is not None and len(pose_result.boxes) > 0:
            for box in pose_result.boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = box
                player_centers.append((float((x1 + x2) / 2.0), float((y1 + y2) / 2.0)))

        current_players = assign_players_with_tracking(player_centers, prev_players)
        detections.append(FrameDetections(frame_idx=frame_idx, ball_center=ball_center, players=current_players))
        prev_players = current_players
    elapsed = max(time.time() - start_time, 1e-6)
    rate = len(detections) / elapsed
    print(f"Detection finished: extracted_frames={len(detections)} avg_fps={rate:.2f} elapsed={elapsed/60:.1f}m")
    return detections


def save_detections(detections: list[FrameDetections], output_path: Path) -> None:
    payload = []
    for det in detections:
        payload.append(
            {
                "frame_idx": det.frame_idx,
                "ball_center": list(det.ball_center) if det.ball_center is not None else None,
                "players": {k: list(v) if v is not None else None for k, v in det.players.items()},
            }
        )
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_detections(path: Path) -> list[FrameDetections]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for item in payload:
        out.append(
            FrameDetections(
                frame_idx=int(item["frame_idx"]),
                ball_center=tuple(item["ball_center"]) if item["ball_center"] is not None else None,
                players={k: tuple(v) if v is not None else None for k, v in item["players"].items()},
            )
        )
    return out


def draw_marker(draw: ImageDraw.ImageDraw, center: tuple[float, float], color: tuple[int, int, int], radius: int) -> None:
    x, y = center
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=3)
    draw.line((x - radius, y, x + radius, y), fill=color, width=2)
    draw.line((x, y - radius, x, y + radius), fill=color, width=2)


def draw_highlight_ring(
    draw: ImageDraw.ImageDraw,
    center: tuple[float, float],
    color: tuple[int, int, int],
    radius: int,
    width: int = 5,
) -> None:
    x, y = center
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=width)


def draw_label(
    draw: ImageDraw.ImageDraw,
    position: tuple[float, float],
    text: str,
    color: tuple[int, int, int],
    font: ImageFont.ImageFont,
) -> None:
    x, y = position
    left, top, right, bottom = draw.textbbox((x, y), text, font=font)
    draw.rounded_rectangle((left - 4, top - 2, right + 4, bottom + 2), radius=4, fill=color)
    draw.text((x, y), text, font=font, fill=(0, 0, 0))


def render_detection_video(
    video_path: Path,
    detections: list[FrameDetections],
    output_path: Path,
    fps: float,
    action_rows: list[ActionRow] | None = None,
    target_frame_indices: set[int] | None = None,
    frame_step: int = 1,
    max_frames: int | None = None,
    progress_interval: int = 100,
) -> None:
    if iio is None:
        raise RuntimeError("imageio is required for visualization export. Install it with `pip install imageio`.")

    det_map = {det.frame_idx: det for det in detections}
    font = ImageFont.load_default()
    writer_fps = fps / max(frame_step, 1)
    rendered_frames: list[Image.Image] = []
    rendered_arrays: list[np.ndarray] = []
    action_segments: list[ActionRow] = []
    active_action_idx = 0
    if action_rows is not None:
        action_segments = sorted(action_rows, key=lambda row: (row.start_frame, row.end_frame))

    total_targets = len(target_frame_indices) if target_frame_indices is not None else None
    start_time = time.time()
    print(
        f"Starting visualization export: output={output_path.name}, "
        f"frame_step={frame_step}, max_frames={max_frames}, "
        f"target_frames={'all' if target_frame_indices is None else len(target_frame_indices)}"
    )
    rendered = 0
    for frame_idx, frame in enumerate(iio.imiter(video_path)):
        if target_frame_indices is not None:
            if frame_idx not in target_frame_indices:
                continue
        elif frame_idx % frame_step != 0:
            continue
        if max_frames is not None and rendered >= max_frames:
            break
        if progress_interval > 0 and rendered % progress_interval == 0:
            elapsed = max(time.time() - start_time, 1e-6)
            rate = rendered / elapsed
            if total_targets:
                remaining = max(total_targets - rendered, 0)
                eta = remaining / max(rate, 1e-6)
                print(
                    f"[viz] rendered={rendered}/{total_targets} "
                    f"current_frame_idx={frame_idx} fps={rate:.2f} eta={eta/60:.1f}m"
                )
            else:
                print(f"[viz] rendered={rendered} current_frame_idx={frame_idx} fps={rate:.2f}")

        image = Image.fromarray(frame)
        draw = ImageDraw.Draw(image)
        det = det_map.get(frame_idx)
        while active_action_idx < len(action_segments) and action_segments[active_action_idx].end_frame < frame_idx:
            active_action_idx += 1

        active_action = None
        active_position = None
        if active_action_idx < len(action_segments):
            candidate = action_segments[active_action_idx]
            if candidate.start_frame <= frame_idx <= candidate.end_frame:
                active_action = candidate
                active_position = candidate.position

        if det is not None:
            if det.ball_center is not None:
                draw_marker(draw, det.ball_center, BALL_COLOR, radius=8)
                draw_label(draw, (det.ball_center[0] + 10, det.ball_center[1] - 18), "ball", BALL_COLOR, font)

            for key in PLAYER_KEYS:
                center = det.players.get(key)
                if center is None:
                    continue
                color = PLAYER_COLORS[key]
                if active_position == key:
                    draw_highlight_ring(draw, center, (255, 255, 255), radius=18, width=6)
                draw_marker(draw, center, color, radius=12)
                draw_label(draw, (center[0] + 10, center[1] - 18), key, color, font)

        draw_label(draw, (12, 12), f"frame={frame_idx}", (220, 220, 220), font)
        if active_action is not None:
            action_name = active_action.action_name
            position = active_action.position
            start_frame = active_action.start_frame
            end_frame = active_action.end_frame
            score = active_action.score
            draw_label(draw, (12, 34), f"action={action_name}", (255, 182, 193), font)
            draw_label(draw, (12, 56), f"position={position}  score={score}", (173, 216, 230), font)
            draw_label(draw, (12, 78), f"segment={start_frame}-{end_frame}", (176, 224, 230), font)

        if output_path.suffix.lower() == ".gif":
            rendered_frames.append(image.convert("P", palette=Image.ADAPTIVE))
        else:
            rendered_arrays.append(np.asarray(image))
        rendered += 1

    if output_path.suffix.lower() == ".gif":
        if not rendered_frames:
            raise RuntimeError("No frames were rendered for GIF export.")
        duration_ms = max(int(1000 / max(writer_fps, 1e-6)), 1)
        rendered_frames[0].save(
            output_path,
            save_all=True,
            append_images=rendered_frames[1:],
            loop=0,
            duration=duration_ms,
        )
        elapsed = max(time.time() - start_time, 1e-6)
        rate = rendered / elapsed
        print(f"Visualization finished: rendered_frames={rendered} avg_fps={rate:.2f} elapsed={elapsed/60:.1f}m")
        return

    try:
        writer = iio.imopen(output_path, "w", plugin="pyav")
    except Exception as exc:
        raise RuntimeError(
            "MP4 export requires a video backend. Install one with "
            "`pip install av` or `pip install imageio[ffmpeg]`, "
            "or export to a `.gif` file instead."
        ) from exc

    with writer:
        writer.init_video_stream("libx264", fps=writer_fps)
        for frame_array in rendered_arrays:
            writer.write_frame(frame_array)
    elapsed = max(time.time() - start_time, 1e-6)
    rate = rendered / elapsed
    print(f"Visualization finished: rendered_frames={rendered} avg_fps={rate:.2f} elapsed={elapsed/60:.1f}m")


def compute_window_velocity(
    points: list[tuple[float, float] | None],
    fps: float,
    frame_radius: int = 2,
) -> list[float | None]:
    if fps <= 0:
        return [None] * len(points)

    radius = max(int(frame_radius), 1)
    velocities: list[float | None] = []
    for idx in range(len(points)):
        left_idx = max(0, idx - radius)
        right_idx = min(len(points) - 1, idx + radius)
        if left_idx == right_idx:
            velocities.append(None)
            continue
        dt = (right_idx - left_idx) * (1.0 / fps)
        dist = euclidean(points[right_idx], points[left_idx])
        velocities.append(None if dist is None or dt <= 0 else dist / dt)
    return velocities


def quadratic_or_linear_landing(ball_points: list[tuple[float, float] | None], window: int) -> tuple[float, float] | None:
    valid = [(idx, p) for idx, p in enumerate(ball_points[-window:]) if p is not None]
    if len(valid) < 2:
        return None

    times = np.array([v[0] for v in valid], dtype=np.float64)
    xs = np.array([v[1][0] for v in valid], dtype=np.float64)
    ys = np.array([v[1][1] for v in valid], dtype=np.float64)
    target_t = float(window)

    try:
        deg = 2 if len(valid) >= 3 else 1
        coef_x = np.polyfit(times, xs, deg=deg)
        coef_y = np.polyfit(times, ys, deg=deg)
        pred_x = float(np.polyval(coef_x, target_t))
        pred_y = float(np.polyval(coef_y, target_t))
        if np.isfinite(pred_x) and np.isfinite(pred_y):
            return pred_x, pred_y
    except Exception:
        pass

    last_two = [p for p in ball_points if p is not None][-2:]
    if len(last_two) < 2:
        return None
    dx = last_two[-1][0] - last_two[-2][0]
    dy = last_two[-1][1] - last_two[-2][1]
    return (last_two[-1][0] + 1.5 * dx, last_two[-1][1] + 1.5 * dy)


def get_opponent_keys(actor_pos: str) -> tuple[str, str]:
    if actor_pos in TOP_KEYS:
        return ("bl", "br")
    return ("tl", "tr")


def compute_action_features_for_row(
    row: ActionRow,
    frame_map: dict[int, FrameDetections],
    fps: float,
    window: int,
) -> dict[str, float]:
    frame_indices = list(range(row.start_frame, row.end_frame + 1))
    frames = [frame_map.get(idx) for idx in frame_indices]

    ball_points_raw = [f.ball_center if f is not None else None for f in frames]
    actor_points_raw = [f.players.get(row.position) if f is not None else None for f in frames]
    # Interpolate sparse trajectories before feature computation so speed is more stable.
    ball_points = fill_missing_points(ball_points_raw)
    actor_points = fill_missing_points(actor_points_raw)

    ball_velocities = compute_window_velocity(ball_points, fps=fps, frame_radius=2)
    player_velocities = compute_window_velocity(actor_points, fps=fps, frame_radius=2)

    pdistance = euclidean(actor_points[0], actor_points[-1]) if actor_points else None

    landing = quadratic_or_linear_landing(ball_points, window=window)
    target_distance = 0.0
    if landing is not None and frames:
        opponent_keys = get_opponent_keys(row.position)
        opponent_centers_raw = []
        for key in opponent_keys:
            opponent_centers_raw.append(
                [f.players.get(key) if f is not None else None for f in frames]
            )
        opponent_centers = []
        for seq in opponent_centers_raw:
            filled_seq = fill_missing_points(seq)
            opponent_centers.append(filled_seq[-1] if filled_seq else None)
        opponent_dists = [euclidean(landing, center) for center in opponent_centers]
        opponent_dists = [d for d in opponent_dists if d is not None]
        if opponent_dists:
            target_distance = float(min(opponent_dists))

    return {
        "target_distance": float(target_distance),
        "pdistance": float(pdistance or 0.0),
        "pspeed": safe_mean(player_velocities),
        "bspeed": safe_mean(ball_velocities),
    }


def compute_features_for_csv(
    csv_path: Path,
    detections: list[FrameDetections],
    fps: float,
    window: int,
    output_csv: Path,
) -> list[dict[str, str]]:
    rows, fieldnames = read_action_csv(csv_path)
    normalized_rows = normalize_action_rows(rows)
    frame_map = {det.frame_idx: det for det in detections}

    for col in ["target_distance", "pdistance", "pspeed", "bspeed"]:
        if col not in fieldnames:
            fieldnames.append(col)

    for action_row in normalized_rows:
        feats = compute_action_features_for_row(action_row, frame_map=frame_map, fps=fps, window=window)
        for key, value in feats.items():
            action_row.raw[key] = f"{value:.6f}"

    write_action_csv(output_csv, rows, fieldnames)
    return rows


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract action-level features from sports video.")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--detections-json", type=Path, default=None)
    parser.add_argument("--ball-model", type=str, default="yolo11n.pt")
    parser.add_argument("--pose-model", type=str, default="yolo11n-pose.pt")
    parser.add_argument("--window", type=int, default=3)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--compute-only", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--visualization-output", type=Path, default=None)
    parser.add_argument("--progress-interval", type=int, default=100)
    parser.add_argument("--ball-imgsz", type=int, default=640)
    parser.add_argument("--pose-imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--context-padding", type=int, default=3)
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    output_csv = args.output_csv or args.csv.with_name(f"{args.csv.stem}_features.csv")
    detections_json = args.detections_json or args.csv.with_name(f"{args.csv.stem}_detections.json")
    visualization_output = args.visualization_output or args.csv.with_name(f"{args.csv.stem}_viz.mp4")
    raw_rows, _ = read_action_csv(args.csv)
    action_rows = normalize_action_rows(raw_rows)
    target_frame_indices = build_target_frame_indices(
        action_rows=action_rows,
        frame_step=args.frame_step,
        max_frames=args.max_frames,
        context_padding=args.context_padding,
    )

    fps = args.fps
    if fps is None:
        fps, _ = read_video_metadata(args.video)

    if args.compute_only:
        detections = load_detections(detections_json)
    else:
        detections = detect_from_video(
            video_path=args.video,
            ball_model_path=args.ball_model,
            pose_model_path=args.pose_model,
            target_frame_indices=target_frame_indices,
            frame_step=args.frame_step,
            max_frames=args.max_frames,
            progress_interval=args.progress_interval,
            ball_imgsz=args.ball_imgsz,
            pose_imgsz=args.pose_imgsz,
            device=args.device,
            half=args.half,
        )
        save_detections(detections, detections_json)

    rows = compute_features_for_csv(
        csv_path=args.csv,
        detections=detections,
        fps=fps,
        window=args.window,
        output_csv=output_csv,
    )

    if args.visualize:
        render_detection_video(
            video_path=args.video,
            detections=detections,
            output_path=visualization_output,
            fps=fps,
            action_rows=action_rows,
            target_frame_indices=target_frame_indices,
            frame_step=args.frame_step,
            max_frames=args.max_frames,
            progress_interval=args.progress_interval,
        )

    print(f"Saved features to: {output_csv}")
    print(f"Saved detections to: {detections_json}")
    if args.visualize:
        print(f"Saved visualization video to: {visualization_output}")
    preview_rows = rows[:5]
    if preview_rows:
        preview_keys = list(preview_rows[0].keys())
        for row in preview_rows:
            print({key: row.get(key, "") for key in preview_keys})


if __name__ == "__main__":
    main()
