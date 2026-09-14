import argparse
import os
import sys
import time
from collections import defaultdict, deque
from typing import Dict, List, Tuple
import configparser

import cv2
import numpy as np
from ultralytics import YOLO

from cctv_stream import CCTVStream

# Target COCO Classes
# 0: Person (Pedestrian)
# 2: Car, 3: Motorcycle, 5: Bus, 7: Truck
PEDESTRIAN_CLASSES = {0: "Pedestrian"}
VEHICLE_CLASSES = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}
ALL_TARGET_CLASSES = {**PEDESTRIAN_CLASSES, **VEHICLE_CLASSES}

# Visual Color Palette (BGR)
COLOR_PEDESTRIAN = (255, 200, 0)     # Bright Cyan / Aqua
COLOR_VEHICLE = (50, 220, 50)        # Vibrant Green
COLOR_STOPPED = (80, 120, 255)       # Coral / Orange for stationary
COLOR_HUD_BG = (20, 24, 28)          # Dark Slate Gray
COLOR_HUD_CARD = (32, 38, 46)        # Card Background
COLOR_TEXT_WHITE = (245, 245, 245)
COLOR_TEXT_MUTED = (160, 175, 190)
COLOR_ACCENT = (0, 210, 255)         # Bright Gold / Yellow


def calculate_motion_stats(
    history: deque,
    window: int = 5,
) -> Tuple[Tuple[float, float], float, str]:
    """
    Computes:
    - (vx, vy): smoothed velocity vector in px/frame
    - speed: magnitude of velocity in px/frame
    - heading: 8-point cardinal compass direction (N, NE, E, SE, S, SW, W, NW)
    """
    if len(history) < 2:
        return (0.0, 0.0), 0.0, "--"

    pts = list(history)
    w_pts = pts[-min(len(pts), window):]
    dt = len(w_pts) - 1

    if dt == 0:
        return (0.0, 0.0), 0.0, "--"

    vx = (w_pts[-1][0] - w_pts[0][0]) / dt
    vy = (w_pts[-1][1] - w_pts[0][1]) / dt
    speed = float(np.hypot(vx, vy))

    if speed < 0.6:
        return (vx, vy), speed, "Stopped"

    # Compute angle in degrees (0 = East, 90 = South in image coordinates)
    angle = np.degrees(np.arctan2(vy, vx)) % 360
    directions = ["E", "SE", "S", "SW", "W", "NW", "N", "NE"]
    idx = int((angle + 22.5) // 45) % 8
    heading = directions[idx]

    return (vx, vy), speed, heading


def draw_translucent_panel(
    frame: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    bg_color: Tuple[int, int, int] = COLOR_HUD_BG,
    alpha: float = 0.75,
    border_color: Tuple[int, int, int] = (60, 70, 80),
) -> None:
    """Draws an anti-aliased semi-transparent rectangle overlay on the frame."""
    fh, fw = frame.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(fw, x + w), min(fh, y + h)

    if x2 <= x1 or y2 <= y1:
        return

    sub_img = frame[y1:y2, x1:x2]
    rect = np.full_like(sub_img, bg_color, dtype=np.uint8)
    cv2.addWeighted(rect, alpha, sub_img, 1 - alpha, 0, sub_img)

    if border_color is not None:
        cv2.rectangle(frame, (x1, y1), (x2 - 1, y2 - 1), border_color, 1)


class IntersectionAnalytics:
    """Tracks historical metrics, cumulative counts, and intersection statistics."""

    def __init__(self):
        self.seen_vehicles = set()
        self.seen_pedestrians = set()
        self.speed_history = deque(maxlen=60)
        self.fps_history = deque(maxlen=30)
        self.start_time = time.time()

    def update(self, active_vehicles: List[dict], active_pedestrians: List[dict], fps: float):
        for v in active_vehicles:
            self.seen_vehicles.add(v["id"])
            if v["speed"] >= 0.6:
                self.speed_history.append(v["speed"])

        for p in active_pedestrians:
            self.seen_pedestrians.add(p["id"])

        if fps > 0:
            self.fps_history.append(fps)

    @property
    def total_vehicles(self) -> int:
        return len(self.seen_vehicles)

    @property
    def total_pedestrians(self) -> int:
        return len(self.seen_pedestrians)

    @property
    def average_vehicle_speed(self) -> float:
        return float(np.mean(self.speed_history)) if self.speed_history else 0.0

    @property
    def smoothed_fps(self) -> float:
        return float(np.mean(self.fps_history)) if self.fps_history else 0.0

    def get_traffic_density_label(self, active_veh_count: int) -> Tuple[str, Tuple[int, int, int]]:
        if active_veh_count == 0:
            return "Clear", (100, 200, 100)
        elif active_veh_count <= 4:
            return "Light", (80, 220, 100)
        elif active_veh_count <= 8:
            return "Moderate", (0, 215, 255)
        else:
            return "Heavy", (50, 50, 240)

    def reset(self):
        self.seen_vehicles.clear()
        self.seen_pedestrians.clear()
        self.speed_history.clear()
        self.start_time = time.time()


def render_dashboard(
    frame: np.ndarray,
    stream: CCTVStream,
    analytics: IntersectionAnalytics,
    active_vehicles: List[dict],
    active_pedestrians: List[dict],
    show_trails: bool = True,
    show_hud: bool = True,
) -> np.ndarray:
    """Renders visual overlays, tracking trajectories, and dashboard analytics HUD."""
    annotated = frame.copy()
    h, w = frame.shape[:2]

    # 1. Render Object Trails and Bounding Boxes
    all_objects = [("vehicle", v) for v in active_vehicles] + [("pedestrian", p) for p in active_pedestrians]

    for obj_type, obj in all_objects:
        bx1, by1, bx2, by2 = map(int, obj["box"])
        tid = obj["id"]
        cls_name = obj["class"]
        speed = obj["speed"]
        heading = obj["heading"]
        history = obj["history"]

        # Determine theme color
        if obj_type == "pedestrian":
            primary_color = COLOR_PEDESTRIAN
        else:
            primary_color = COLOR_STOPPED if speed < 0.6 else COLOR_VEHICLE

        # Draw trajectory trail
        if show_trails and len(history) > 1:
            pts = np.array(history, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated, [pts], isClosed=False, color=primary_color, thickness=2, lineType=cv2.LINE_AA)

        # Draw bounding box
        cv2.rectangle(annotated, (bx1, by1), (bx2, by2), primary_color, 2)

        # Draw label tag with background pill (in px/frame)
        label_text = f"#{tid} {cls_name} | {speed:.1f} px/f {heading}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.40
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(label_text, font, scale, thickness)

        pill_y1 = max(0, by1 - th - 6)
        pill_y2 = by1
        pill_x1 = bx1
        pill_x2 = min(w, bx1 + tw + 8)

        cv2.rectangle(annotated, (pill_x1, pill_y1), (pill_x2, pill_y2), primary_color, -1)
        cv2.putText(
            annotated,
            label_text,
            (pill_x1 + 4, pill_y2 - 4),
            font,
            scale,
            (10, 15, 20),
            thickness,
            lineType=cv2.LINE_AA,
        )

    if not show_hud:
        return annotated

    # 2. Top Header Bar
    bar_height = 36
    draw_translucent_panel(annotated, 0, 0, w, bar_height, bg_color=COLOR_HUD_BG, alpha=0.85, border_color=(50, 60, 70))

    # Stream title & status
    buf_info = f" | Buffer: {stream.buffer_level}f" if stream.is_network_stream else ""
    fps_text = f"{analytics.smoothed_fps:.1f} FPS"
    header_title = f"INTERSECTION ANALYTICS  •  {stream.display_name}{buf_info}"

    cv2.putText(annotated, header_title, (14, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.48, COLOR_TEXT_WHITE, 1, lineType=cv2.LINE_AA)
    cv2.putText(annotated, fps_text, (w - 90, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.48, COLOR_ACCENT, 1, lineType=cv2.LINE_AA)

    # 3. Analytics Widget Panel (Top Right Overlay)
    panel_w = 260
    panel_h = 160
    panel_x = w - panel_w - 14
    panel_y = bar_height + 12

    draw_translucent_panel(annotated, panel_x, panel_y, panel_w, panel_h, bg_color=COLOR_HUD_BG, alpha=0.82)

    # Panel Title
    cv2.putText(
        annotated,
        "METRICS & TRAFFIC FLOW",
        (panel_x + 12, panel_y + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        COLOR_ACCENT,
        1,
        lineType=cv2.LINE_AA,
    )
    cv2.line(annotated, (panel_x + 12, panel_y + 26), (panel_x + panel_w - 12, panel_y + 26), (50, 60, 70), 1)

    # Metrics rows (Displaying px/f)
    density_label, density_color = analytics.get_traffic_density_label(len(active_vehicles))

    metrics = [
        ("Active Vehicles", f"{len(active_vehicles)} (Total: {analytics.total_vehicles})", COLOR_VEHICLE),
        ("Active Pedestrians", f"{len(active_pedestrians)} (Total: {analytics.total_pedestrians})", COLOR_PEDESTRIAN),
        ("Avg Vehicle Speed", f"{analytics.average_vehicle_speed:.1f} px/f", COLOR_TEXT_WHITE),
        ("Traffic Density", density_label, density_color),
    ]

    curr_y = panel_y + 48
    for label, val, val_color in metrics:
        cv2.putText(annotated, label, (panel_x + 14, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_TEXT_MUTED, 1, lineType=cv2.LINE_AA)
        cv2.putText(annotated, val, (panel_x + 130, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, val_color, 1, lineType=cv2.LINE_AA)
        curr_y += 24

    # 4. Bottom Controls Bar
    ctrl_y = h - 22
    draw_translucent_panel(annotated, 0, h - 28, w, 28, bg_color=COLOR_HUD_BG, alpha=0.75, border_color=(40, 50, 60))
    controls_text = "[SPACE] Pause/Resume   [H] Toggle HUD   [T] Toggle Trails   [R] Reset Counters   [Q] Quit"
    cv2.putText(annotated, controls_text, (14, ctrl_y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.36, COLOR_TEXT_MUTED, 1, lineType=cv2.LINE_AA)

    return annotated


def main():
    config = configparser.ConfigParser()
    config.read('settings.ini')

    print(f"[*] Initializing stream from: {config['gui']['source']}")
    stream = CCTVStream(config['gui']['source'])

    print(f"[*] Loading YOLO tracking model: {config['model']['model']}")
    model = YOLO(config['model']['model'])

    ret, frame = stream.read()
    if not ret or frame is None:
        print("[!] Error: Failed to retrieve video stream.")
        stream.release()
        sys.exit(1)

    height, width = frame.shape[:2]
    display_scale = config.getfloat('gui', 'display-scale')
    disp_w, disp_h = int(width * display_scale), int(height * display_scale)

    # Tracking trajectories: {track_id: deque([(x, y), ...])}
    track_history = defaultdict(lambda: deque(maxlen=30))
    analytics = IntersectionAnalytics()

    writer = None
    if config['model']['save']:
        os.makedirs(os.path.dirname(os.path.abspath(config['model']['save'])), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(config['model']['save'], fourcc, stream.target_fps, (disp_w, disp_h))
        print(f"[+] Recording output video to: {config['model']['save']}")

    show_trails = config.getboolean('gui', 'show-trails')
    show_hud = config.getboolean('gui', 'show-hud')
    paused = False

    target_class_ids = list(ALL_TARGET_CLASSES.keys())

    print(f"[+] Monitoring {stream.display_name} @ {stream.target_fps:.1f} FPS")
    print("[+] Targets: Cars/Vehicles & Pedestrians (Speed Unit: px/frame)")
    print("[+] Controls: 'SPACE' = Pause, 'h' = Toggle HUD, 't' = Toggle Trails, 'r' = Reset, 'q' = Quit\n")

    fps_timer = time.time()
    current_fps = stream.target_fps

    while True:
        loop_start = time.time()

        if not paused:
            ret, frame = stream.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            if display_scale != 1:
                frame = cv2.resize(frame, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)

            # YOLO multi-object tracking (ByteTrack)
            results = model.track(
                frame,
                persist=True,
                classes=target_class_ids,
                conf=config.getfloat('model', 'confidence'),
                tracker="bytetrack.yaml",
                device=config.getint('model', 'device'),
                verbose=False,
            )[0]

            active_vehicles = []
            active_pedestrians = []
            current_active_ids = set()

            if results.boxes is not None and results.boxes.id is not None:
                boxes = results.boxes.xyxy.cpu().numpy()
                track_ids = results.boxes.id.int().cpu().numpy()
                classes = results.boxes.cls.int().cpu().numpy()

                for box, track_id, cls_id in zip(boxes, track_ids, classes):
                    cls_id = int(cls_id)
                    current_active_ids.add(track_id)
                    x1, y1, x2, y2 = box
                    center_pt = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

                    track_history[track_id].append(center_pt)
                    velocity, speed, heading = calculate_motion_stats(track_history[track_id])

                    cls_label = ALL_TARGET_CLASSES.get(cls_id, model.names.get(cls_id, "object"))

                    obj_data = {
                        "id": track_id,
                        "box": box,
                        "class": cls_label,
                        "center": center_pt,
                        "velocity": velocity,
                        "speed": speed,
                        "heading": heading,
                        "history": list(track_history[track_id]),
                    }

                    if cls_id in PEDESTRIAN_CLASSES:
                        active_pedestrians.append(obj_data)
                    else:
                        active_vehicles.append(obj_data)

            # Clean up stale track histories not seen for a long time
            for tid in list(track_history.keys()):
                if tid not in current_active_ids and len(track_history[tid]) > 0:
                    track_history[tid].popleft()
                    if len(track_history[tid]) == 0:
                        del track_history[tid]

            # Update FPS calculation
            now = time.time()
            fps_delta = now - fps_timer
            if fps_delta >= 0.2:
                current_fps = 1.0 / (loop_start - fps_timer if fps_timer != loop_start else 0.033)
                fps_timer = now

            analytics.update(active_vehicles, active_pedestrians, current_fps)

            # Render Dashboard
            annotated_frame = render_dashboard(
                frame=frame,
                stream=stream,
                analytics=analytics,
                active_vehicles=active_vehicles,
                active_pedestrians=active_pedestrians,
                show_trails=show_trails,
                show_hud=show_hud,
            )

            if writer:
                writer.write(annotated_frame)

            cv2.imshow("Intersection Analytics Dashboard", annotated_frame)

        elapsed = time.time() - loop_start
        sleep_delay = stream.get_sleep_interval(elapsed)

        key = cv2.waitKey(sleep_delay if not paused else 30) & 0xFF
        if key == ord("q"):
            break
        elif key == 32:  # Spacebar
            paused = not paused
        elif key == ord("h") or key == ord("H"):
            show_hud = not show_hud
        elif key == ord("t") or key == ord("T"):
            show_trails = not show_trails
        elif key == ord("r") or key == ord("R"):
            analytics.reset()
            print("[+] Intersection counters reset.")

    stream.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print("[+] Dashboard session ended.")


if __name__ == "__main__":
    main()
