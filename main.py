import argparse
import configparser
import json
import os
import sys
import time
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from calculate_motion_stats import calculate_vehicle_speed
from calibrate import (
    auto_calibrate,
    calibrate,
    update_master_trajectory_store,
)
from cctv_stream import CCTVStream

# Target COCO Classes
VEHICLE_CLASSES = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}
ALL_TARGET_CLASSES = {**VEHICLE_CLASSES}

# Visual Color Palette (BGR)
COLOR_VEHICLE = (50, 220, 50)
COLOR_STOPPED = (80, 120, 255)
COLOR_HUD_BG = (20, 24, 28)
COLOR_HUD_CARD = (32, 38, 46)
COLOR_TEXT_WHITE = (245, 245, 245)
COLOR_TEXT_MUTED = (160, 175, 190)
COLOR_ACCENT = (0, 210, 255)
COLOR_BEV_UP = (255, 0, 0)
COLOR_BEV_DOWN = (0, 165, 255)


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

    def __init__(self):
        self.seen_vehicles = set()
        self.speed_history = deque(maxlen=60)
        self.fps_history = deque(maxlen=30)
        self.start_time = time.time()

    def update(self, active_vehicles: List[dict], fps: float):
        for v in active_vehicles:
            self.seen_vehicles.add(v["id"])
            if v["speed"] is not None and v["speed"] >= 0.6:
                self.speed_history.append(v["speed"])

        if fps > 0:
            self.fps_history.append(fps)

    @property
    def total_vehicles(self) -> int:
        return len(self.seen_vehicles)

    @property
    def average_vehicle_speed(self) -> float:
        return (
            float(np.mean(self.speed_history)) if self.speed_history else 0.0
        )

    @property
    def smoothed_fps(self) -> float:
        return float(np.mean(self.fps_history)) if self.fps_history else 0.0

    def get_traffic_density_label(
        self, active_veh_count: int
    ) -> Tuple[str, Tuple[int, int, int]]:
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
        self.speed_history.clear()
        self.start_time = time.time()


def render_dashboard(
    frame: np.ndarray,
    stream: CCTVStream,
    analytics: IntersectionAnalytics,
    active_vehicles: List[dict],
    show_trails: bool = True,
    show_hud: bool = True,
    bev_boxes: Optional[List[np.ndarray]] = None,
) -> np.ndarray:
    annotated = frame.copy()
    h, w = frame.shape[:2]

    if bev_boxes:
        box_colors = [COLOR_BEV_UP, COLOR_BEV_DOWN]
        for idx, box in enumerate(bev_boxes):
            if box is not None and len(box) == 4:
                bev_pts = box.reshape((-1, 1, 2)).astype(np.int32)
                color = box_colors[idx % len(box_colors)]
                cv2.polylines(
                    annotated,
                    [bev_pts],
                    isClosed=True,
                    color=color,
                    thickness=2,
                )

    all_objects = [("vehicle", v) for v in active_vehicles]

    for obj_type, obj in all_objects:
        bx1, by1, bx2, by2 = map(int, obj["box"])
        tid = obj["id"]
        cls_name = obj["class"]
        speed = obj["speed"]
        history = obj["history"]

        primary_color = (
            COLOR_STOPPED
            if speed is not None and speed < 0.6
            else COLOR_VEHICLE
        )

        if show_trails and len(history) > 1:
            pts = np.array(history, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(
                annotated,
                [pts],
                isClosed=False,
                color=primary_color,
                thickness=2,
                lineType=cv2.LINE_AA,
            )

        cv2.rectangle(annotated, (bx1, by1), (bx2, by2), primary_color, 2)

        label_text = (
            f"#{tid} {cls_name} | {speed:.1f} MPH"
            if speed is not None
            else f"#{tid} {cls_name}"
        )

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.40
        thickness = 1
        (tw, th), _ = cv2.getTextSize(label_text, font, scale, thickness)

        pill_y1 = max(0, by1 - th - 6)
        pill_y2 = by1
        pill_x1 = bx1
        pill_x2 = min(w, bx1 + tw + 8)

        cv2.rectangle(
            annotated,
            (pill_x1, pill_y1),
            (pill_x2, pill_y2),
            primary_color,
            -1,
        )
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

    bar_height = 36
    draw_translucent_panel(
        annotated,
        0,
        0,
        w,
        bar_height,
        bg_color=COLOR_HUD_BG,
        alpha=0.85,
        border_color=(50, 60, 70),
    )

    buf_info = (
        f" | Buffer: {stream.buffer_level}f" if stream.is_network_stream else ""
    )
    fps_text = f"{analytics.smoothed_fps:.1f} FPS"
    header_title = (
        f"INTERSECTION ANALYTICS  •  {stream.display_name}{buf_info}"
    )

    cv2.putText(
        annotated,
        header_title,
        (14, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        COLOR_TEXT_WHITE,
        1,
        lineType=cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        fps_text,
        (w - 90, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        COLOR_ACCENT,
        1,
        lineType=cv2.LINE_AA,
    )

    panel_w = 260
    panel_h = 160
    panel_x = w - panel_w - 14
    panel_y = bar_height + 12

    draw_translucent_panel(
        annotated,
        panel_x,
        panel_y,
        panel_w,
        panel_h,
        bg_color=COLOR_HUD_BG,
        alpha=0.82,
    )

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
    cv2.line(
        annotated,
        (panel_x + 12, panel_y + 26),
        (panel_x + panel_w - 12, panel_y + 26),
        (50, 60, 70),
        1,
    )

    density_label, density_color = analytics.get_traffic_density_label(
        len(active_vehicles)
    )

    metrics = [
        (
            "Active Vehicles",
            f"{len(active_vehicles)} (Total: {analytics.total_vehicles})",
            COLOR_VEHICLE,
        ),
        (
            "Avg Vehicle Speed",
            f"{analytics.average_vehicle_speed:.1f} MPH",
            COLOR_TEXT_WHITE,
        ),
        ("Traffic Density", density_label, density_color),
    ]

    curr_y = panel_y + 48
    for label, val, val_color in metrics:
        cv2.putText(
            annotated,
            label,
            (panel_x + 14, curr_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            COLOR_TEXT_MUTED,
            1,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            val,
            (panel_x + 130, curr_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            val_color,
            1,
            lineType=cv2.LINE_AA,
        )
        curr_y += 24

    ctrl_y = h - 22
    draw_translucent_panel(
        annotated,
        0,
        h - 28,
        w,
        28,
        bg_color=COLOR_HUD_BG,
        alpha=0.75,
        border_color=(40, 50, 60),
    )
    controls_text = "[SPACE] Pause/Resume   [H] Toggle HUD   [T] Toggle Trails   [R] Reset Counters   [Q] Quit   [C] Calibrate   [B] Toggle BEV"
    cv2.putText(
        annotated,
        controls_text,
        (14, ctrl_y + 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.36,
        COLOR_TEXT_MUTED,
        1,
        lineType=cv2.LINE_AA,
    )

    return annotated


def setup_bev_transform(
    road_coords: Optional[np.ndarray],
    display_scale: float,
    frame: np.ndarray,
    bev_background: bool = False,
    num_lanes: int = 3,
):
    if road_coords is None or len(road_coords) != 4:
        return None, None, 0, 0, None

    src_pts = np.float32(road_coords) * np.float32(display_scale)
    tl, tr, br, bl = src_pts

    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    bev_width = int(max(width_top, width_bottom))

    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    bev_height = int(max(height_left, height_right))

    if bev_width <= 0 or bev_height <= 0:
        return None, None, 0, 0, None

    dst_pts = np.float32(
        [[0, 0], [bev_width, 0], [bev_width, bev_height], [0, bev_height]]
    )
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

    if bev_background:
        bev_frame = cv2.warpPerspective(frame, matrix, (bev_width, bev_height))
    else:
        bev_frame = np.zeros((bev_height, bev_width, 3), dtype=np.uint8)
        bev_frame[:] = (255, 255, 255)

    dtl, dtr, dbr, dbl = dst_pts[0], dst_pts[1], dst_pts[2], dst_pts[3]
    for i in range(1, num_lanes):
        t = i / num_lanes
        pt_top = ((1 - t) * dtl + t * dtr).astype(int)
        pt_bottom = ((1 - t) * dbl + t * dbr).astype(int)
        cv2.line(
            bev_frame,
            tuple(pt_top),
            tuple(pt_bottom),
            color=(180, 180, 180),
            thickness=1,
        )

    return matrix, bev_frame, bev_width, bev_height, src_pts


def load_camera_config(source_key: str):
    """Helper to safely load camera calibration config from config.json."""
    try:
        with open("config.json", "r") as file:
            data = json.load(file)
    except FileNotFoundError:
        data = {}

    cam_data = data.get(source_key, {})
    coords_up = (
        np.float32(cam_data["points_up"])
        if "points_up" in cam_data
        else (
            np.float32(cam_data["points"]) if "points" in cam_data else None
        )
    )
    coords_down = (
        np.float32(cam_data["points_down"])
        if "points_down" in cam_data
        else None
    )

    num_lanes_up = cam_data.get("lanes_up", cam_data.get("num_lanes_up", 3))
    num_lanes_down = cam_data.get(
        "lanes_down", cam_data.get("num_lanes_down", 3)
    )

    distance_up = cam_data.get("distance_up", (0.0, 0.0))
    distance_down = cam_data.get("distance_down", (0.0, 0.0))

    return (
        coords_up,
        coords_down,
        num_lanes_up,
        num_lanes_down,
        distance_up,
        distance_down,
    )


def main():
    config = configparser.ConfigParser()
    config.read("settings.ini")

    source_key = config["gui"]["source"]
    print(f"[*] Initializing stream from: {source_key}")
    stream = CCTVStream(source_key)

    print(f"[*] Loading YOLO tracking model: {config['model']['model']}")
    model = YOLO(config["model"]["model"])

    ret, frame = stream.read()
    if not ret or frame is None:
        print("[!] Error: Failed to retrieve video stream.")
        stream.release()
        sys.exit(1)

    height, width = frame.shape[:2]
    display_scale = config.getfloat("gui", "display-scale")
    disp_w, disp_h = int(width * display_scale), int(height * display_scale)

    # Initial calibration parameters load
    (
        coords_up,
        coords_down,
        num_lanes_up,
        num_lanes_down,
        distance_up,
        distance_down,
    ) = load_camera_config(source_key)

    bev_background = False

    track_history = defaultdict(lambda: deque(maxlen=120))
    track_bev_history_up = defaultdict(lambda: deque(maxlen=120))
    track_bev_history_down = defaultdict(lambda: deque(maxlen=120))
    vehicle_speeds = {}
    analytics = IntersectionAnalytics()

    writer = None
    if config["model"]["save"]:
        os.makedirs(
            os.path.dirname(os.path.abspath(config["model"]["save"])),
            exist_ok=True,
        )
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            config["model"]["save"],
            fourcc,
            stream.target_fps,
            (disp_w, disp_h),
        )

    show_trails = config.getboolean("gui", "show-trails")
    show_hud = config.getboolean("gui", "show-hud")
    paused = False

    target_class_ids = list(ALL_TARGET_CLASSES.keys())

    print(f"[+] Monitoring {stream.display_name} @ {stream.target_fps:.1f} FPS")

    fps_timer = time.time()
    current_fps = stream.target_fps

    bev_frame_up, bev_frame_down = None, None

    while True:
        loop_start = time.time()

        if not paused:
            ret, frame = stream.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            if display_scale != 1:
                frame = cv2.resize(
                    frame, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR
                )

            # Perform manual GUI calibration if coordinates are missing
            if coords_up is None and coords_down is None:
                calibrate(frame, source_key, display_scale)
                (
                    coords_up,
                    coords_down,
                    num_lanes_up,
                    num_lanes_down,
                    distance_up,
                    distance_down,
                ) = load_camera_config(source_key)

            # Setup perspective transforms
            (
                matrix_up,
                bev_frame_up,
                bev_w_up,
                bev_h_up,
                src_pts_up,
            ) = setup_bev_transform(
                coords_up, display_scale, frame, bev_background, num_lanes_up
            )
            (
                matrix_down,
                bev_frame_down,
                bev_w_down,
                bev_h_down,
                src_pts_down,
            ) = setup_bev_transform(
                coords_down,
                display_scale,
                frame,
                bev_background,
                num_lanes_down,
            )

            # YOLO multi-object tracking (ByteTrack)
            results = model.track(
                frame,
                persist=True,
                classes=target_class_ids,
                conf=config.getfloat("model", "confidence"),
                tracker="bytetrack.yaml",
                device=config.getint("model", "device"),
                verbose=False,
            )[0]

            active_vehicles = []
            current_active_ids = set()

            if results.boxes is not None and results.boxes.id is not None:
                boxes = results.boxes.xyxy.cpu().numpy()
                track_ids = results.boxes.id.int().cpu().numpy()
                classes = results.boxes.cls.int().cpu().numpy()

                for box, track_id, cls_id in zip(boxes, track_ids, classes):
                    cls_id = int(cls_id)
                    current_active_ids.add(track_id)
                    x1, y1, x2, y2 = box
                    ground_pt = ((x1 + x2) / 2.0, y2)
                    track_history[track_id].append(ground_pt)

                    input_pts = np.array(
                        ground_pt, dtype=np.float32
                    ).reshape(-1, 1, 2)

                    # Transform point to UP perspective
                    bx_up, by_up = -1, -1
                    if matrix_up is not None:
                        t_pt = (
                            cv2.perspectiveTransform(input_pts, matrix_up)
                            .reshape(-1, 2)
                            .astype(int)[0]
                        )
                        bx_up, by_up = t_pt
                        track_bev_history_up[track_id].append(t_pt)

                    # Transform point to DOWN perspective
                    bx_dn, by_dn = -1, -1
                    if matrix_down is not None:
                        t_pt = (
                            cv2.perspectiveTransform(input_pts, matrix_down)
                            .reshape(-1, 2)
                            .astype(int)[0]
                        )
                        bx_dn, by_dn = t_pt
                        track_bev_history_down[track_id].append(t_pt)

                    # Check zone presence
                    in_up = (
                        (0 <= bx_up < bev_w_up and 0 <= by_up < bev_h_up)
                        if matrix_up is not None
                        else False
                    )
                    in_down = (
                        (0 <= bx_dn < bev_w_down and 0 <= by_dn < bev_h_down)
                        if matrix_down is not None
                        else False
                    )

                    # Speed calculation using real-world distances
                    if track_id in vehicle_speeds:
                        speed = vehicle_speeds[track_id]
                        velocity = (0.0, 0.0)
                    else:
                        if in_up:
                            direction = "up"
                            active_bev_hist = track_bev_history_up.get(
                                track_id, []
                            )
                            active_h = bev_h_up
                            active_dist = distance_up
                        elif in_down:
                            direction = "down"
                            active_bev_hist = track_bev_history_down.get(
                                track_id, []
                            )
                            active_h = bev_h_down
                            active_dist = distance_down
                        else:
                            direction = "none"
                            active_bev_hist = []
                            active_h = 0
                            active_dist = (0.0, 0.0)

                        dist_w = (
                            active_dist[0]
                            if active_dist and len(active_dist) > 0
                            else 0.0
                        )
                        dist_l = (
                            active_dist[1]
                            if active_dist and len(active_dist) > 1
                            else 0.0
                        )

                        if (
                            len(active_bev_hist) > 0
                            and active_h > 0
                            and dist_l > 0
                        ):
                            velocity, speed = calculate_vehicle_speed(
                                bev_history=active_bev_hist,
                                bev_height=active_h,
                                direction=direction,
                                fps=stream.target_fps,
                                length=dist_l,
                                width=dist_w,
                            )
                        else:
                            velocity, speed = (0.0, 0.0), None

                        if speed is not None:
                            print(
                                f"#{track_id} {direction} speed: {speed:.2f}"
                                " MPH"
                            )
                            vehicle_speeds[track_id] = speed

                    cls_label = ALL_TARGET_CLASSES.get(
                        cls_id, model.names.get(cls_id, "object")
                    )

                    obj_data = {
                        "id": track_id,
                        "box": box,
                        "class": cls_label,
                        "center": ground_pt,
                        "velocity": velocity,
                        "speed": speed,
                        "history": list(track_history[track_id]),
                    }

                    primary_color = (
                        COLOR_STOPPED
                        if speed is not None and speed < 0.6
                        else COLOR_VEHICLE
                    )
                    label_text = (
                        f"#{track_id} {cls_label} | {speed:.1f} MPH"
                        if speed is not None
                        else f"#{track_id} {cls_label}"
                    )

                    # Top-down BEV vehicle overlays
                    if (
                        bev_frame_up is not None
                        and 0 <= bx_up < bev_w_up
                        and 0 <= by_up < bev_h_up
                    ):
                        cv2.circle(
                            bev_frame_up, (bx_up, by_up), 6, primary_color, -1
                        )
                        cv2.putText(
                            bev_frame_up,
                            label_text,
                            (bx_up + 8, by_up + 4),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            (0, 0, 0),
                            1,
                        )

                    if (
                        bev_frame_down is not None
                        and 0 <= bx_dn < bev_w_down
                        and 0 <= by_dn < bev_h_down
                    ):
                        cv2.circle(
                            bev_frame_down,
                            (bx_dn, by_dn),
                            6,
                            primary_color,
                            -1,
                        )
                        cv2.putText(
                            bev_frame_down,
                            label_text,
                            (bx_dn + 8, by_dn + 4),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            (0, 0, 0),
                            1,
                        )

                    active_vehicles.append(obj_data)

            # Cleanup stale tracking IDs
            for tid in list(track_history.keys()):
                if (
                    tid not in current_active_ids
                    and len(track_history[tid]) > 0
                ):
                    track_history[tid].popleft()
                    if len(track_history[tid]) == 0:
                        del track_history[tid]

            for tid in list(track_bev_history_up.keys()):
                if tid not in current_active_ids:
                    if len(track_bev_history_up[tid]) > 0:
                        track_bev_history_up[tid].popleft()
                    if len(track_bev_history_up[tid]) == 0:
                        del track_bev_history_up[tid]

            for tid in list(track_bev_history_down.keys()):
                if tid not in current_active_ids:
                    if len(track_bev_history_down[tid]) > 0:
                        track_bev_history_down[tid].popleft()
                    if len(track_bev_history_down[tid]) == 0:
                        del track_bev_history_down[tid]

            for tid in list(vehicle_speeds.keys()):
                if tid not in current_active_ids:
                    del vehicle_speeds[tid]

            update_master_trajectory_store(track_history)

            # FPS counter
            now = time.time()
            fps_delta = now - fps_timer
            if fps_delta >= 0.2:
                current_fps = 1.0 / (
                    loop_start - fps_timer
                    if fps_timer != loop_start
                    else 0.033
                )
                fps_timer = now

            analytics.update(active_vehicles, current_fps)

            # Render Dashboard and overlays
            bev_boxes = [coords_up, coords_down]
            annotated_frame = render_dashboard(
                frame=frame,
                stream=stream,
                analytics=analytics,
                active_vehicles=active_vehicles,
                show_trails=show_trails,
                show_hud=show_hud,
                bev_boxes=bev_boxes,
            )

            if writer:
                writer.write(annotated_frame)

            cv2.imshow("Intersection Analytics Dashboard", annotated_frame)
            if bev_frame_up is not None:
                cv2.imshow("BEV - Up Stream", bev_frame_up)
            if bev_frame_down is not None:
                cv2.imshow("BEV - Down Stream", bev_frame_down)

        elapsed = time.time() - loop_start
        sleep_delay = stream.get_sleep_interval(elapsed)

        key = cv2.waitKey(sleep_delay if not paused else 30) & 0xFF
        if key == ord("q"):
            break
        elif key == 32:  # Spacebar
            paused = not paused
        elif key in (ord("h"), ord("H")):
            show_hud = not show_hud
        elif key in (ord("t"), ord("T")):
            show_trails = not show_trails
        elif key in (ord("r"), ord("R")):
            analytics.reset()
            print("[+] Intersection counters reset.")
        elif key in (ord("c"), ord("C")):
            # Auto-calibrate dual trapezoids
            res = auto_calibrate(
                frame=frame,
                bev_frame_down=bev_frame_down,
                bev_frame_up=bev_frame_up,
            )
            (
                coords_up,
                coords_down,
                num_lanes_up,
                num_lanes_down,
                distance_up,
                distance_down,
                num_dashes_up,
                num_dashes_down,
            ) = res

            try:
                with open("config.json", "r") as file:
                    data = json.load(file)
            except FileNotFoundError:
                data = {}

            if source_key not in data:
                data[source_key] = {}

            if coords_up is not None:
                data[source_key]["points_up"] = coords_up.tolist()
                data[source_key]["lanes_up"] = num_lanes_up
                data[source_key]["distance_up"] = distance_up
                data[source_key]["num_dashes_up"] = num_dashes_up

            if coords_down is not None:
                data[source_key]["points_down"] = coords_down.tolist()
                data[source_key]["lanes_down"] = num_lanes_down
                data[source_key]["distance_down"] = distance_down
                data[source_key]["num_dashes_down"] = num_dashes_down

            with open("config.json", "w") as file:
                json.dump(data, file, indent=4)

            print("[+] Auto-calibrated both BEV directions successfully.")
        elif key in (ord("b"), ord("B")):
            bev_background = not bev_background

    stream.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print("[+] Dashboard session ended.")


if __name__ == "__main__":
    main()