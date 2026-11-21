"""Vehicle identification and speed estimation using CCTV cameras

Author: Cooper Bowden
Date: 2026-11-14
Version: 1.0.0
"""

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

from calculate_motion_stats import calculate_vehicle_speed, estimate_bev_length
from calibrate import (
    auto_calibrate,
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


class highway_analytics:

    def __init__(self):
        self.seen_vehicles = set()
        self.recorded_speeds = set()  # Track which IDs have already been averaged
        self.speed_history_up = deque(maxlen=60)
        self.speed_history_down = deque(maxlen=60)
        self.fps_history = deque(maxlen=30)
        self.start_time = time.time()

    def update(self, active_vehicles: List[dict], fps: float):
        for v in active_vehicles:
            tid = v["id"]
            self.seen_vehicles.add(tid)

            if (
                    tid not in self.recorded_speeds
                    and v["speed"] is not None
                    and v["speed"] >= 0.6
            ):
                if v["direction"] == "up":
                    self.speed_history_up.append(v["speed"])
                    self.recorded_speeds.add(tid)
                elif v["direction"] == "down":
                    self.speed_history_down.append(v["speed"])
                    self.recorded_speeds.add(tid)

        if fps > 0:
            self.fps_history.append(fps)

    @property
    def total_vehicles(self) -> int:
        return len(self.seen_vehicles)

    @property
    def average_vehicle_speed_up(self) -> float:
        return (
            float(np.mean(self.speed_history_up)) if self.speed_history_up else 0.0
        )
    @property
    def average_vehicle_speed_down(self) -> float:
        return (
            float(np.mean(self.speed_history_down)) if self.speed_history_down else 0.0
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
        self.recorded_speeds.clear()
        self.speed_history_up.clear()
        self.speed_history_down.clear()
        self.start_time = time.time()


def render_dashboard(
    frame: np.ndarray,
    stream: CCTVStream,
    analytics: highway_analytics,
    distances: List[int],
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
    font = cv2.FONT_HERSHEY_SIMPLEX
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
        font,
        0.48,
        COLOR_TEXT_WHITE,
        1,
        lineType=cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        fps_text,
        (w - 90, 23),
        font,
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
        font,
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
            "Avg Speed Up",
            f"{analytics.average_vehicle_speed_up:.1f} MPH",
            COLOR_TEXT_WHITE,
        ),
        (
            "Avg Speed Down",
            f"{analytics.average_vehicle_speed_down:.1f} MPH",
            COLOR_TEXT_WHITE,
        ),
        (
            "Distance Up",
            f"{distances[0]:.1f} ft",
            COLOR_TEXT_WHITE,
        ),
        (
            "Distance Down",
            f"{distances[1]:.1f} ft",
            COLOR_TEXT_WHITE,
        ),
    ]

    curr_y = panel_y + 48
    for label, val, val_color in metrics:
        cv2.putText(
            annotated,
            label,
            (panel_x + 14, curr_y),
            font,
            0.38,
            COLOR_TEXT_MUTED,
            1,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            val,
            (panel_x + 130, curr_y),
            font,
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
    controls_text = ("[SPACE] Pause/Resume   [H] Toggle HUD   [T] Toggle Trails   [R] Reset Counters   [Q] Quit   "
                     "[C] Calibrate   [B] Toggle BEV")
    cv2.putText(
        annotated,
        controls_text,
        (14, ctrl_y + 12),
        font,
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
    masked_frame: np.ndarray,
    bev_background: bool = True,
    num_lanes: int = 3,
):
    if road_coords is None or len(road_coords) != 4:
        return None, None, 0, 0, None

    src_pts = np.float32(road_coords)
    tl, tr, br, bl = road_coords

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

    bev_frame_masked = cv2.warpPerspective(masked_frame, matrix, (bev_width, bev_height))

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

    return matrix, bev_frame, bev_frame_masked, bev_width, bev_height, src_pts


def load_camera_config(source_key: str):
    """Helper to safely load camera calibration config from config.json."""
    try:
        with open("../config.json", "r") as file:
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

    distance_up = cam_data.get("distance_up", 0)
    distance_down = cam_data.get("distance_down", 0)
    dash_length = cam_data.get("dash_length", 0)

    return (
        coords_up,
        coords_down,
        num_lanes_up,
        num_lanes_down,
        distance_up,
        distance_down,
        dash_length,
    )


def main():
    config = configparser.ConfigParser()
    config.read("../settings.ini")
    font = cv2.FONT_HERSHEY_SIMPLEX

    source_key = config["gui"]["source"]
    save_images = config["gui"]["save-images"]
    print(f"[*] Initializing stream from: {source_key}")
    stream = CCTVStream(source_key)

    print(f"[*] Loading YOLO tracking model: {config['model']['model']}")
    model = YOLO("models/" + config["model"]["model"])

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
        dash_length
    ) = load_camera_config(source_key)

    bev_background = True

    track_history = defaultdict(lambda: deque(maxlen=120))
    track_bev_history_up = defaultdict(lambda: deque(maxlen=120))
    track_bev_history_down = defaultdict(lambda: deque(maxlen=120))
    vehicle_speeds = {}
    analytics = highway_analytics()

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

    bev_frame_annotated_up, bev_frame_annotated_down = None, None

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

            # create mask for BEV exclusion to detect lanes without car interference
            veh_exclusion_mask = np.ones((disp_h, disp_w), dtype=np.uint8) * 255
            pad = 10  # Padding around bounding boxes to mask vehicle body and shadows

            if results.boxes is not None and len(results.boxes) > 0:
                boxes_np = results.boxes.xyxy.cpu().numpy()
                for box in boxes_np:
                    x1, y1, x2, y2 = map(int, box)
                    px1, py1 = max(0, x1 - pad), max(0, y1 - pad)
                    px2, py2 = min(disp_w, x2 + pad), min(disp_h, y2 + pad)
                    cv2.rectangle(veh_exclusion_mask, (px1, py1), (px2, py2), 0, -1)

            # Apply 2D mask to clear out vehicles before perspective warping
            masked_frame = cv2.bitwise_and(frame, frame, mask=veh_exclusion_mask)

            # check if calibration is done
            if not (coords_up is None and coords_down is None):
                # Setup perspective transforms
                (
                    matrix_up,
                    bev_frame_annotated_up,
                    bev_frame_masked_up,
                    bev_w_up,
                    bev_h_up,
                    src_pts_up,
                ) = setup_bev_transform(
                    coords_up, display_scale, frame, masked_frame, bev_background, num_lanes_up
                )
                (
                    matrix_down,
                    bev_frame_annotated_down,
                    bev_frame_masked_down,
                    bev_w_down,
                    bev_h_down,
                    src_pts_down,
                ) = setup_bev_transform(
                    coords_down,
                    display_scale,
                    frame,
                    masked_frame,
                    bev_background,
                    num_lanes_down,
                )
            else:
                matrix_down = None
                matrix_up = None


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
                        speed, direction = vehicle_speeds[track_id]
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
                            active_dist = 0.0


                        if (
                            len(active_bev_hist) > 0
                            and active_h > 0
                            and active_dist != (0.0, 0.0)
                        ):
                            velocity, speed = calculate_vehicle_speed(
                                bev_history=active_bev_hist,
                                bev_height=active_h,
                                direction=direction,
                                fps=stream.target_fps,
                                length=active_dist,
                            )
                        else:
                            velocity, speed = (0.0, 0.0), None

                        # Flag when vehicle speed is 120% greater than average
                        if speed is not None:
                            print(f"#{track_id} {direction} speed: {speed:.2f} MPH")
                            vehicle_speeds[track_id] = (speed, direction)
                            if (direction == "up"):
                                avg_speed = analytics.average_vehicle_speed_up
                            if (direction == "down"):
                                avg_speed = analytics.average_vehicle_speed_down
                            if (speed > (avg_speed * 1.20) and avg_speed != 0):
                                print(f"* High speed Detected * [{track_id} {direction}]: {speed:.2f} MPH -> "
                                      f"{(speed/avg_speed)*100:.1f}% faster than average "
                                      f"({avg_speed:.2f} MPH)")


                    cls_label = ALL_TARGET_CLASSES.get(
                        cls_id, model.names.get(cls_id, "object")
                    )

                    obj_data = {
                        "id": track_id,
                        "box": box,
                        "class": cls_label,
                        "center": ground_pt,
                        "velocity": velocity,
                        "direction": direction,
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
                        bev_frame_annotated_up is not None
                        and 0 <= bx_up < bev_w_up
                        and 0 <= by_up < bev_h_up
                    ):
                        cv2.circle(
                            bev_frame_annotated_up, (bx_up, by_up), 6, primary_color, -1
                        )
                        cv2.putText(
                            bev_frame_annotated_up,
                            label_text,
                            (bx_up + 8, by_up + 4),
                            font,
                            0.4,
                            (0, 0, 0),
                            1,
                        )

                    if (
                        bev_frame_annotated_down is not None
                        and 0 <= bx_dn < bev_w_down
                        and 0 <= by_dn < bev_h_down
                    ):
                        cv2.circle(
                            bev_frame_annotated_down,
                            (bx_dn, by_dn),
                            6,
                            primary_color,
                            -1,
                        )
                        cv2.putText(
                            bev_frame_annotated_down,
                            label_text,
                            (bx_dn + 8, by_dn + 4),
                            font,
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

            update_master_trajectory_store(track_history, store_file=(f"../point_clouds/trajectories_{source_key}.npz"))

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
            bev_boxes = [src_pts_up, src_pts_down]
            annotated_frame = render_dashboard(
                frame=frame,
                stream=stream,
                analytics=analytics,
                distances=[distance_up, distance_down],
                active_vehicles=active_vehicles,
                show_trails=show_trails,
                show_hud=show_hud,
                bev_boxes=bev_boxes,
            )
            if (matrix_down is None and matrix_up is None):
                cv2.putText(
                    annotated_frame,
                    "Press C to calibrate for speed information",
                    (14, 100),
                    font,
                    1,
                    (0, 255, 0),
                    2,
                )
            if writer:
                writer.write(annotated_frame)

            cv2.imshow("Main Frame", annotated_frame)
            if bev_frame_annotated_up is not None:
                cv2.imshow("BEV - Up Stream", bev_frame_annotated_up)
            if bev_frame_annotated_down is not None:
                cv2.imshow("BEV - Down Stream", bev_frame_annotated_down)

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
            if (save_images == True):
                cv2.imwrite("../screenshots/frame.png", frame)
                cv2.imwrite("../screenshots/annotated_frame.png", annotated_frame)

            res = auto_calibrate(
                frame=frame,
                store_file=(f"../point_clouds/trajectories_{source_key}.npz"),
                save_images=save_images
            )
            (
                coords_up,
                coords_down,
                num_lanes_up,
                num_lanes_down,
            ) = res

            # Generate perspective transforms for length calculation
            if coords_up is not None:
                if (save_images == True): cv2.imwrite(
                    "../screenshots/bev_frame_annotated_up.png", bev_frame_annotated_up)
                (
                    matrix_up,
                    bev_frame_annotated_up,
                    bev_frame_masked_up,
                    bev_w_up,
                    bev_h_up,
                    _,
                ) = setup_bev_transform(
                    coords_up, display_scale, frame, masked_frame, bev_background, num_lanes_up
                )
                if bev_frame_masked_up is not None:
                    distance_up = estimate_bev_length(
                        bev_frame_masked_up,
                        display_scale=display_scale,
                        num_lanes=num_lanes_up,
                        save_images=save_images)
                    if (save_images == True): cv2.imwrite(
                        "../screenshots/bev_frame_masked_up.png", bev_frame_masked_up)

            if coords_down is not None:
                if (save_images == True): cv2.imwrite(
                    "../screenshots/bev_frame_annotated_down.png", bev_frame_annotated_down)
                (
                    matrix_down,
                    bev_frame_annotated_down,
                    bev_frame_masked_down,
                    bev_w_down,
                    bev_h_down,
                    _,
                ) = setup_bev_transform(
                    coords_down, display_scale, frame, masked_frame, bev_background, num_lanes_down
                )
                if bev_frame_masked_down is not None:
                    distance_down = estimate_bev_length(
                        bev_frame_masked_down,
                        display_scale=display_scale,
                        num_lanes=num_lanes_down)
                    if (save_images == True): cv2.imwrite(
                        "../screenshots/bev_frame_masked_down.png", bev_frame_masked_down)

            try:
                with open("../config.json", "r") as file:
                    data = json.load(file)
            except FileNotFoundError:
                data = {}

            if source_key not in data:
                data[source_key] = {}

            if coords_up is not None:
                data[source_key]["points_up"] = coords_up.tolist()
                data[source_key]["lanes_up"] = num_lanes_up

            if coords_down is not None:
                data[source_key]["points_down"] = coords_down.tolist()
                data[source_key]["lanes_down"] = num_lanes_down

            with open("../config.json", "w") as file:
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