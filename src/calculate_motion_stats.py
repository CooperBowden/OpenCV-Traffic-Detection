import numpy as np
import cv2
import math
import os


def calculate_vehicle_speed(
    bev_history: list,
    bev_height: int,
    length: int,
    direction: str,
    fps: float = 30.0,
    margin_percent: float = 0.25,
) -> tuple[tuple[float, float], float]:
    """Calculates vehicle speed and velocity vector using a virtual BEV speed trap.

    Defines entry (Line A) and exit (Line B) thresholds inset from the top and
    bottom edges of the Bird's-Eye View (BEV) canvas[cite: 1]. When a vehicle crosses the
    exit line on the current frame step, sub-frame linear interpolation is used
    to determine the exact fractional frame delta between crossings and convert
    the known physical trap distance into miles per hour (MPH)[cite: 1].

    Args:
        bev_history: Ordered sequence of tracked `(x, y)` coordinates in BEV
            pixel space[cite: 1].
        bev_height: Height of the BEV frame canvas in pixels[cite: 1].
        length: Total real-world longitudinal length of the BEV zone in feet[cite: 1].
        direction: Travel direction (`"down"` for top-to-bottom motion; any
            other value is treated as bottom-to-top motion)[cite: 1].
        fps: Target video frame rate in frames per second. Defaults to 30.0[cite: 1].
        margin_percent: Fractional inset safety margin applied to both the top
            and bottom edges to position the trap lines. Defaults to 0.25[cite: 1].

    Returns:
        A 2-tuple `((vx_mph, vy_mph), speed_mph)` containing the 2D velocity
        components in MPH and the scalar speed in MPH[cite: 1]. Returns
        `((0.0, 0.0), None)` if fewer than 2 points exist, inputs are invalid,
        or the vehicle has not completed both trap-line crossings on this
        step; returns `((0.0, 0.0), 0.0)` if computed speed is under 1.0 MPH[cite: 1].
    """
    if len(bev_history) < 2 or fps <= 0 or bev_height <= 0:
        return (0.0, 0.0), None

    pts = list(bev_history)
    # Define Trap Line Y-coordinates
    if direction == "down":
        line_a_y = bev_height * margin_percent
        line_b_y = bev_height * (1.0 - margin_percent)
    else:
        line_a_y = bev_height * (1.0 - margin_percent)
        line_b_y = bev_height * margin_percent

    # Real-world distance covered between the two trap lines (feet)
    trap_distance_ft = length * (1.0 - 2.0 * margin_percent)

    # Scan track history for exact sub-frame crossing indices
    y_prev_step = pts[-2][1]
    y_curr_step = pts[-1][1]

    just_crossed_b = (y_prev_step < line_b_y <= y_curr_step) or (
        y_prev_step > line_b_y >= y_curr_step
    )

    # If Line B was not crossed on this frame step, exit early
    if not just_crossed_b:
        return (0.0, 0.0), None

    idx_a = None
    # fractional frame where line b was crossed
    idx_b = (len(pts) - 2) + (
        (line_b_y - y_prev_step) / (y_curr_step - y_prev_step)
        if (y_curr_step - y_prev_step) != 0
        else 0.0
    )
    for i in range(1, len(pts)):
        y_prev, y_curr = pts[i - 1][1], pts[i][1]
        if (
            (y_prev < line_a_y <= y_curr) or (y_prev > line_a_y >= y_curr)
        ) and idx_a is None:
            dy = y_curr - y_prev
            ratio = (line_a_y - y_prev) / dy if dy != 0 else 0.0
            idx_a = (i - 1) + ratio
            break
    if idx_a is None:
        return (0.0, 0.0), None

    # 3. Calculate Speed
    FTS_TO_MPH = 3600.0 / 5280.0  # Conversion constant (ft/s to mph)

    if idx_a is not None and idx_b is not None:
        # Exact Speed Trap calculation (both lines crossed)
        frame_delta = abs(idx_b - idx_a)
        time_sec = frame_delta / fps
        speed_fts = trap_distance_ft / time_sec if time_sec > 0 else 0.0
        speed_mph = speed_fts * FTS_TO_MPH
    else:
        speed_mph = None
        vx_mph, vy_mph = None, None
    if speed_mph != None:
        # Filter out tracking jitter for stationary objects
        if speed_mph < 1.0:
            return (0.0, 0.0), 0.0

        # Compute Velocity Vector and
        dx_px = pts[-1][0] - pts[0][0]
        dy_px = pts[-1][1] - pts[0][1]

        total_px_dist = np.hypot(dx_px, dy_px)
        if total_px_dist > 0:
            vx_mph = (dx_px / total_px_dist) * speed_mph
            vy_mph = (dy_px / total_px_dist) * speed_mph
        else:
            vx_mph, vy_mph = 0.0, 0.0
    return (vx_mph, vy_mph), speed_mph


def estimate_bev_length(
    bev_frame: np.ndarray,
    display_scale,
    num_lanes: int,
    save_images: bool = False,
) -> float:
    """Estimates the real-world length of a BEV frame from painted lane dashes.

    Thresholds and morphologically filters narrow vertical strips along estimated
    lane divider positions on a vehicle-masked BEV image, filters out clipped
    or outlier dash contours via IQR, and scales pixel height to feet using a
    standard 10-foot highway dash reference[cite: 1].

    Args:
        bev_frame: Bird's-Eye View BGR or grayscale image frame (typically with
            vehicles masked out)[cite: 1].
        display_scale: Scaling multiplier applied to the display frame relative
            to native resolution[cite: 1].
        num_lanes: Number of traffic lanes spanned by `bev_frame`[cite: 1].
        save_images: Whether to save the annotated dash visualization to
            `../screenshots/dash_length_estimation.png`. Defaults to False[cite: 1].

    Returns:
        Estimated real-world longitudinal length of the BEV region in feet[cite: 1].
        Returns `3.0` if `bev_frame` is `None` or empty, and `0` if no valid
        lane dashes are detected[cite: 1].
    """
    basepath = os.path.dirname(__file__)
    if bev_frame is None or bev_frame.size == 0:
        return 3.0

    if len(bev_frame.shape) == 3:
        gray = cv2.cvtColor(bev_frame, cv2.COLOR_BGR2GRAY)
        viz_frame = bev_frame.copy()
    else:
        gray = bev_frame.copy()
        viz_frame = cv2.cvtColor(bev_frame, cv2.COLOR_GRAY2BGR)

    h, w = gray.shape[:2]

    # 1. Thresholding to isolate bright white road paint
    _, binary = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY)

    # Vertical opening to eliminate horizontal noise/reflections
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 7))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # 2. Locate isolated lane divider X positions
    lane_width_px = w / float(num_lanes)
    divider_x_positions = [int(i * lane_width_px) for i in range(1, num_lanes)]
    if not divider_x_positions:
        divider_x_positions = [int(w / 2.0)]

    strip_w = 30  # Half-width of vertical strip around divider line
    dash_lengths = []

    for div_x in divider_x_positions:
        x_start = max(0, div_x - strip_w)
        x_end = min(w, div_x + strip_w)

        # Slice narrow isolated vertical strip along the divider line
        strip = binary[:, x_start:x_end]
        cv2.imwrite(
            os.path.abspath(
                os.path.join(basepath, "..", "screenshots", "dash_strip.png")
            ),
            strip,
        )
        # Extract contours within this specific lane segment
        contours, _ = cv2.findContours(
            strip, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            # Filter for vertically elongated dashes (height >= 10px, height > width)
            if bh >= 10 and bh > bw and bw <= 35:
                # Map local strip coordinates back to global BEV frame coordinates
                x_mid_global = x_start + x + (bw // 2)
                pt_start = (x_mid_global, y)
                pt_end = (x_mid_global, y + bh)
                # find distance of points and add to list
                dash_lengths.append(math.dist(pt_start, pt_end) / display_scale)
                # Draw single centerline per dash
                cv2.line(
                    viz_frame,
                    pt_start,
                    pt_end,
                    (0, 255, 0),
                    2,
                    lineType=cv2.LINE_AA,
                )
    # return 0 if can't calculate
    if len(dash_lengths) == 0:
        return 0

    def remove_outliers_iqr(list):
        """Filters numeric outliers outside 1.5x the interquartile range (IQR)[cite: 1].

        Args:
            list: Sequence of numeric measurements[cite: 1].

        Returns:
            List of values falling within `[Q1 - 1.5 * IQR, Q3 + 1.5 * IQR]`[cite: 1].
        """
        q1 = np.percentile(list, 25)
        q3 = np.percentile(list, 75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        # Filter the list
        return [x for x in list if lower_bound <= x <= upper_bound]

    filtered_lengths = remove_outliers_iqr(dash_lengths)
    avg_length = sum(filtered_lengths) / float(len(filtered_lengths))

    ft_per_px = 10.0 / avg_length  # how many feet per pixel of the frame
    frame_length_px = bev_frame.shape[:2][0] / display_scale
    frame_length_ft = frame_length_px * ft_per_px  # length of frame in feet

    if save_images == True:
        cv2.imwrite(
            os.path.abspath(
                os.path.join(
                    basepath,
                    "..",
                    "screenshots",
                    "dash_length_estimation.png",
                )
            ),
            viz_frame,
        )
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return frame_length_ft