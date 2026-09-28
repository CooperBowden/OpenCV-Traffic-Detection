import json
import cv2
import numpy as np


def calibrate(
    frame: np.ndarray,
    camera: str,
    scale: float,
):
    def trackbar_event(val):
        pass

    def redraw():
        """Redraws all points and connecting lines on a fresh frame copy."""
        nonlocal frame_display
        frame_display = frame.copy()

        num_pts = len(clicked_points)
        if num_pts == 0:
            return

        # 1. Draw lines connecting consecutive points
        for i in range(num_pts - 1):
            pt1 = tuple((clicked_points[i] * scale).astype(int))
            pt2 = tuple((clicked_points[i + 1] * scale).astype(int))
            cv2.line(frame_display, pt1, pt2, (255, 0, 0), 2, cv2.LINE_AA)

        # 2. Close the box if all 4 points are set
        if num_pts == 4:
            pt_first = tuple((clicked_points[0] * scale).astype(int))
            pt_last = tuple((clicked_points[3] * scale).astype(int))
            cv2.line(
                frame_display, pt_last, pt_first, (255, 0, 0), 2, cv2.LINE_AA
            )

        # 3. Draw circles and text labels for every point
        for i, pt in enumerate(clicked_points):
            px, py = int(pt[0] * scale), int(pt[1] * scale)
            cv2.circle(frame_display, (px, py), 5, (0, 0, 255), -1)
            cv2.putText(
                frame_display,
                f"P{i + 1}: ({px}, {py})",
                (px + 10, py - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

    def click_event(event, x, y, flags, param):
        """Callback function to handle mouse clicks on the video frame."""
        nonlocal clicked_points
        if event == cv2.EVENT_LBUTTONDOWN:
            new_point = np.array([int(x / scale), int(y / scale)])

            if len(clicked_points) == 4:
                # Compute Euclidean distances and replace closest point
                distances_sq = np.sum((clicked_points - new_point) ** 2, axis=1)
                closest_idx = np.argmin(distances_sq)
                clicked_points[closest_idx] = new_point
            else:
                # Append new point
                if len(clicked_points) == 0:
                    clicked_points = new_point.reshape(1, 2)
                else:
                    clicked_points = np.vstack([clicked_points, new_point])

            redraw()

    try:
        with open("config.json", "r") as file:
            data = json.load(file)
    except FileNotFoundError:
        data = {}

    if camera not in data:
        data[camera] = {}

    # Initialize clicked_points array
    clicked_points = np.empty((0, 2), dtype=int)

    # Base frame copy for drawing
    frame_display = frame.copy()

    # Setup the OpenCV Window and Mouse Callback
    cv2.namedWindow("Select 4 Ground Points")
    cv2.setMouseCallback("Select 4 Ground Points", click_event)
    try:
        with open("config.json", "r") as file:
            data = json.load(file)
    except FileNotFoundError:
        data = {}
    # Trackbars
    cv2.createTrackbar(
        "Number of lanes", "Select 4 Ground Points", data[camera]["num_lanes"], 5, trackbar_event
    )
    cv2.createTrackbar(
        "Number of dashes", "Select 4 Ground Points", data[camera]["num_lanes"], 10, trackbar_event
    )

    print("INSTRUCTIONS:")
    print("Click 4 points on the ground plane in CLOCKWISE order:")
    print(
        "1. Top-Left  ->  2. Top-Right  ->  3. Bottom-Right  ->  4. Bottom-Left"
    )
    print("Clicking a 5th time replaces the closest point.")
    print("Press 'r' to reset points. Press 's' to save. Press 'q' to quit.")

    # Calibration loop
    while True:
        display_frame = frame_display.copy()

        num_lanes = cv2.getTrackbarPos(
            "Number of lanes", "Select 4 Ground Points"
        )
        num_dashes = cv2.getTrackbarPos(
            "Number of dashes", "Select 4 Ground Points"
        )

        # Draw trackbar info overlay (split across lines since cv2.putText doesn't handle \n)
        cv2.putText(
            display_frame,
            f"Number of lanes: {num_lanes}",
            (30, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            display_frame,
            f"Number of dashes: {num_dashes}",
            (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        cv2.imshow("Select 4 Ground Points", display_frame)
        key = cv2.waitKey(30) & 0xFF

        # Reset choices
        if key == ord("r"):
            clicked_points = np.empty((0, 2), dtype=int)
            redraw()
            print("Points reset. Start over from Top-Left.")

        # Quit without saving
        if key == ord("q"):
            cv2.destroyWindow("Select 4 Ground Points")
            return None

        # Save configuration
        if key == ord("s"):
            if len(clicked_points) < 4:
                print("Please select 4 points before saving.")
                continue

            print("Points saved:")
            print(f"src_pts = np.float32(\n{np.float32(clicked_points)}\n)")

            data[camera]["points"] = clicked_points.tolist()
            data[camera]["num_lanes"] = num_lanes
            data[camera]["num_dashes"] = num_dashes

            with open("config.json", "w") as file:
                json.dump(data, file, indent=4)

            cv2.destroyWindow("Select 4 Ground Points")
            return clicked_points, num_lanes, num_dashes