"""
CCTV and Video Stream Ingestion Module.

Provides robust, thread-safe, and low-latency ingestion for:
- Live HLS HTTP streams (.m3u8) with chunk-boundary jitter smoothing
- RTSP / RTMP network feeds
- Local video files (.mp4, .avi, .mkv)
- Pre-configured CCTV camera registries (e.g. VDOT 511 traffic cameras)
"""

import os
import queue
import json
import sys
import threading
import time
from typing import Optional, Tuple, Union
import cv2
import numpy as np

basepath = os.path.dirname(__file__)
filepath = os.path.abspath(os.path.join(basepath, "..", "config.json"))
# Preset Traffic Cameras
with open(filepath, "r") as file:
    PRESET_CAMERAS = json.load(file)


class CCTVStream:
    """
    High-performance, threaded video ingestion engine with deep jitter buffering,
    adaptive clock synchronization, and automatic reconnection.
    """

    def __init__(
        self,
        source: Union[str, int],
        buffer_size: int = 150,
        prebuffer_frames: int = 40,
        reconnect_delay: float = 1.0,
    ):
        """
        Initialize the stream ingestor.

        :param source: URL string (.m3u8, rtsp), camera preset key, local video path, or webcam int ID.
        :param buffer_size: Maximum capacity of the sliding jitter buffer.
        :param prebuffer_frames: Initial frames to buffer before starting playback.
        :param reconnect_delay: Seconds to wait before attempting stream reconnection.
        """
        # Resolve camera presets
        if isinstance(source, str) and source.lower() in PRESET_CAMERAS:
            preset = PRESET_CAMERAS[source.lower()]
            self.source = preset["url"]
            self.display_name = preset["name"]
        else:
            self.source = source
            self.display_name = str(source)

        self.is_network_stream = isinstance(self.source, str) and (
            self.source.startswith("http://")
            or self.source.startswith("https://")
            or self.source.startswith("rtsp://")
        )

        self.buffer_size = buffer_size
        self.prebuffer_frames = prebuffer_frames
        self.reconnect_delay = reconnect_delay

        self.frame_queue = queue.Queue(maxsize=self.buffer_size)
        self.stopped = False
        self.last_valid_frame: Optional[np.ndarray] = None
        self.lock = threading.Lock()

        # Telemetry metrics
        self.frames_received = 0
        self.frames_dropped = 0
        self.reconnect_count = 0

        # Initialize capture backend
        self.cap: Optional[cv2.VideoCapture] = None
        self._init_capture()

        # Detect native properties
        cam_fps = self.cap.get(cv2.CAP_PROP_FPS) if self.cap else 15.0
        self.target_fps = cam_fps if (cam_fps and 5.0 <= cam_fps <= 60.0) else 15.0
        self.nominal_interval = 1.0 / self.target_fps

        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if self.cap else 0
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if self.cap else 0
        self.resolution: Tuple[int, int] = (width, height)

        # Start ingestion thread for network streams
        if self.is_network_stream:
            self.worker_thread = threading.Thread(target=self._ingest_worker, daemon=True)
            self.worker_thread.start()
            self._await_prebuffer()

    def _init_capture(self):
        """Initializes OpenCV VideoCapture with validation."""
        if self.is_network_stream:
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                "fflags;nobuffer|flags;low_delay|reorder_queue_size;0"
            )
            self.cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        else:
            # Check if local file exists before opening
            if isinstance(self.source, str) and not os.path.exists(self.source):
                print(f"[!] Error: Local file '{self.source}' not found at path: {os.path.abspath(self.source)}")
                self.cap = None
                return

            self.cap = cv2.VideoCapture(self.source)

        if not self.cap or not self.cap.isOpened():
            print(f"[!] Error: OpenCV failed to open source: {self.source}")

    def _await_prebuffer(self):
        """Pre-fills jitter queue to avoid chunk starvation on initial playback."""
        print(f"[*] Pre-buffering stream ({self.prebuffer_frames} frames @ {self.target_fps:.0f} FPS)...")
        for _ in range(int(self.prebuffer_frames * 1.5)):
            if self.frame_queue.qsize() >= self.prebuffer_frames or self.stopped:
                break
            time.sleep(0.08)

    def _ingest_worker(self):
        """Background thread continuously pulling frames from the network stream."""
        while not self.stopped:
            if not self.cap or not self.cap.isOpened():
                self._handle_reconnect()
                continue

            ret, frame = self.cap.read()
            if not ret or frame is None:
                time.sleep(0.05)
                continue

            self.frames_received += 1

            # Prevent queue blocking: drop oldest frame if saturated
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                    self.frames_dropped += 1
                except queue.Empty:
                    pass

            self.frame_queue.put(frame)
            with self.lock:
                self.last_valid_frame = frame

    def _handle_reconnect(self):
        """Handles automatic network reconnection upon packet loss or disconnection."""
        self.reconnect_count += 1
        print(f"[!] Stream disconnected. Attempting reconnect #{self.reconnect_count}...")
        time.sleep(self.reconnect_delay)
        try:
            if self.cap:
                self.cap.release()
        except Exception:
            pass
        self._init_capture()

    # In cctv_stream.py -> read()
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self.is_network_stream:
            try:
                frame = self.frame_queue.get(timeout=0.25)
                return True, frame
            except queue.Empty:
                with self.lock:
                    if self.last_valid_frame is not None:
                        return True, self.last_valid_frame.copy()
                return False, None
        else:
            if not self.cap or not self.cap.isOpened():
                return False, None

            ret, frame = self.cap.read()
            # Automatically loop local video when reaching End-Of-File (EOF)
            if not ret:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()

            return ret, frame

    def get_sleep_interval(self, elapsed_loop_time: float) -> int:
        """
        Computes dynamic sleep delay in milliseconds based on buffer state
        to maintain continuous, fluid playback without drift or stutter.
        """
        if not self.is_network_stream:
            delay = self.nominal_interval - elapsed_loop_time
            return max(1, int(delay * 1000))

        qsize = self.frame_queue.qsize()

        # Dynamic pace governor
        if qsize < 15:
            effective_interval = self.nominal_interval * 1.15  # Buffer low: ease rate
        elif qsize > 70:
            effective_interval = self.nominal_interval * 0.88  # Buffer high: catch up
        else:
            effective_interval = self.nominal_interval

        delay = effective_interval - elapsed_loop_time
        return max(1, int(delay * 1000))

    @property
    def buffer_level(self) -> int:
        """Returns the number of frames currently in the jitter buffer."""
        return self.frame_queue.qsize() if self.is_network_stream else 0

    def isOpened(self) -> bool:
        return self.cap.isOpened() if self.cap else False

    def release(self):
        """Stops background threads and releases camera capture resources."""
        self.stopped = True
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
