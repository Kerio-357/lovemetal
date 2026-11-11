"""Webcam capture + YOLOv8-pose hand tracking (best.pt) in a background thread.

Two threads keep input lag low:
  * grabber  - reads the camera as fast as it delivers and keeps ONLY the newest frame;
  * tracker  - runs YOLO on that newest frame. If the model is slower than the camera,
               the frames in between are skipped (frame skipping) instead of queueing
               up, so the cursor never falls behind real time.
The game loop never blocks on either: it calls `get_state()` and receives the newest result. Everything in the worker thread is wrapped so a bad
frame, a failed inference or a camera hiccup degrades to "no hand this frame"
instead of crashing the game.

Model contract (verified from best.pt): YOLOv8n-pose, 1 class ("hand"),
21 keypoints in MediaPipe order, each (x, y, confidence).

Each HandState also carries the webcam picture scaled to cover the whole screen
(mirrored and tinted), which the renderer uses as the game background.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from config import Config

# Keypoint indices (Ultralytics hand-keypoints / MediaPipe order)
WRIST, THUMB_TIP, INDEX_TIP, MIDDLE_MCP = 0, 4, 8, 9
NUM_KPTS = 21
REQUIRED_KPTS = (WRIST, THUMB_TIP, INDEX_TIP, MIDDLE_MCP)

# Skeleton edges for the on-screen hand overlay (pairs of keypoint indices)
HAND_EDGES = ((0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
              (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18),
              (18, 19), (19, 20), (0, 17))


@dataclass
class HandState:
    """One tracking result. Cursor coordinates are in *screen pixels*."""
    detected: bool = False
    cursor_xy: tuple[float, float] = (0.0, 0.0)   # raw (unsmoothed) cursor point
    pinch_ratio: float = 1.0                       # scale-invariant thumb-index distance
    t: float = 0.0                                 # time.perf_counter() at frame capture
    frame_id: int = -1
    frame: np.ndarray | None = None                # screen-sized RGB webcam picture (tinted)
    infer_ms: float = 0.0                          # smoothed model latency
    track_fps: float = 0.0                         # hand-tracking updates per second
    dropped: int = 0                               # camera frames skipped because the model was busy


def _np(x) -> np.ndarray:
    """torch.Tensor or ndarray -> ndarray."""
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


class HandTracker:
    """Captures frames, runs best.pt, maps the cursor point to screen pixels."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.error: str | None = None          # latest non-fatal problem, shown in the HUD
        self._cap: cv2.VideoCapture | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._state = HandState()
        self._running = False
        self._frame_id = 0
        self._ready = threading.Event()
        self._grabber: threading.Thread | None = None
        self._frame_cond = threading.Condition()
        self._latest: tuple | None = None      # (frame_bgr, capture_time, sequence_number)
        self._cam_lost = False
        self._init_error: Exception | None = None

    # -- lifecycle -------------------------------------------------------
    def start(self) -> "HandTracker":
        """Open camera, load model, start worker. Raises with a clear message on failure."""
        weights = self._resolve_weights()
        if not weights.is_file():
            raise FileNotFoundError(f"Model weights not found: {weights}\n"
                                    f"Put best.pt next to main.py or pass --weights PATH.")
        self._open_camera()
        self._running = True
        self._grabber = threading.Thread(target=self._grab_loop, name="camera-grabber", daemon=True)
        self._thread = threading.Thread(target=self._worker, args=(weights,),
                                        name="hand-tracker", daemon=True)
        self._grabber.start()
        self._thread.start()
        # Model is loaded inside the worker thread (same thread that runs inference).
        if not self._ready.wait(timeout=180):
            self.stop()
            raise TimeoutError("Timed out while loading the YOLO model.")
        if self._init_error is not None:
            self.stop()
            raise self._init_error
        return self

    def stop(self) -> None:
        self._running = False
        with self._frame_cond:
            self._frame_cond.notify_all()
        for th in (self._thread, self._grabber):
            if th and th.is_alive():
                th.join(timeout=2.0)
        if self._cap is not None:
            self._cap.release()

    def get_state(self) -> HandState:
        with self._lock:
            return self._state

    def _resolve_weights(self) -> Path:
        p = Path(self.cfg.weights).expanduser()
        return p if p.is_absolute() else Path(__file__).parent / p

    def _open_camera(self) -> None:
        cfg = self.cfg
        if self._cap is not None:
            self._cap.release()
        cap = cv2.VideoCapture(cfg.camera_index)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"Cannot open camera index {cfg.camera_index} "
                               f"(try --camera 1, and close other apps using the webcam)")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.cam_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.cam_h)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)        # always the freshest frame
        self._cap = cap

    # -- model -----------------------------------------------------------
    def _load_model(self, weights: Path):
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError("ultralytics is not installed. Run: pip install ultralytics") from e

        model = YOLO(str(weights))
        if getattr(model, "task", None) != "pose":
            raise ValueError(f"{weights.name} is a '{model.task}' model, expected a 'pose' model")
        kpt_shape = tuple(getattr(model.model, "kpt_shape", ()) or ())
        if not kpt_shape or kpt_shape[0] != NUM_KPTS:
            raise ValueError(f"Expected {NUM_KPTS} keypoints per hand, model has {kpt_shape}")
        # Warm-up: the first predict() builds the predictor and is slow; do it now.
        self._predict(model, np.zeros((self.cfg.cam_h, self.cfg.cam_w, 3), np.uint8))
        return model

    def _predict(self, model, frame_bgr: np.ndarray):
        c = self.cfg
        # imgsz: YOLO letterboxes the frame down to this size internally (320 by default),
        # so inference cost does not depend on the camera or screen resolution.
        return model.predict(frame_bgr, imgsz=c.yolo_imgsz, conf=c.yolo_conf, max_det=1,
                             device=c.yolo_device, half=c.yolo_half, verbose=False)[0]

    def _extract(self, res):
        """Ultralytics result -> (kpts_xy_norm (21,2) ndarray, box_xyxy_norm) or None."""
        if res.keypoints is None or res.boxes is None or len(res.boxes) == 0:
            return None
        i = int(np.argmax(_np(res.boxes.conf)))        # most confident hand
        xyn = _np(res.keypoints.xyn)[i]                # (21, 2), normalized to frame size
        kc = res.keypoints.conf
        kconf = _np(kc)[i] if kc is not None else np.ones(len(xyn))
        if len(xyn) < NUM_KPTS or any(kconf[k] < self.cfg.kpt_conf for k in REQUIRED_KPTS):
            return None                                # keypoints we need aren't trustworthy
        return xyn, _np(res.boxes.xyxyn)[i]

    # -- geometry --------------------------------------------------------
    def cover_transform(self, w: int, h: int) -> tuple[float, float, float]:
        """How a w x h camera frame is scaled/cropped to cover the screen.

        Returns (scale, off_x, off_y): screen_px = cam_px * scale - offset.
        """
        c = self.cfg
        scale = max(c.screen_w / w, c.screen_h / h)
        return scale, (w * scale - c.screen_w) / 2, (h * scale - c.screen_h) / 2

    def normalize_to_screen(self, nx: float, ny: float, w: int | None = None,
                            h: int | None = None) -> tuple[float, float]:
        """Camera-normalized (0..1) point -> screen pixels.

        Uses the same cover scaling as the full-screen background, so the cursor
        lands exactly on the hand in the picture. Result is clamped to the screen.
        """
        c = self.cfg
        w, h = w or c.cam_w, h or c.cam_h
        scale, ox, oy = self.cover_transform(w, h)
        x, y = nx * w * scale - ox, ny * h * scale - oy
        if c.active_margin_x > 0:        # optional: stretch the centre towards the edges
            x = (x / c.screen_w - c.active_margin_x) / (1 - 2 * c.active_margin_x) * c.screen_w
        if c.active_margin_y > 0:
            y = (y / c.screen_h - c.active_margin_y) / (1 - 2 * c.active_margin_y) * c.screen_h
        return min(max(x, 0.0), c.screen_w - 1.0), min(max(y, 0.0), c.screen_h - 1.0)

    def _background(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Camera frame -> screen-sized, mirrored, red-tinted RGB picture."""
        c = self.cfg
        h, w = frame_bgr.shape[:2]
        scale, ox, oy = self.cover_transform(w, h)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (max(c.screen_w, round(w * scale)), max(c.screen_h, round(h * scale))),
                          interpolation=cv2.INTER_LINEAR)
        ox, oy = int(ox), int(oy)
        gray = gray[oy:oy + c.screen_h, ox:ox + c.screen_w]
        if c.mirror:
            gray = cv2.flip(gray, 1)
        # Gothic filter: grayscale picture pushed into dark red.
        return cv2.merge([cv2.convertScaleAbs(gray, alpha=float(m)) for m in c.bg_tint])

    @staticmethod
    def pinch_ratio(kpts: np.ndarray, w: int, h: int) -> float:
        """|thumb tip - index tip| / palm size, measured in camera pixels."""
        px = kpts * np.array([w, h])
        palm = np.linalg.norm(px[WRIST] - px[MIDDLE_MCP]) + 1e-6
        return float(np.linalg.norm(px[THUMB_TIP] - px[INDEX_TIP]) / palm)

    def _cursor_point(self, kpts: np.ndarray) -> tuple[float, float]:
        if self.cfg.cursor_anchor == "index_tip":
            return float(kpts[INDEX_TIP][0]), float(kpts[INDEX_TIP][1])
        mid = (kpts[THUMB_TIP] + kpts[INDEX_TIP]) / 2   # barely moves while pinching
        return float(mid[0]), float(mid[1])

    # -- worker ----------------------------------------------------------
    def _publish(self, state: HandState) -> None:
        with self._lock:
            self._state = state

    def _grab_loop(self) -> None:
        """Camera thread: always hold the newest frame, drop everything older."""
        fails, last_reopen, seq = 0, 0.0, 0
        while self._running:
            try:
                ok, frame = self._cap.read()
            except Exception:  # noqa: BLE001
                ok, frame = False, None
            if ok and frame is not None:
                fails, self._cam_lost = 0, False
                seq += 1
                with self._frame_cond:
                    self._latest = (frame, time.perf_counter(), seq)   # capture time stamps the pinch
                    self._frame_cond.notify()
                continue
            fails += 1
            time.sleep(0.01)
            if fails >= 30:                           # ~0.3 s of nothing: camera is gone
                self._cam_lost = True
                if time.perf_counter() - last_reopen > 1.0:
                    last_reopen = time.perf_counter()
                    try:
                        self._open_camera()
                    except Exception:  # noqa: BLE001
                        pass

    def _worker(self, weights: Path) -> None:
        try:
            model = self._load_model(weights)
        except Exception as e:  # noqa: BLE001 - reported to start()
            self._init_error = e
            self._ready.set()
            return
        self._ready.set()

        cfg = self.cfg
        infer_ms, last_seq, dropped = 0.0, 0, 0
        fps, last_done = 0.0, None
        while self._running:
            # Wait for a frame newer than the one processed last time.
            with self._frame_cond:
                self._frame_cond.wait_for(
                    lambda: not self._running or (self._latest is not None and self._latest[2] != last_seq),
                    timeout=0.25)
                latest = self._latest
            if not self._running:
                break
            if latest is None or latest[2] == last_seq:
                if self._cam_lost:                    # no picture: report "no hand", keep the game alive
                    self.error = "Camera disconnected - retrying..."
                    self._frame_id += 1
                    self._publish(HandState(False, t=time.perf_counter(), frame_id=self._frame_id,
                                            infer_ms=infer_ms, dropped=dropped))
                continue
            frame, t, seq = latest
            if last_seq:
                dropped += seq - last_seq - 1         # frames that arrived while the model was busy
            last_seq = seq
            self._frame_id += 1

            bg = None
            if cfg.camera_background:
                try:
                    bg = self._background(frame)
                except Exception:  # noqa: BLE001 - background is cosmetic, never fatal
                    bg = None

            state = HandState(False, t=t, frame_id=self._frame_id, frame=bg, infer_ms=infer_ms,
                              track_fps=fps, dropped=dropped)
            try:
                t0 = time.perf_counter()
                res = self._predict(model, frame)     # raw (un-mirrored) frame: matches training data
                infer_ms = 0.9 * infer_ms + 0.1 * (time.perf_counter() - t0) * 1000 \
                    if infer_ms else (time.perf_counter() - t0) * 1000
                found = self._extract(res)
                self.error = None
                if found is not None:
                    kpts, box = found
                    h, w = frame.shape[:2]
                    ratio = self.pinch_ratio(kpts, w, h)
                    if cfg.mirror:                    # mirror the coordinates, not the pixels
                        kpts = kpts.copy(); kpts[:, 0] = 1.0 - kpts[:, 0]
                        box = np.array([1.0 - box[2], box[1], 1.0 - box[0], box[3]])
                    nx, ny = self._cursor_point(kpts)
                    if bg is not None and cfg.draw_hand_skeleton:
                        self._draw_skeleton(bg, kpts, w, h, ratio)
                    state = HandState(True, self.normalize_to_screen(nx, ny, w, h), ratio, t,
                                      self._frame_id, bg, infer_ms, fps, dropped)
                else:
                    state.infer_ms = infer_ms
            except Exception as e:  # noqa: BLE001 - never let one bad frame kill tracking
                self.error = f"Tracking error: {type(e).__name__}: {e}"[:120]
                time.sleep(0.02)
            done = time.perf_counter()
            if last_done is not None and done > last_done:
                inst = 1.0 / (done - last_done)
                fps = inst if fps == 0.0 else 0.9 * fps + 0.1 * inst
            last_done = done
            self._publish(state)

    def _draw_skeleton(self, bg: np.ndarray, kpts: np.ndarray, w: int, h: int, ratio: float) -> None:
        """Draw the 21 detected keypoints on the full-screen picture (already mirrored coords)."""
        scale, ox, oy = self.cover_transform(w, h)
        pts = [(int(x * w * scale - ox), int(y * h * scale - oy)) for x, y in kpts]
        for a, b in HAND_EDGES:
            cv2.line(bg, pts[a], pts[b], (150, 150, 160), 2, cv2.LINE_AA)
        for p in pts:
            cv2.circle(bg, p, 4, (225, 225, 235), -1, cv2.LINE_AA)
        pinched = ratio < self.cfg.pinch_close_thresh
        cv2.line(bg, pts[THUMB_TIP], pts[INDEX_TIP], (255, 255, 255) if pinched else (230, 0, 38),
                 3, cv2.LINE_AA)
