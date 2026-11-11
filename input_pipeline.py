"""Turns raw HandState samples into a stable cursor + click events.

    HandState -> smoothing filter   -> CursorState.pos
    HandState -> pinch hysteresis   -> CursorState.pinched / click_time
"""
from __future__ import annotations

from dataclasses import dataclass

from config import Config
from hand_tracker import HandState
from smoothing import make_smoother


class PinchClickDetector:
    """Thumb-index PINCH -> one click event at the moment the fingers close.

    * Input: pinch_ratio = fingertip distance / palm size (see config.py).
    * Hysteresis: closes below `pinch_close_thresh`, re-opens only above
      `pinch_open_thresh`, so a ratio hovering around one threshold cannot flicker.
    * A clearly closed pinch (ratio < `pinch_instant_ratio`) is accepted on its first
      camera frame; a borderline one must persist for `pinch_min_frames` frames.
    * Only the closing EDGE fires. Holding the fingers together and sweeping over
      notes does nothing - every note needs its own pinch.
    """

    def __init__(self, cfg: Config):
        self.close_thresh = cfg.pinch_close_thresh
        self.open_thresh = cfg.pinch_open_thresh
        self.cfg = cfg
        self.reset()

    def set_sensitivity(self, close_thresh: float) -> None:
        """Live tuning ([ and ] keys): open threshold follows at 1.5x."""
        self.close_thresh = min(max(close_thresh, 0.08), 0.60)
        self.open_thresh = self.close_thresh * 1.5

    def reset(self) -> None:
        self.pinched = False
        self._count = 0
        self._run_t = 0.0
        self._last_click = -1e9

    def update(self, ratio: float, t: float) -> tuple[bool, float | None]:
        """Feed one camera frame. Returns (pinched, click_time or None)."""
        c = self.cfg
        wants_flip = ratio > self.open_thresh if self.pinched else ratio < self.close_thresh
        click_t = None
        if wants_flip:
            if self._count == 0:
                self._run_t = t                    # capture time of the first frame of the change
            self._count += 1
            instant = not self.pinched and ratio < min(c.pinch_instant_ratio, self.close_thresh)
            if instant or self._count >= c.pinch_min_frames:
                self.pinched = not self.pinched
                self._count = 0
                if self.pinched and t - self._last_click >= c.click_cooldown_s:
                    click_t = self._run_t          # the pinch happened when that frame was captured
                    self._last_click = t
        else:
            self._count = 0
        return self.pinched, click_t


@dataclass
class CursorState:
    pos: tuple[float, float]
    visible: bool = False              # we have a position (hand seen at least once)
    tracking: bool = False             # hand is being tracked right now
    pinched: bool = False              # fingers are currently held together
    click_time: float | None = None    # perf_counter time of a pinch detected this update
    pinch_ratio: float = 1.0           # latest raw ratio (for the on-screen pinch meter)


class CursorController:
    """Smoothed cursor + click events.

    If the hand disappears the cursor stays parked at its last smoothed position
    (no crash, no jump to a corner). Clicks are impossible while tracking is lost.
    After `lost_grace_s` the filters are re-seeded so the cursor snaps to the hand
    when it comes back instead of gliding from stale velocity.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.smoother = make_smoother(cfg)
        self.gesture = PinchClickDetector(cfg)
        self._pos = (cfg.screen_w / 2, cfg.screen_h / 2)
        self._has_pos = False
        self._last_id = -1
        self._last_seen = -1e9
        self._pinched = False
        self._tracking = False
        self._ratio = 1.0

    def update(self, state: HandState, now: float) -> CursorState:
        click_t = None
        if state.frame_id != self._last_id:          # only react to *new* camera frames
            self._last_id = state.frame_id
            if state.detected:
                self._pos = self.smoother.update(state.cursor_xy, state.t)
                self._pinched, click_t = self.gesture.update(state.pinch_ratio, state.t)
                self._ratio = state.pinch_ratio
                self._last_seen = state.t
                self._has_pos = self._tracking = True

        if self._tracking and now - self._last_seen > self.cfg.lost_grace_s:
            self._tracking = False
            self._pinched = False
            self.smoother.reset()
            self.gesture.reset()
        return CursorState(self._predicted(now), self._has_pos, self._tracking, self._pinched,
                           click_t, self._ratio)

    def _predicted(self, now: float) -> tuple[float, float]:
        """Filtered position extrapolated to `now` along the filter's velocity.

        The newest camera frame is already (camera + model latency) old when it gets
        here; looking ahead by that age puts the cursor where the hand is now and lets
        it glide at render fps between camera frames. Fades out for a resting hand.
        """
        c = self.cfg
        vx, vy = getattr(self.smoother, "velocity", (0.0, 0.0))
        if not self._tracking or c.cursor_predict_s <= 0 or (vx == 0.0 and vy == 0.0):
            return self._pos
        speed = (vx * vx + vy * vy) ** 0.5
        ahead = min(max(now - self._last_seen, 0.0), c.cursor_predict_s)
        ahead *= min(1.0, speed / max(c.cursor_predict_min_speed, 1e-6))
        x = min(max(self._pos[0] + vx * ahead, 0.0), c.screen_w - 1.0)
        y = min(max(self._pos[1] + vy * ahead, 0.0), c.screen_h - 1.0)
        return x, y
