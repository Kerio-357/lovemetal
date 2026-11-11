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
    """Thumb-index pinch -> click on the press edge.

    Hysteresis (separate close/open thresholds) plus a short frame-confirmation
    window stop a noisy ratio from producing phantom double clicks.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self.pinched = False
        self._count = 0
        self._run_t = 0.0
        self._last_click = -1e9

    def update(self, ratio: float, t: float) -> tuple[bool, float | None]:
        c = self.cfg
        wants_flip = ratio > c.pinch_open_thresh if self.pinched else ratio < c.pinch_close_thresh
        click_t = None
        if wants_flip:
            if self._count == 0:
                self._run_t = t                    # timestamp of the first frame of the change
            self._count += 1
            if self._count >= c.pinch_min_frames:
                self.pinched = not self.pinched
                self._count = 0
                if self.pinched and t - self._last_click >= c.click_cooldown_s:
                    click_t = self._run_t
                    self._last_click = t
        else:
            self._count = 0
        return self.pinched, click_t


@dataclass
class CursorState:
    pos: tuple[float, float]
    visible: bool = False              # we have a position (hand seen at least once)
    tracking: bool = False             # hand is being tracked right now
    pinched: bool = False
    click_time: float | None = None    # perf_counter time of a click detected this update


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

    def update(self, state: HandState, now: float) -> CursorState:
        click_t = None
        if state.frame_id != self._last_id:          # only react to *new* camera frames
            self._last_id = state.frame_id
            if state.detected:
                self._pos = self.smoother.update(state.cursor_xy, state.t)
                self._pinched, click_t = self.gesture.update(state.pinch_ratio, state.t)
                self._last_seen = state.t
                self._has_pos = self._tracking = True

        if self._tracking and now - self._last_seen > self.cfg.lost_grace_s:
            self._tracking = False
            self._pinched = False
            self.smoother.reset()
            self.gesture.reset()
        return CursorState(self._pos, self._has_pos, self._tracking, self._pinched, click_t)
