"""Mouse stand-in for the hand tracker (`--mouse`), so you can develop and test the
game without a camera. Same interface as HandTracker. Left button = pinch."""
from __future__ import annotations

import time

import pygame

from hand_tracker import HandState


class MouseTracker:
    def __init__(self, cfg):
        self.cfg = cfg
        self._id = 0
        self.error = None

    def start(self):
        return self

    def stop(self):
        pass

    def get_state(self) -> HandState:
        self._id += 1
        x, y = pygame.mouse.get_pos()
        pressed = pygame.mouse.get_pressed()[0]
        return HandState(True, (float(x), float(y)), 0.0 if pressed else 1.0,
                         time.perf_counter(), self._id, None)
