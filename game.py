"""Pure game logic (no pygame): beat-synced beatmap, hit detection, timing, scoring.

All times passed in are time.perf_counter() values; the core converts them to
song time (seconds since the music started).
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

from config import Config, Track

MISS, MEH, OK, GREAT = 0, 50, 100, 300
LABELS = {MISS: "MISS!", MEH: "GOOD!", OK: "GREAT!", GREAT: "PERFECT!"}   # judgement pop-ups


@dataclass
class Note:
    index: int
    x: float
    y: float
    hit_time: float            # song time of the beat this note sits on
    state: str = "pending"     # pending | hit | missed


@dataclass
class HitEvent:
    points: int                # MISS / MEH / OK / GREAT
    x: float
    y: float
    at: float                  # perf_counter time, for effects
    offset: float | None = None   # seconds early(-)/late(+); None for misses


def note_times(cfg: Config, track: Track, beats_per_note: int) -> list[float]:
    """RHYTHM GENERATOR: song times (seconds) at which notes must be hit.

    Default: every `beats_per_note`-th beat of the track's grid,
        t_k = offset + k * (60 / bpm) * beats_per_note
    If the track defines explicit `timestamps`, those are used instead.
    """
    if track.timestamps:
        limit = min(track.duration, cfg.max_song_seconds or track.duration)
        return [t + cfg.audio_offset_s for t in sorted(track.timestamps) if t <= limit]
    interval = 60.0 / track.bpm * beats_per_note
    anchor = track.offset + cfg.audio_offset_s
    start = max(cfg.lead_in_s, track.lead_in, cfg.approach_time + 0.5)
    end = min(track.duration, cfg.max_song_seconds or track.duration) - 1.5
    k = math.ceil((start - anchor) / interval)
    times = []
    while anchor + k * interval <= end:
        times.append(anchor + k * interval)
        k += 1
    return times


def generate_beatmap(cfg: Config, track: Track, beats_per_note: int,
                     rng: random.Random) -> list[Note]:
    x0, x1 = cfg.spawn_margin_x, cfg.screen_w - cfg.spawn_margin_x
    y0, y1 = cfg.spawn_margin_top, cfg.screen_h - cfg.spawn_margin_bottom
    prev = ((x0 + x1) / 2, (y0 + y1) / 2)
    notes: list[Note] = []
    for i, t in enumerate(note_times(cfg, track, beats_per_note)):
        for _ in range(60):   # rejection-sample: not too close to, not too far from, the last note
            x, y = rng.uniform(x0, x1), rng.uniform(y0, y1)
            if cfg.min_note_spacing <= math.hypot(x - prev[0], y - prev[1]) <= cfg.max_note_spacing:
                break
        notes.append(Note(i, x, y, t))
        prev = (x, y)
    return notes


class GameCore:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.rng = random.Random(cfg.rng_seed)
        self.track: Track | None = None
        self.notes: list[Note] = []
        self.t0: float | None = None
        self.w_perfect = self.w_great = self.w_good = self.w_touch = 0.0   # set in start()
        self._reset_stats()

    def _reset_stats(self) -> None:
        self.score = 0
        self.combo = 0
        self.max_combo = 0
        self.health = 1.0
        self.counts = {GREAT: 0, OK: 0, MEH: 0, MISS: 0}
        self.offsets: list[float] = []
        self._next = 0                 # index of the earliest pending note

    # -- lifecycle -------------------------------------------------------
    def start(self, now: float, track: Track, beats_per_note: int) -> None:
        """`now` must be the moment the music starts playing."""
        self.track = track
        self.notes = generate_beatmap(self.cfg, track, beats_per_note, self.rng)
        # Strict beat window: seconds = fraction of one beat of THIS song.
        beat = 60.0 / track.bpm
        self.w_perfect = self.cfg.window_beats_perfect * beat
        self.w_great = self.cfg.window_beats_great * beat
        self.w_good = self.cfg.window_beats_good * beat
        self.w_touch = self.cfg.touch_window_beats * beat
        self.t0 = now
        self._reset_stats()

    def song_time(self, now: float) -> float:
        return now - self.t0

    @property
    def end_time(self) -> float:
        """Song time at which the round is over."""
        return (self.notes[-1].hit_time if self.notes else 0.0) + 1.5

    def next_deadline(self) -> float:
        """Song time at which the next pending note turns into a MISS (inf if none left)."""
        if self._next >= len(self.notes):
            return math.inf
        return self.notes[self._next].hit_time + self.w_good

    @property
    def finished(self) -> bool:
        return self.t0 is not None and self._next >= len(self.notes)

    @property
    def accuracy(self) -> float:
        judged = sum(self.counts.values())
        if judged == 0:
            return 100.0
        pts = GREAT * self.counts[GREAT] + OK * self.counts[OK] + MEH * self.counts[MEH]
        return 100.0 * pts / (GREAT * judged)

    @property
    def mean_offset_ms(self) -> float:
        return 1000.0 * sum(self.offsets) / len(self.offsets) if self.offsets else 0.0

    # -- per-frame -------------------------------------------------------
    def update(self, now: float) -> list[HitEvent]:
        """Mark notes whose hit window has fully passed as missed."""
        t = self.song_time(now)
        events = []
        while self._next < len(self.notes):
            n = self.notes[self._next]
            if t <= n.hit_time + self.w_good:
                break
            n.state = "missed"
            self._next += 1
            self._register(MISS)
            events.append(HitEvent(MISS, n.x, n.y, now))
        return events

    def visible_notes(self, now: float) -> list[Note]:
        t = self.song_time(now)
        out = []
        for n in self.notes[self._next:]:
            if n.hit_time - self.cfg.approach_time > t:
                break
            out.append(n)
        return out

    def touch(self, pos: tuple[float, float], now: float) -> HitEvent | None:
        """Touch mode: the hand being on the target inside its window counts as a hit."""
        return self.click(pos, now, early_limit=self.w_touch)

    def click(self, pos: tuple[float, float], now: float,
              early_limit: float | None = None) -> HitEvent | None:
        """Judge a hit attempt at `pos`, made at time `now`.

        A hit registers only inside the note's ACTIVE RHYTHM WINDOW
        (hit_time - w_good .. hit_time + w_good, in seconds = fractions of a beat) and only on the *earliest*
        pending note (osu! note lock). Outside the window, or beside the note, nothing
        happens; a note whose window passes untouched becomes a MISS in update().
        """
        c = self.cfg
        if self._next >= len(self.notes):
            return None
        target = self.notes[self._next]
        dt = self.song_time(now) - target.hit_time
        if dt < -(self.w_good if early_limit is None else early_limit):
            return None                             # too early: window not open yet
        if math.hypot(pos[0] - target.x, pos[1] - target.y) > c.note_radius * c.hit_forgiveness:
            return None                             # clicked beside the note: no penalty
        adt = abs(dt)
        pts = GREAT if adt <= self.w_perfect else OK if adt <= self.w_great else MEH
        target.state = "hit"
        self._next += 1
        self.offsets.append(dt)
        self._register(pts)
        return HitEvent(pts, target.x, target.y, now, dt)

    # -- scoring ---------------------------------------------------------
    def _register(self, pts: int) -> None:
        self.counts[pts] += 1
        if pts == MISS:
            self.combo = 0
            self.health = max(0.0, self.health - self.cfg.health_loss)
        else:
            self.score += round(pts * (1 + min(self.combo, 40) * 0.05))  # combo multiplier
            self.combo += 1
            self.max_combo = max(self.max_combo, self.combo)
            self.health = min(1.0, self.health + self.cfg.health_gain)
