"""All pygame drawing in the Lovemetal gothic theme.

Palette, heartagram and gothic frame come from the lovemetal_ui.py concept.
Draw order each frame:  begin() -> menu / notes+HUD / results -> effects -> cursor.
"""
from __future__ import annotations

import math
import random
import time
from collections import deque
from pathlib import Path

import numpy as np
import pygame

from config import Config
from game import GREAT, MEH, MISS, OK, GameCore
from hand_tracker import HandState
from input_pipeline import CursorState

# ----------------------------------------------------------------------------- palette
COLOR_BG = (10, 7, 8)            # Obsidian Black
COLOR_PANEL = (43, 10, 17)       # Deep Crimson Panel
COLOR_SILVER = (192, 192, 200)   # Silver Metallic
COLOR_BLOOD = (230, 0, 38)       # Blood Red Accent
COLOR_WINE = (96, 10, 26)        # Dark Wine Fill
COLOR_WHITE = (240, 240, 245)
COLOR_DARK = (20, 14, 18)

HIT_COLORS = {GREAT: COLOR_WHITE, OK: COLOR_SILVER, MEH: (200, 120, 130), MISS: COLOR_BLOOD}


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def _lerp_color(a, b, k):
    return tuple(int(a[i] + (b[i] - a[i]) * k) for i in range(3))


# ----------------------------------------------------------------------------- logo asset
# assets/heartagram.png: white heartagram on a transparent background (brightness -> alpha).
LOGO_PATH = Path(__file__).parent / "assets" / "heartagram.png"
_logo_cache: list = []


def load_logo():
    """The white-on-transparent logo surface, or None if the file is missing."""
    if not _logo_cache:
        try:
            _logo_cache.append(pygame.image.load(str(LOGO_PATH)))
        except (pygame.error, FileNotFoundError):
            _logo_cache.append(None)
    return _logo_cache[0]


def tinted_logo(size: int, color, alpha: int = 255):
    """Logo scaled to size x size and recoloured (the artwork is white, so multiply)."""
    logo = load_logo()
    if logo is None:
        return None
    s = pygame.transform.smoothscale(logo, (int(size), int(size)))
    s.fill((*color, 255), special_flags=pygame.BLEND_RGBA_MULT)
    if alpha < 255:
        s.set_alpha(int(alpha))
    return s


def make_window_icon(size: int = 64):
    """Taskbar / window-title icon: blood-red logo on obsidian. Call before set_mode()."""
    logo = tinted_logo(size - 6, COLOR_BLOOD)
    if logo is None:
        return None
    icon = pygame.Surface((size, size))
    icon.fill(COLOR_BG)
    icon.blit(logo, (3, 3))
    return icon


def draw_logo(surface, center, size: int, color, alpha: int = 255) -> None:
    """Draw the logo; falls back to the procedural heartagram if the file is missing."""
    s = tinted_logo(size, color, alpha)
    if s is None:
        draw_heartagram(surface, center, int(size * 0.4), color, width=2)
    else:
        surface.blit(s, (int(center[0] - size / 2), int(center[1] - size / 2)))


# ----------------------------------------------------------------------------- shapes
def draw_heartagram(surface, center, radius, color, width=2):
    """Fallback only (used when assets/heartagram.png is missing). Gothic heartagram: inverted pentagram + two heart arcs inside a circle."""
    cx, cy = center
    points = []
    for i in range(5):
        angle = math.radians(90 + i * 72)
        points.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle)))
    star = [points[i] for i in (0, 2, 4, 1, 3, 0)]
    pygame.draw.lines(surface, color, False, star, width)
    pygame.draw.circle(surface, color, (int(cx), int(cy)), int(radius * 1.15), width)
    heart_r = radius * 0.45
    for sx in (-1, 1):
        pygame.draw.circle(surface, color, (int(cx + sx * heart_r), int(cy - radius * 0.4)),
                           int(heart_r), width)


def draw_gothic_frame(surface, rect, border_color, fill_color=None, fill_alpha=255):
    """Ornate rectangular panel with blood-red corner flourishes."""
    rect = pygame.Rect(rect)
    if fill_color is not None:
        if fill_alpha >= 255:
            pygame.draw.rect(surface, fill_color, rect, border_radius=4)
        else:
            panel = pygame.Surface(rect.size, pygame.SRCALPHA)
            panel.fill((*fill_color, int(fill_alpha)))
            surface.blit(panel, rect.topleft)
    pygame.draw.rect(surface, border_color, rect, width=2, border_radius=4)
    c = 12
    x, y, w, h = rect.x, rect.y, rect.w, rect.h
    for cx, cy in ((x, y), (x + w, y), (x, y + h), (x + w, y + h)):
        dx = c if cx == x else -c
        dy = c if cy == y else -c
        pygame.draw.line(surface, COLOR_BLOOD, (cx, cy), (cx + dx, cy), 2)
        pygame.draw.line(surface, COLOR_BLOOD, (cx, cy), (cx, cy + dy), 2)


# ----------------------------------------------------------------------------- renderer
class Renderer:
    def __init__(self, screen: pygame.Surface, cfg: Config):
        self.screen, self.cfg = screen, cfg
        w, h = cfg.screen_w, cfg.screen_h
        self.font_title = pygame.font.SysFont("georgia", 64, bold=True)
        self.font_header = pygame.font.SysFont("georgia", 28, bold=True)
        self.font_body = pygame.font.SysFont("georgia", 20)
        self.font_small = pygame.font.SysFont("georgia", 16)
        self.font_judge = pygame.font.SysFont("georgia", 40, bold=True)

        # Menu / results layout. Buttons are large on purpose: they are "pinched" by hand.
        self.track_rects = [pygame.Rect(150, 270 + i * 95, 540, 76) for i in range(3)]
        self.btn_start = pygame.Rect(800, 392, 330, 86)
        self.btn_pace = pygame.Rect(800, 496, 330, 64)
        self.btn_mode = pygame.Rect(800, 574, 330, 64)
        self.btn_return = pygame.Rect(w // 2 - 165, 570, 330, 84)
        self.hud_rect = pygame.Rect(50, 20, w - 100, 74)

        # Backgrounds
        self.bg_plain = self._make_plain_background()
        self.vignette = self._make_vignette()
        self.bg_cam = pygame.Surface((w, h)).convert()
        self._bg_frame_id = None
        self._have_cam = False

        self.note_sprite = self._make_note_sprite()
        self.trail: deque = deque(maxlen=12)
        self.popups: list[dict] = []         # floating judgement texts
        self.sparks: list[dict] = []         # crimson / silver sparks
        self.shatters: list[dict] = []       # expanding, fading heartagram "shatter" rings
        self.shatter_frames = self._make_shatter_frames()
        self.click_rings: list[tuple[float, float, float]] = []

    # -- prebuilt surfaces ---------------------------------------------
    def _make_plain_background(self) -> pygame.Surface:
        """Fallback when there is no camera picture (mouse mode, camera off/lost)."""
        w, h = self.cfg.screen_w, self.cfg.screen_h
        bg = pygame.Surface((w, h)).convert()
        for y in range(h):
            pygame.draw.line(bg, _lerp_color(COLOR_BG, (38, 8, 14), y / h), (0, y), (w, y))
        draw_logo(bg, (w // 2, h // 2 + 10), 560, (34, 10, 15))
        return bg

    def _make_vignette(self) -> pygame.Surface:
        """Dark corners so HUD and notes stay readable on top of the webcam picture."""
        w, h = self.cfg.screen_w, self.cfg.screen_h
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        d = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
        alpha = np.clip((d - 0.45) / 0.75, 0, 1) ** 1.5 * self.cfg.bg_vignette
        rgba = np.zeros((h, w, 4), np.uint8)
        rgba[..., 0], rgba[..., 1], rgba[..., 2] = COLOR_BG
        rgba[..., 3] = alpha.astype(np.uint8)
        return pygame.image.frombuffer(rgba.tobytes(), (w, h), "RGBA").convert_alpha()

    def _make_note_sprite(self) -> pygame.Surface:
        r = self.cfg.note_radius
        size = 2 * r + 8
        s = pygame.Surface((size, size), pygame.SRCALPHA)
        c = (size // 2, size // 2)
        pygame.draw.circle(s, (*COLOR_WINE, 215), c, r)
        if load_logo() is not None:
            draw_logo(s, c, 2 * r - 8, COLOR_WHITE)          # the logo's own ring is the target ring
            pygame.draw.circle(s, COLOR_SILVER, c, r, 3)
        else:
            pygame.draw.circle(s, COLOR_SILVER, c, r, 4)
            pygame.draw.circle(s, COLOR_BLOOD, c, r - 6, 2)
            draw_heartagram(s, (c[0], c[1] + 2), int(r * 0.55), COLOR_WHITE, width=2)
        return s

    def _make_shatter_frames(self) -> list:
        """Pre-scaled blood-red logos for the 'shattering heart' burst (cheap to draw later)."""
        r = self.cfg.note_radius
        frames = [tinted_logo(int(2 * r * k), COLOR_BLOOD) for k in (1.0, 1.25, 1.5, 1.8, 2.15, 2.5)]
        return [] if any(f is None for f in frames) else frames

    # -- background ------------------------------------------------------
    def begin(self, state: HandState, use_camera: bool = True) -> None:
        """Full-screen webcam picture (already tinted by the tracker) or the plain backdrop."""
        frame = state.frame if use_camera else None
        if frame is not None:
            if state.frame_id != self._bg_frame_id:          # convert only new camera frames
                self._bg_frame_id = state.frame_id
                h, w = frame.shape[:2]
                try:
                    cam = pygame.image.frombuffer(frame.tobytes(), (w, h), "RGB")
                    self.bg_cam.blit(cam, (0, 0))
                    self.bg_cam.blit(self.vignette, (0, 0))
                    self._have_cam = True
                except (ValueError, pygame.error):
                    self._have_cam = False
        else:
            self._have_cam = False
        self.screen.blit(self.bg_cam if self._have_cam else self.bg_plain, (0, 0))

    # -- helpers ---------------------------------------------------------
    def _alpha_circle(self, color, radius, alpha, pos, width=0) -> None:
        r = max(1, int(radius))
        s = pygame.Surface((2 * r + 4, 2 * r + 4), pygame.SRCALPHA)
        pygame.draw.circle(s, (*color, int(max(0, min(255, alpha)))), (r + 2, r + 2), r, width)
        self.screen.blit(s, (int(pos[0]) - r - 2, int(pos[1]) - r - 2))

    def _text(self, txt, font, color, pos, alpha=255, center=False, shadow=True) -> pygame.Rect:
        """Draw text (with a drop shadow so it stays readable over the webcam picture)."""
        surf = font.render(txt, True, color)
        rect = surf.get_rect(center=pos) if center else surf.get_rect(topleft=pos)
        if shadow:
            sh = font.render(txt, True, (0, 0, 0))
            if alpha < 255:
                sh.set_alpha(int(alpha))
            self.screen.blit(sh, (rect.x + 2, rect.y + 2))
        if alpha < 255:
            surf.set_alpha(int(alpha))
        self.screen.blit(surf, rect)
        return rect

    def _button(self, rect, label, hover, font=None, sub=None) -> None:
        fill = COLOR_BLOOD if hover else COLOR_PANEL
        draw_gothic_frame(self.screen, rect, COLOR_WHITE if hover else COLOR_SILVER, fill, 235)
        cy = rect.centery - (10 if sub else 0)
        self._text(label, font or self.font_header, COLOR_WHITE if hover else COLOR_SILVER,
                   (rect.centerx, cy), center=True)
        if sub:
            self._text(sub, self.font_small, COLOR_SILVER, (rect.centerx, rect.centery + 20), center=True)

    @staticmethod
    def _hover(rect, cur: CursorState) -> bool:
        return cur.visible and rect.collidepoint(cur.pos)

    # -- main menu -------------------------------------------------------
    def draw_menu(self, tracks, selected: int, pace_name: str, pace_beats: int,
                  cur: CursorState, hit_mode: str = "pinch", music: str = "") -> None:
        scr = self.screen
        pulse = 0.5 + 0.5 * math.sin(time.perf_counter() * 1.6)      # slow "breathing" glow
        draw_logo(scr, (965, 215), 300, COLOR_PANEL, 255)
        draw_logo(scr, (965, 215), 262, COLOR_BLOOD, int(170 + 85 * pulse))

        self._text("LOVEMETAL", self.font_title, COLOR_SILVER, (150, 60))
        self._text("- A GOTHIC RHYTHM ECLIPSE -", self.font_body, COLOR_BLOOD, (155, 140))
        pygame.draw.line(scr, COLOR_BLOOD, (150, 185), (690, 185), 2)
        pygame.draw.line(scr, COLOR_SILVER, (150, 189), (600, 189), 1)
        self._text("SELECT TRACK", self.font_header, COLOR_SILVER, (150, 222))

        for i, (track, rect) in enumerate(zip(tracks, self.track_rects)):
            sel, hov = i == selected, self._hover(rect, cur)
            fill = COLOR_PANEL if sel else COLOR_DARK
            border = COLOR_BLOOD if sel else COLOR_WHITE if hov else COLOR_SILVER
            draw_gothic_frame(scr, rect, border, fill, 235)
            if sel:                                           # selection marker
                x, y = rect.x + 20, rect.centery
                pygame.draw.polygon(scr, COLOR_BLOOD, [(x, y - 9), (x + 14, y), (x, y + 9)])
            self._text(track.title, self.font_header if sel else self.font_body,
                       COLOR_WHITE if sel or hov else COLOR_SILVER,
                       (rect.x + 48, rect.centery - (16 if sel else 12)), shadow=False)
            bpm = self.font_small.render(f"{track.bpm:.0f} BPM", True, COLOR_SILVER)
            scr.blit(bpm, (rect.right - bpm.get_width() - 18, rect.centery - 9))

        if music:
            self._text(music, self.font_small, COLOR_SILVER, (150, 556))

        self._button(self.btn_start, "START RITE", self._hover(self.btn_start, cur))
        self._button(self.btn_pace, f"PACE: {pace_name}", self._hover(self.btn_pace, cur),
                     sub=f"a note every {pace_beats} beat{'s' if pace_beats > 1 else ''}")
        self._button(self.btn_mode, f"HIT: {hit_mode.upper()}", self._hover(self.btn_mode, cur),
                     sub="pinch on the target" if hit_mode == "pinch" else "just touch the target")

        self._text("Pinch thumb + index to click   |   UP/DOWN track   ENTER start   TAB pace"
                   "   H hit mode   C camera   ESC quit", self.font_small, COLOR_SILVER,
                   (self.cfg.screen_w // 2, 692), center=True)

    # -- notes -----------------------------------------------------------
    def draw_notes(self, core: GameCore, now: float) -> None:
        c = self.cfg
        t = core.song_time(now)
        visible = core.visible_notes(now)
        half = self.note_sprite.get_width() // 2
        for k, n in reversed(list(enumerate(visible))):      # earliest note drawn on top
            age = t - (n.hit_time - c.approach_time)
            fade = _clamp01(age / 0.3)
            alpha = 255 * fade * (1.0 if k == 0 else 0.55)   # the next note to hit is brightest
            self.note_sprite.set_alpha(int(alpha))
            self.screen.blit(self.note_sprite, (int(n.x) - half, int(n.y) - half))
            # Approach ring: shrinks onto the note exactly on the beat.
            prog = _clamp01(age / c.approach_time)
            ring_r = c.note_radius * (1 + (c.approach_scale - 1) * (1 - prog))
            col = _lerp_color(COLOR_BLOOD, COLOR_WHITE, prog)
            if k == 0:
                pygame.draw.circle(self.screen, col, (int(n.x), int(n.y)), int(ring_r), 4)
            else:
                self._alpha_circle(col, ring_r, 120 * fade, (n.x, n.y), 3)

    # -- cursor ----------------------------------------------------------
    def draw_cursor(self, cur: CursorState) -> None:
        if not cur.visible:
            self.trail.clear()
            return
        c = self.cfg
        x, y = int(cur.pos[0]), int(cur.pos[1])
        if not cur.tracking:                       # hand lost: parked, dimmed cursor
            self.trail.clear()
            self._alpha_circle(COLOR_SILVER, c.cursor_radius, 130, cur.pos, 3)
            return
        self.trail.append(cur.pos)
        for i, p in enumerate(self.trail):
            k = (i + 1) / len(self.trail)
            self._alpha_circle(COLOR_BLOOD, c.cursor_radius * k * 0.7, 110 * k, p)
        r = c.cursor_radius
        if cur.pinched:
            pygame.draw.circle(self.screen, COLOR_BLOOD, (x, y), r)
            pygame.draw.circle(self.screen, COLOR_WHITE, (x, y), r, 3)
        else:
            pygame.draw.circle(self.screen, (0, 0, 0), (x, y), r + 2, 5)
            pygame.draw.circle(self.screen, COLOR_WHITE, (x, y), r, 3)
            pygame.draw.circle(self.screen, COLOR_BLOOD, (x, y), 5)

    # -- effects ---------------------------------------------------------
    def add_popup(self, text: str, x: float, y: float, color, now: float) -> None:
        """Floating judgement text (PERFECT! / GREAT! / MISS!) above the hit location."""
        self.popups.append({"txt": text, "x": x, "y": y - self.cfg.note_radius - 10,
                            "born": now, "col": color})

    def spawn_particles(self, x: float, y: float, now: float) -> None:
        """Crimson/silver spark explosion plus a shattering heartagram ring at (x, y)."""
        for _ in range(self.cfg.particle_count):
            a = random.uniform(0, math.tau)
            v = random.uniform(180, 620)
            self.sparks.append({"x": x, "y": y, "vx": math.cos(a) * v, "vy": math.sin(a) * v - 80,
                                "born": now, "life": random.uniform(0.35, 0.65),
                                "col": random.choice((COLOR_BLOOD, COLOR_BLOOD, COLOR_SILVER, COLOR_WHITE))})
        self.shatters.append({"x": x, "y": y, "born": now})

    def add_click_ring(self, pos, now) -> None:
        self.click_rings.append((pos[0], pos[1], now))

    def clear_effects(self) -> None:
        self.popups.clear()
        self.sparks.clear()
        self.shatters.clear()
        self.click_rings.clear()

    def draw_effects(self, now: float) -> None:
        # Shattering heart: the target's logo flies outwards and fades as the note disappears.
        life = 0.35
        self.shatters = [h for h in self.shatters if now - h["born"] < life]
        for h in self.shatters:
            k = (now - h["born"]) / life
            if self.shatter_frames:
                frame = self.shatter_frames[min(int(k * len(self.shatter_frames)), len(self.shatter_frames) - 1)]
                frame.set_alpha(int(230 * (1 - k)))
                self.screen.blit(frame, (int(h["x"] - frame.get_width() / 2),
                                         int(h["y"] - frame.get_height() / 2)))
            else:
                self._alpha_circle(COLOR_BLOOD, self.cfg.note_radius * (1 + 1.5 * k), 230 * (1 - k),
                                   (h["x"], h["y"]), 4)

        # Sparks: short streaks that slow down, fall slightly and fade.
        self.sparks = [p for p in self.sparks if now - p["born"] < p["life"]]
        for p in self.sparks:
            dt = now - p["born"]
            k = dt / p["life"]
            drag = 1 - 0.55 * k                                   # decelerate
            x = p["x"] + p["vx"] * dt * drag
            y = p["y"] + p["vy"] * dt * drag + 420 * dt * dt      # gravity
            tail = 0.035 * (1 - k)
            x0, y0 = x - p["vx"] * tail, y - p["vy"] * tail
            col = _lerp_color(p["col"], COLOR_WINE, k)
            pygame.draw.line(self.screen, col, (int(x0), int(y0)), (int(x), int(y)), 3 if k < 0.5 else 2)

        # Judgement pop-ups: quick rise + fade over cfg.popup_time (0.5 s).
        dur = self.cfg.popup_time
        self.popups = [t for t in self.popups if now - t["born"] < dur]
        for t in self.popups:
            k = (now - t["born"]) / dur
            rise = 46 * (1 - (1 - k) ** 2)                        # fast at first, then settles
            alpha = 255 if k < 0.6 else 255 * (1 - (k - 0.6) / 0.4)
            self._text(t["txt"], self.font_judge, t["col"], (t["x"], t["y"] - rise),
                       alpha=alpha, center=True)

        self.click_rings = [r for r in self.click_rings if now - r[2] < 0.3]
        for x, y, born in self.click_rings:
            k = (now - born) / 0.3
            self._alpha_circle(COLOR_WHITE, 20 + 40 * k, 220 * (1 - k), (x, y), 3)

    # -- in-game HUD -----------------------------------------------------
    def draw_hud(self, core: GameCore, now: float, beat: tuple | None = None) -> None:
        scr, hud = self.screen, self.hud_rect
        if beat is not None and beat[0] in ("stems", "duck"):    # adaptive-audio indicator
            what = "DRUMS" if beat[0] == "stems" else "MUSIC"
            alive = beat[1]
            pygame.draw.circle(scr, COLOR_WHITE if alive else COLOR_WINE, (hud.x + 12, hud.bottom + 18), 6)
            self._text(f"{what}: {'ALIVE' if alive else 'SILENT - hit the beat'}", self.font_small,
                       COLOR_SILVER if alive else COLOR_BLOOD, (hud.x + 26, hud.bottom + 8))
        draw_gothic_frame(scr, hud, COLOR_SILVER, COLOR_PANEL, 205)
        self._text(f"SCORE: {core.score:07d}", self.font_header, COLOR_SILVER, (hud.x + 30, hud.y + 21))
        self._text(f"COMBO: {core.combo}x", self.font_header, COLOR_BLOOD, (hud.x + 350, hud.y + 21))
        self._text(f"{core.accuracy:.1f}%", self.font_body, COLOR_SILVER, (hud.x + 570, hud.y + 27))

        # "SOUL RHYTHM" bar: fills on hits, drains on misses (visual only).
        bar = pygame.Rect(hud.right - 500, hud.y + 34, 470, 26)
        self._text("SOUL RHYTHM", self.font_small, COLOR_SILVER, (bar.x, hud.y + 9), shadow=False)
        draw_gothic_frame(scr, bar, COLOR_SILVER, (20, 5, 8))
        fill_w = int((bar.w - 4) * _clamp01(core.health))
        if fill_w > 0:
            pygame.draw.rect(scr, COLOR_WINE, (bar.x + 2, bar.y + 2, fill_w, bar.h - 4))
            pygame.draw.rect(scr, COLOR_BLOOD, (bar.x + 2, bar.y + 2, fill_w, 4))

        # Song progress line along the bottom edge of the HUD.
        t = core.song_time(now)
        prog = _clamp01(t / core.end_time) if core.end_time > 0 else 0
        pygame.draw.line(scr, COLOR_BLOOD, (hud.x + 2, hud.bottom - 3),
                         (hud.x + 2 + int((hud.w - 4) * prog), hud.bottom - 3), 3)

        # Track title: big while the intro plays, then small in the corner.
        if core.track is not None:
            if t < 2.5:
                a = 255 * _clamp01((2.5 - t) / 0.8)
                self._text(core.track.title, self.font_title, COLOR_SILVER,
                           (self.cfg.screen_w // 2, self.cfg.screen_h // 2 - 30), alpha=a, center=True)
                self._text("- GET READY -", self.font_header, COLOR_BLOOD,
                           (self.cfg.screen_w // 2, self.cfg.screen_h // 2 + 40), alpha=a, center=True)
            self._text(f"HIM - {core.track.title}   |   ESC = menu", self.font_small, COLOR_SILVER,
                       (20, self.cfg.screen_h - 28))

    # -- results ---------------------------------------------------------
    def draw_results(self, core: GameCore, cur: CursorState) -> None:
        cx = self.cfg.screen_w // 2
        panel = pygame.Rect(cx - 330, 60, 660, 480)
        draw_gothic_frame(self.screen, panel, COLOR_SILVER, COLOR_PANEL, 225)
        self._text("RITE COMPLETE", self.font_title, COLOR_SILVER, (cx, 120), center=True)
        if core.track is not None:
            self._text(core.track.title, self.font_body, COLOR_BLOOD, (cx, 172), center=True)
        pygame.draw.line(self.screen, COLOR_BLOOD, (panel.x + 60, 198), (panel.right - 60, 198), 2)
        rows = [("SCORE", f"{core.score:07d}"),
                ("ACCURACY", f"{core.accuracy:.2f}%"),
                ("MAX COMBO", f"{core.max_combo}x"),
                ("PERFECT / GREAT / GOOD / MISS", f"{core.counts[GREAT]} / {core.counts[OK]} / "
                                           f"{core.counts[MEH]} / {core.counts[MISS]}"),
                ("MEAN TIMING", f"{core.mean_offset_ms:+.0f} ms")]
        for i, (label, value) in enumerate(rows):
            y = 225 + i * 58
            self._text(label, self.font_body, COLOR_SILVER, (panel.x + 70, y + 6), shadow=False)
            val = self.font_header.render(value, True, COLOR_WHITE)
            self.screen.blit(val, (panel.right - 70 - val.get_width(), y))
        self._button(self.btn_return, "RETURN", self._hover(self.btn_return, cur))

    # -- status ----------------------------------------------------------
    def draw_status(self, msg: str, warn: bool = True) -> None:
        self._text(msg, self.font_body, (255, 170, 90) if warn else COLOR_SILVER,
                   (self.cfg.screen_w // 2, 655), center=True)

    def draw_fps(self, fps: float, infer_ms: float = 0.0) -> None:
        txt = f"{fps:.0f} fps" + (f" | model {infer_ms:.0f} ms" if infer_ms else "")
        s = self.font_small.render(txt, True, (150, 150, 160))
        self.screen.blit(s, (self.cfg.screen_w - s.get_width() - 14, self.cfg.screen_h - 26))
