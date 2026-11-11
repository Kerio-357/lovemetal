"""Minimal stand-in for pygame so main.py / render.py can be smoke-tested on a machine
without a display or pygame installed. It checks our code paths, not pygame itself."""
import sys, types
from pathlib import Path
from unittest.mock import MagicMock


class Rect:
    def __init__(self, *a):
        if len(a) == 1: a = tuple(a[0]) if not isinstance(a[0], Rect) else (a[0].x, a[0].y, a[0].w, a[0].h)
        if len(a) == 2: a = (*a[0], *a[1])
        self.x, self.y, self.w, self.h = (int(v) for v in a)
    def __iter__(self): return iter((self.x, self.y, self.w, self.h))
    def __getitem__(self, i): return (self.x, self.y, self.w, self.h)[i]
    size = property(lambda s: (s.w, s.h)); topleft = property(lambda s: (s.x, s.y))
    right = property(lambda s: s.x + s.w); bottom = property(lambda s: s.y + s.h)
    centerx = property(lambda s: s.x + s.w // 2); centery = property(lambda s: s.y + s.h // 2)
    center = property(lambda s: (s.centerx, s.centery))
    def collidepoint(self, p): return self.x <= p[0] < self.right and self.y <= p[1] < self.bottom


class Surface:
    def __init__(self, size=(10, 10), flags=0): self.size = tuple(int(v) for v in size)
    def convert(self): return self
    convert_alpha = convert
    def fill(self, c, rect=None, special_flags=0): assert len(c) in (3, 4)
    def blit(self, src, dest): assert isinstance(src, Surface), src; float(dest[0]); float(dest[1])
    def set_alpha(self, a): assert 0 <= int(a) <= 255, a
    def get_width(self): return self.size[0]
    def get_height(self): return self.size[1]
    def get_rect(self, center=None, topleft=None):
        r = Rect(0, 0, *self.size)
        if center is not None: r.x, r.y = int(center[0] - r.w / 2), int(center[1] - r.h / 2)
        if topleft is not None: r.x, r.y = int(topleft[0]), int(topleft[1])
        return r


class Font:
    def render(self, txt, aa, color):
        assert isinstance(txt, str) and len(color) == 3 and all(0 <= c <= 255 for c in color), color
        return Surface((len(txt) * 10, 20))


def _color(c): assert len(c) in (3, 4) and all(0 <= v <= 255 for v in c), c


icons = []


def install(events_script, mouse_state):
    pg = types.ModuleType("pygame")
    pg.Rect, pg.Surface, pg.error = Rect, Surface, type("error", (Exception,), {})
    pg.SRCALPHA, pg.FULLSCREEN, pg.SCALED, pg.BLEND_RGBA_MULT = 1, 2, 4, 8
    pg.QUIT, pg.KEYDOWN, pg.MOUSEBUTTONDOWN = 100, 101, 102
    for i, k in enumerate("ESCAPE c SPACE UP DOWN w s TAB LEFT RIGHT RETURN KP_ENTER p h m LEFTBRACKET RIGHTBRACKET".split()):
        setattr(pg, "K_" + k, 200 + i)
    pg.init = lambda: None; pg.quit = lambda: None
    pg.display = types.SimpleNamespace(set_mode=lambda s, f=0: Surface(s), set_caption=lambda t: None, set_icon=lambda s: icons.append(s),
                                       flip=lambda: None)
    pg.font = types.SimpleNamespace(SysFont=lambda n, s, bold=False: Font(), Font=lambda n, s: Font())
    def circle(surf, color, center, radius, width=0):
        _color(color); assert isinstance(center[0], int) and isinstance(center[1], int) and radius >= 0
    def line(surf, color, a, b, width=1): _color(color); float(a[0]); float(b[1])
    def lines(surf, color, closed, pts, width=1): _color(color); assert len(pts) >= 2
    def rect(surf, color, r, width=0, border_radius=0): _color(color); Rect(r)
    def polygon(surf, color, pts): _color(color); assert len(pts) >= 3
    pg.draw = types.SimpleNamespace(circle=circle, line=line, lines=lines, rect=rect, polygon=polygon)
    def frombuffer(buf, size, fmt):
        assert len(buf) == size[0] * size[1] * len(fmt), (len(buf), size, fmt); return Surface(size)
    def load(path):
        assert Path(path).is_file(), path; return Surface((512, 512))
    pg.image = types.SimpleNamespace(frombuffer=frombuffer, load=load)
    pg.transform = types.SimpleNamespace(smoothscale=lambda s, size: Surface(size))
    it = iter(events_script)
    pg.event = types.SimpleNamespace(get=lambda: next(it, [types.SimpleNamespace(type=pg.QUIT)]))
    pg.mouse = types.SimpleNamespace(get_pos=lambda: mouse_state["pos"],
                                     get_pressed=lambda: (mouse_state["down"], False, False))
    pg.time = types.SimpleNamespace(Clock=lambda: types.SimpleNamespace(get_fps=lambda: 60.0, tick=lambda f: None))
    pg.mixer = MagicMock(); pg.mixer.music = MagicMock()
    sys.modules["pygame"] = pg
    return pg
