"""Headless sanity tests (no camera, no pygame)."""
import dataclasses, math, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from config import CFG, TRACKS, PACES
from smoothing import EMAFilter, KalmanFilter2D
from input_pipeline import PinchClickDetector, CursorController
from hand_tracker import HandState, HandTracker
from game import GameCore, note_times, GREAT, OK, MEH, MISS


def test_smoothing_reduces_jitter():
    rng = np.random.default_rng(0)
    truth = [(600.0, 400.0)] * 200
    noisy = [(x + rng.normal(0, 6), y + rng.normal(0, 6)) for x, y in truth]
    for f in (EMAFilter(0.35), KalmanFilter2D(400, 25)):
        out = [f.update(p, i / 30) for i, p in enumerate(noisy)][50:]
        raw_std = np.std([p[0] for p in noisy[50:]])
        std = np.std([p[0] for p in out])
        assert std < 0.6 * raw_std, (type(f).__name__, std, raw_std)


def test_kalman_tracks_motion_with_low_lag():
    k, e = KalmanFilter2D(400, 25), EMAFilter(0.2)
    errs_k, errs_e = [], []
    for i in range(120):
        x = 10.0 * i                    # 300 px/s ramp at 30 fps
        pk, pe = k.update((x, 0), i / 30), e.update((x, 0), i / 30)
        if i > 30:
            errs_k.append(abs(pk[0] - x)); errs_e.append(abs(pe[0] - x))
    assert np.mean(errs_k) < np.mean(errs_e)


def test_pinch_hysteresis_and_single_click():
    d = PinchClickDetector(CFG)
    seq = [1.0, 1.0, 0.2, 0.2, 0.2, 0.35, 0.3, 0.2, 0.5, 0.5, 0.5]  # 0.35 is inside the dead band
    clicks, t = [], 0.0
    for r in seq:
        pinched, ct = d.update(r, t); t += 0.033
        if ct is not None: clicks.append(ct)
    assert len(clicks) == 1 and not pinched


def test_pinch_instant_borderline_and_sensitivity():
    d = PinchClickDetector(CFG)
    assert d.update(1.0, 0.0) == (False, None)
    assert d.update(0.10, 0.033) == (True, 0.033)            # clearly closed: fires on the first frame
    assert d.update(0.10, 0.066) == (True, None)             # held pinch: no repeat events
    assert d.update(0.35, 0.10) == (True, None)              # inside the hysteresis band: still pinched
    d.update(0.9, 0.2); d.update(0.9, 0.233)                 # opened
    assert not d.pinched
    assert d.update(0.25, 0.40) == (False, None)             # borderline: needs a second frame
    assert d.update(0.25, 0.433) == (True, 0.40)             # confirmed, stamped with the first frame
    d.update(0.9, 0.6); d.update(0.9, 0.633)
    assert d.update(0.25, 0.80)[1] is None and d.update(0.9, 0.833)[0] is False   # one-frame glitch ignored
    d.set_sensitivity(0.20)                                  # stricter: 0.25 no longer counts
    assert abs(d.open_thresh - 0.30) < 1e-9
    assert d.update(0.25, 1.0)[0] is False and d.update(0.25, 1.033)[0] is False
    d.set_sensitivity(5); assert d.close_thresh == 0.60


def test_cursor_prediction_follows_fast_hand():
    lag = {}
    for name, cfg in (("off", dataclasses.replace(CFG, cursor_predict_s=0.0)), ("on", CFG)):
        c = CursorController(cfg); errs = []
        for i in range(120):                                 # hand moving right at 900 px/s, 30 fps camera
            t = i / 30; st = HandState(True, (100 + 900 * t, 300.0), 1.0, t, i)
            for sub in (0.035, 0.035 + 1 / 60):              # result arrives 35 ms late, drawn at 60 fps
                cur = c.update(st, t + sub)
                if i > 30 and cur.pos[0] < CFG.screen_w - 2: errs.append(100 + 900 * (t + sub) - cur.pos[0])
        lag[name] = sum(errs) / len(errs)
    assert lag["on"] < 0.3 * lag["off"] and lag["off"] > 25, lag
    c = CursorController(CFG)                                # resting hand: no look-ahead, no drift
    for i in range(30): cur = c.update(HandState(True, (500.0, 300.0), 1.0, i / 30, i), i / 30 + 0.04)
    assert abs(cur.pos[0] - 500) < 0.5 and abs(cur.pos[1] - 300) < 0.5


def test_cursor_parks_on_loss():
    c = CursorController(CFG)
    cur = c.update(HandState(True, (100, 100), 1.0, 1.0, 1), 1.0)
    assert cur.visible and cur.tracking
    cur = c.update(HandState(False, t=2.0, frame_id=2), 2.0)
    assert cur.visible and not cur.tracking and cur.pos == (100, 100)


def test_normalize_to_screen_matches_fullscreen_picture():
    t = HandTracker(CFG)                         # 640x480 camera covering a 1280x720 screen
    assert t.cover_transform(640, 480) == (2.0, 0.0, 120.0)
    assert t.normalize_to_screen(0.5, 0.5) == (640.0, 360.0)
    assert t.normalize_to_screen(0.25, 0.5) == (320.0, 360.0)
    assert t.normalize_to_screen(0.0, 0.0) == (0.0, 0.0)            # cropped area -> clamped
    assert t.normalize_to_screen(1.0, 1.0) == (CFG.screen_w - 1.0, CFG.screen_h - 1.0)
    # a 16:9 camera maps 1:1
    assert t.normalize_to_screen(0.25, 0.25, 1280, 720) == (320.0, 180.0)


def test_background_frame():
    import numpy as np
    t = HandTracker(CFG)
    cam = np.zeros((480, 640, 3), np.uint8); cam[:, :320] = 200       # left half bright
    bg = t._background(cam)                                          # half-size picture, stretched later
    assert bg.shape == (360, 640, 3) and bg.dtype == np.uint8 and bg.flags["C_CONTIGUOUS"]
    assert bg[180, 500, 0] > bg[180, 100, 0]                         # mirrored: bright on the right
    assert bg[180, 500, 0] > bg[180, 500, 1]                         # red tint
    assert bg[180, 400, 0] > bg[10, 630, 0] + 20                     # vignette: corners darker
    full = HandTracker(dataclasses.replace(CFG, bg_downscale=1))._background(cam)
    assert full.shape == (720, 1280, 3)
    wide = np.zeros((720, 1280, 3), np.uint8)                        # 16:9 camera: no crop needed
    assert t._background(wide).shape == (360, 640, 3)


def test_knuckle_anchor_is_still_during_pinch():
    import numpy as np
    k_open = np.zeros((21, 2), np.float32); k_open[5] = (0.40, 0.50); k_open[9] = (0.46, 0.50)
    k_open[4] = (0.30, 0.40); k_open[8] = (0.42, 0.30)
    k_pinch = k_open.copy(); k_pinch[4] = (0.37, 0.36); k_pinch[8] = (0.38, 0.36)   # fingertips meet
    quest = HandTracker(CFG)
    assert CFG.cursor_anchor == "knuckles"
    assert quest._cursor_point(k_open) == quest._cursor_point(k_pinch)               # pointer does not move
    assert np.allclose(quest._cursor_point(k_open), (0.43, 0.50), atol=1e-6)
    tips = HandTracker(dataclasses.replace(CFG, cursor_anchor="pinch_mid"))
    assert tips._cursor_point(k_open) != tips._cursor_point(k_pinch)


TRACK = TRACKS[0]


def test_notes_sit_on_the_beat_grid():
    cfg = dataclasses.replace(CFG, rng_seed=1)
    for track in TRACKS:
        for _, beats in PACES:
            times = note_times(cfg, track, beats)
            beat = 60.0 / track.bpm
            assert len(times) > 10
            assert times[0] >= max(cfg.lead_in_s, track.lead_in) and times[-1] <= cfg.max_song_seconds
            for a, b in zip(times, times[1:]):
                assert abs((b - a) - beats * beat) < 1e-9
            for x in times:                                   # every note is a whole number of beats from the anchor
                k = (x - track.offset) / beat
                assert abs(k - round(k)) < 1e-6
    full = note_times(dataclasses.replace(cfg, max_song_seconds=None), TRACK, 2)
    assert full[-1] > 260


def test_game_judgement_and_scoring():
    cfg = dataclasses.replace(CFG, rng_seed=1)
    g = GameCore(cfg); g.start(100.0, TRACK, 2)
    n0, n1, n2 = g.notes[:3]
    assert g.click((n0.x, n0.y), 100.0 + n0.hit_time - 1.0) is None          # too early -> ignored
    ev = g.click((n0.x, n0.y), 100.0 + n0.hit_time + 0.03)
    assert ev.points == GREAT and g.combo == 1
    assert g.click((n1.x + 300, n1.y), 100.0 + n1.hit_time) is None and g.combo == 1   # beside the note
    edge = cfg.note_radius * cfg.hit_forgiveness - 1                         # forgiving hit radius
    ev = g.click((n1.x + edge, n1.y), 100.0 + n1.hit_time + 0.13)
    assert ev.points == OK
    h = g.health
    evs = g.update(100.0 + n2.hit_time + g.w_good + 0.01)               # never hit -> miss
    assert [e.points for e in evs] == [MISS] and g.combo == 0 and g.health < h
    assert g.max_combo == 2 and 0 < g.accuracy < 100 and not g.finished


def test_timing_window_is_strict():
    cfg = dataclasses.replace(CFG, rng_seed=2)
    g = GameCore(cfg); g.start(0.0, TRACK, 2)
    n = g.notes[0]
    assert g.click((n.x, n.y), n.hit_time - g.w_good - 0.01) is None          # pinch before the window
    assert g.click((n.x + 400, n.y), n.hit_time) is None                      # pinch beside the target
    assert n.state == "pending"
    assert g.click((n.x, n.y), n.hit_time + 0.02).points == GREAT             # on the beat -> PERFECT!
    n = g.notes[1]
    assert g.click((n.x, n.y), n.hit_time - g.w_great + 0.001).points == OK   # a bit early -> GREAT!
    n = g.notes[2]
    assert g.click((n.x, n.y), n.hit_time + g.w_good - 0.01).points == MEH    # late edge -> GOOD!
    n = g.notes[3]
    assert [e.points for e in g.update(n.hit_time + g.w_good + 0.01)] == [MISS]
    assert n.state == "missed" and not hasattr(g, "touch")                    # no hover/touch path exists
    from game import LABELS
    assert LABELS == {GREAT: "PERFECT!", OK: "GREAT!", MEH: "GOOD!", MISS: "MISS!"}


def test_windows_scale_with_bpm():
    g = GameCore(CFG)
    for track in TRACKS:
        g.start(0.0, track, 2)
        beat = 60.0 / track.bpm
        assert abs(g.w_perfect - CFG.window_beats_perfect * beat) < 1e-12
        assert g.w_perfect < g.w_great < g.w_good < beat / 2          # never overlaps the next beat
        assert g.next_deadline() == g.notes[0].hit_time + g.w_good
    fast, slow = GameCore(CFG), GameCore(CFG)
    fast.start(0.0, TRACKS[0], 2); slow.start(0.0, TRACKS[2], 2)
    assert fast.w_good < slow.w_good                                   # faster song -> tighter window


def test_explicit_timestamps_override_bpm_grid():
    tr = dataclasses.replace(TRACK, timestamps=(5.0, 3.5, 4.25, 500.0))
    assert note_times(CFG, tr, 2) == [3.5, 4.25, 5.0]


def test_hit_sound_asset():
    import wave
    with wave.open(str(Path(__file__).resolve().parents[1] / "assets" / "hit.wav")) as w:
        assert w.getframerate() == 44100 and 0.05 < w.getnframes() / w.getframerate() < 0.3


def test_note_lock_and_finish():
    cfg = dataclasses.replace(CFG, rng_seed=3)
    g = GameCore(cfg); g.start(0.0, TRACK, 4)
    n0, n1 = g.notes[:2]
    assert g.click((n1.x, n1.y), n1.hit_time - 0.3) is None and n1.state == "pending"
    g.update(g.notes[-1].hit_time + 1.0)
    assert g.finished and g.counts[MISS] == len(g.notes)


def test_beatmap_bounds_and_spacing():
    cfg = dataclasses.replace(CFG, rng_seed=7, max_song_seconds=None)
    g = GameCore(cfg); g.start(0.0, TRACK, 1)
    assert len(g.notes) > 400
    assert all(cfg.spawn_margin_x <= n.x <= cfg.screen_w - cfg.spawn_margin_x and
               cfg.spawn_margin_top <= n.y <= cfg.screen_h - cfg.spawn_margin_bottom for n in g.notes)
    d = [math.hypot(b.x - a.x, b.y - a.y) for a, b in zip(g.notes, g.notes[1:])]
    assert min(d) >= cfg.min_note_spacing and max(d) <= cfg.max_note_spacing


def test_logo_asset_is_transparent_png():
    import cv2
    img = cv2.imread(str(Path(__file__).resolve().parents[1] / "assets" / "heartagram.png"), cv2.IMREAD_UNCHANGED)
    assert img is not None and img.ndim == 3 and img.shape[2] == 4 and img.shape[0] == img.shape[1]
    a = img[:, :, 3]
    assert a[:6, :6].max() == 0 and a.max() == 255            # transparent corners, opaque strokes
    assert 0.15 < (a > 128).mean() < 0.5


def test_song_files_present():
    songs = Path(__file__).resolve().parents[1] / "songs"
    if not any(songs.glob("*.mp3")):
        print("   (skipped: no mp3 files in songs/ - they are not part of the repository)")
        return
    for t in TRACKS:
        assert (Path(__file__).resolve().parents[1] / "songs" / t.file).is_file(), t.file


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
