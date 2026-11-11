"""Drives main.main() through menu -> play -> results -> menu with a fake pygame,
in --mouse mode and with a (fake) camera frame as background."""
import sys, time, types
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import fake_pygame

# What the engine should pick depends on which audio files are present (none in a fresh clone).
EXPECTED_MODE = ("stems" if (ROOT / "songs" / "stems" / "bury_me_deep_inside_drums.ogg").is_file()
                 else "duck" if (ROOT / "songs" / "bury_me_deep_inside.mp3").is_file() else "off")


def ev(pg, key): return types.SimpleNamespace(type=pg.KEYDOWN, key=key)


def run(script_builder, argv, patch=None):
    mouse = {"pos": (0, 0), "down": False}
    for m in ("main", "render", "mouse_input"): sys.modules.pop(m, None)
    pg = fake_pygame.install([], mouse)
    import main, config, dataclasses
    # short round: 4 s of song, quick approach
    main.CFG = dataclasses.replace(config.CFG, max_song_seconds=14.0, lead_in_s=1.0, approach_time=0.6,
                                   rng_seed=5)
    frames = script_builder(pg, mouse, main)
    it = iter(frames)
    def get():
        try: f = next(it)
        except StopIteration: return [types.SimpleNamespace(type=pg.QUIT)]
        return f() if callable(f) else f
    pg.event.get = get
    pg.time.Clock = lambda: types.SimpleNamespace(get_fps=lambda: 60.0, tick=lambda f: time.sleep(0.004))
    if patch: patch(main)
    sys.argv = ["main.py", *argv]
    main.main()
    return pg, main


def test_full_flow_mouse_mode():
    log = {}
    def build(pg, mouse, main):
        import render
        orig_menu, orig_res, orig_notes = render.Renderer.draw_menu, render.Renderer.draw_results, render.Renderer.draw_notes
        def menu(self, tracks, sel, pn, pb, cur, hm="pinch", music=""): log.setdefault("menu", []).append((sel, pn, hm, music)); return orig_menu(self, tracks, sel, pn, pb, cur, hm, music)
        def res(self, core, cur): log["results"] = (core.score, dict(core.counts), core.track.title); return orig_res(self, core, cur)
        def notes(self, core, now):
            log["played"] = True
            vis = core.visible_notes(now)          # autoplay: hover the next note, click on the beat
            if vis:
                mouse["pos"] = (vis[0].x, vis[0].y)
                mouse["down"] = core.song_time(now) >= vis[0].hit_time
            return orig_notes(self, core, now)
        render.Renderer.draw_menu, render.Renderer.draw_results, render.Renderer.draw_notes = menu, res, notes
        def click(pos, down):
            def f(): mouse["pos"], mouse["down"] = pos, down; return []
            return f
        s = [[], [ev(pg, pg.K_DOWN)], [ev(pg, pg.K_TAB)], []]
        def press(pos): return [click(pos, False)] * 60 + [click(pos, True)] * 4 + [click(pos, False)] * 40
        s += press((400, 500)) + press((900, 525)) + press((900, 430))   # 3rd track, pace, START RITE
        s += [[]] * 4200                                                                     # play ~10 s
        return s
    def patch(main):
        orig = main.AdaptiveAudio.start
        def sm(self, track, wanted): log["music"] = track.file; r = orig(self, track, wanted); log["mode"] = self.mode; return r
        main.AdaptiveAudio.start = sm
        vol = []; log["drum_volumes"] = vol
        orig_up = main.AdaptiveAudio.update
        def up(self, t, dt): orig_up(self, t, dt); vol.append(round(self.gain, 2))
        main.AdaptiveAudio.update = up
        for name in ("play_hit_sound", "spawn_hit_particles", "show_score_popup"):   # count feedback calls
            def wrap(*a, _o=getattr(main, name), _n=name): log.setdefault(_n, []).append(a); return _o(*a)
            setattr(main, name, wrap)
    pg, main = run(build, ["--mouse"], patch)
    sels = [m[0] for m in log["menu"]]
    assert 1 in sels and sels[-1] == 2, sorted(set(sels))                    # keyboard moved to 1, click chose 2
    paces = [m[1] for m in log["menu"]]
    assert paces[0] == "NORMAL" and "FAST" in paces and paces[-1] == "SLOW", set(paces)
    assert log["music"] == "bury_me_deep_inside.mp3"
    assert log["menu"][0][3].startswith("MUSIC: ")
    assert log.get("played") and "results" in log, log.keys()
    score, counts, title = log["results"]
    assert title == "Bury Me Deep Inside" and score > 0 and counts[0] == 0, log["results"]
    assert log["mode"] == EXPECTED_MODE
    if EXPECTED_MODE == "stems":                                 # both stems loaded, song file not streamed
        loaded = [c.args[0] for c in pg.mixer.Sound.call_args_list]
        assert any(p.endswith("bury_me_deep_inside_backing.ogg") for p in loaded)
        assert any(p.endswith("bury_me_deep_inside_drums.ogg") for p in loaded)
        assert pg.mixer.music.load.call_count == 0
    v = log["drum_volumes"]
    assert v[0] == 0.0 and max(v) == 1.0 and v[-1] == 1.0       # silent at first, alive after hits
    # every hit -> one sound, one particle burst, one "PERFECT!" pop-up at the note position
    assert len(log["play_hit_sound"]) == len(log["spawn_hit_particles"]) == counts[300] == 3
    assert [a[0] for a in log["show_score_popup"]] == ["PERFECT!"] * 3
    assert any(c.args[0].endswith("hit.wav") for c in pg.mixer.Sound.call_args_list)
    assert pg.mixer.Sound.return_value.play.call_count == 3
    print("   results:", log["results"])


def test_touch_mode_and_miss_popups():
    """--hit-mode touch: hovering the target on the beat hits it without any click;
    notes that are skipped give MISS! pop-ups and no sound."""
    log = {"popups": []}
    def build(pg, mouse, main):
        import render
        orig_notes = render.Renderer.draw_notes
        def notes(self, core, now):
            vis = core.visible_notes(now)
            if vis and vis[0].index == 0:                      # only chase the first note ...
                on_beat = core.song_time(now) >= vis[0].hit_time
                mouse["pos"] = (vis[0].x, vis[0].y) if on_beat else (5, 5)
            else:
                mouse["pos"] = (5, 5)                          # ... and skip the rest
            return orig_notes(self, core, now)
        render.Renderer.draw_notes = notes
        return [[]] * 3 + [[ev(pg, pg.K_RETURN)]] + [[]] * 4200
    def patch(main):
        o = main.show_score_popup
        def pop(text, x, y): log["popups"].append(text); return o(text, x, y)
        main.show_score_popup = pop
    pg, main = run(build, ["--mouse", "--hit-mode", "touch"], patch)
    assert log["popups"][0] == "PERFECT!" and set(log["popups"][1:]) == {"MISS!"}, log["popups"]
    assert pg.mixer.Sound.return_value.play.call_count == 1


def make_engine(tmp_stems=None, **cfg_over):
    """AdaptiveAudio wired to recording fake channels."""
    from unittest.mock import MagicMock
    import dataclasses, config
    for m in ("main", "render", "mouse_input"): sys.modules.pop(m, None)
    pg = fake_pygame.install([], {"pos": (0, 0), "down": False})
    chans = {0: MagicMock(name="backing"), 1: MagicMock(name="drums")}
    pg.mixer.Channel = lambda i: chans[i]
    import main
    if tmp_stems is not None: main.STEMS_DIR = tmp_stems
    eng = main.AdaptiveAudio(dataclasses.replace(config.CFG, music_volume=1.0, **cfg_over), True)
    return pg, main, eng, chans


def run_engine(eng, t0, t1, fps=60):
    t = t0
    while t < t1 - 1e-9:
        t += 1 / fps; eng.update(t, 1 / fps)


def test_stems_engine_drums_follow_hits():
    import tempfile, config
    from game import GREAT, MEH
    with tempfile.TemporaryDirectory() as d:
        track = config.TRACKS[0]
        for suffix in ("_backing.ogg", "_drums.ogg"): (Path(d) / (Path(track.file).stem + suffix)).write_bytes(b"x")
        pg, main, eng, ch = make_engine(Path(d))
        assert eng.resolve_mode(track, "auto") == "stems" and eng.resolve_mode(config.TRACKS[1], "auto") == "duck"
        assert eng.resolve_mode(config.TRACKS[1], "stems") == "duck" and eng.resolve_mode(track, "off") == "off"
        assert eng.start(track, "auto") is None and eng.mode == "stems"
        pg.mixer.set_reserved.assert_called_with(2)
        ch[0].play.assert_called_once(); ch[1].play.assert_called_once()
        assert ch[1].set_volume.call_args[0][0] == 0.0 and ch[0].set_volume.call_args[0][0] == 1.0
        run_engine(eng, 0, 3); assert ch[1].set_volume.call_args[0][0] == 0.0 and not eng.beat_alive   # no hits: silent beat
        eng.on_hit(GREAT, open_until=4.5)                        # PERFECT hit at t=3
        run_engine(eng, 3, 3.1); assert ch[1].set_volume.call_args[0][0] == 1.0 and eng.beat_alive   # fast attack
        run_engine(eng, 3.1, 4.4); assert ch[1].set_volume.call_args[0][0] == 1.0                    # held for the note
        eng.on_hit(MEH, open_until=5.5); run_engine(eng, 4.4, 4.8)
        assert abs(ch[1].set_volume.call_args[0][0] - 0.6) < 1e-6                                    # GOOD hit: quieter drums
        eng.on_miss(5.5); run_engine(eng, 5.5, 5.8)
        assert ch[1].set_volume.call_args[0][0] == 0.0                                               # miss: beat gone
        eng.on_hit(GREAT, open_until=7.0); run_engine(eng, 5.8, 6.0); eng_gain = eng.gain
        run_engine(eng, 6.0, 7.5)                                # next note never hit and no miss event yet
        assert eng_gain == 1.0 and ch[1].set_volume.call_args[0][0] == 0.0                           # gate closes on its own
        assert all(c.args[0] == 1.0 for c in ch[0].set_volume.call_args_list)                        # backing never ducked
        assert pg.mixer.music.play.call_count == 0
        eng.stop(fade_ms=1500); ch[0].fadeout.assert_called_with(1500); ch[1].fadeout.assert_called_with(1500)
        n = ch[1].set_volume.call_count; eng.update(8, 0.016); assert ch[1].set_volume.call_count == n


def test_duck_fallback_engine():
    import tempfile, config
    from game import GREAT
    with tempfile.TemporaryDirectory() as d:                      # empty stems folder -> fallback
        pg, main, eng, ch = make_engine(Path(d))
        track = config.TRACKS[0]
        if not (main.SONGS_DIR / track.file).is_file():
            print("   (skipped: song file not present)"); return
        note = eng.start(track, "stems")
        assert eng.mode == "duck" and "No stems" in note
        mv = pg.mixer.music
        mv.load.assert_called_once(); mv.play.assert_called_once()
        assert abs(mv.set_volume.call_args[0][0] - 0.30) < 1e-9          # ducked by default
        eng.on_hit(GREAT, open_until=2.0); run_engine(eng, 1.0, 1.2)
        assert mv.set_volume.call_args[0][0] == 1.0                      # hit restores full volume
        run_engine(eng, 1.2, 2.5); assert abs(mv.set_volume.call_args[0][0] - 0.30) < 1e-6   # beat over -> ducked
        eng.on_miss(2.5); run_engine(eng, 2.5, 2.7); assert mv.set_volume.call_args[0][0] == 0.0   # stumble
        run_engine(eng, 2.7, 3.2); assert abs(mv.set_volume.call_args[0][0] - 0.30) < 1e-6
        assert ch[0].play.call_count == 0 and ch[1].play.call_count == 0
        assert eng.start(track, "off") is None and eng.mode == "off" and mv.set_volume.call_args[0][0] == 1.0


def test_window_icon_is_set():
    fake_pygame.icons.clear()
    run(lambda pg, mouse, main: [[]] * 3, ["--mouse"])
    assert len(fake_pygame.icons) == 1 and fake_pygame.icons[0].size == (64, 64)


def test_camera_background_and_escape():
    def build(pg, mouse, main):
        return [[]] * 3 + [[ev(pg, pg.K_RETURN)]] + [[]] * 40 + [[ev(pg, pg.K_c)]] + [[]] * 5 + \
               [[ev(pg, pg.K_ESCAPE)]] + [[]] * 3 + [[ev(pg, pg.K_ESCAPE)]]
    def patch(main):
        import mouse_input
        orig = mouse_input.MouseTracker.get_state
        def gs(self):
            s = orig(self); s.frame = np.full((720, 1280, 3), 60, np.uint8); return s
        mouse_input.MouseTracker.get_state = gs
    run(build, ["--mouse", "--full-song", "--approach", "2.0"], patch)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
