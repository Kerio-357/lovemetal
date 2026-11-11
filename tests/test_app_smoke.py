"""Drives main.main() through menu -> play -> results -> menu with a fake pygame,
in --mouse mode and with a (fake) camera frame as background."""
import sys, time, types
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import fake_pygame


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
        def menu(self, tracks, sel, pn, pb, cur, hm="pinch"): log.setdefault("menu", []).append((sel, pn, hm)); return orig_menu(self, tracks, sel, pn, pb, cur, hm)
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
        orig = main.start_music
        def sm(track, cfg, ok): log["music"] = track.file; return orig(track, cfg, ok)
        main.start_music = sm
        for name in ("play_hit_sound", "spawn_hit_particles", "show_score_popup"):   # count feedback calls
            def wrap(*a, _o=getattr(main, name), _n=name): log.setdefault(_n, []).append(a); return _o(*a)
            setattr(main, name, wrap)
    pg, main = run(build, ["--mouse"], patch)
    sels = [m[0] for m in log["menu"]]
    assert 1 in sels and sels[-1] == 2, sorted(set(sels))                    # keyboard moved to 1, click chose 2
    paces = [m[1] for m in log["menu"]]
    assert paces[0] == "NORMAL" and "FAST" in paces and paces[-1] == "SLOW", set(paces)
    assert log["music"] == "bury_me_deep_inside.mp3"
    assert log.get("played") and "results" in log, log.keys()
    score, counts, title = log["results"]
    assert title == "Bury Me Deep Inside" and score > 0 and counts[0] == 0, log["results"]
    pg.mixer.music.load.assert_called()                          # file exists -> it was loaded
    # every hit -> one sound, one particle burst, one "PERFECT!" pop-up at the note position
    assert len(log["play_hit_sound"]) == len(log["spawn_hit_particles"]) == counts[300] == 3
    assert [a[0] for a in log["show_score_popup"]] == ["PERFECT!"] * 3
    assert pg.mixer.Sound.call_args[0][0].endswith("hit.wav")
    assert pg.mixer.Sound.return_value.play.call_count == 3
    pg.mixer.music.stop.assert_called(); assert pg.mixer.music.pause.call_count == 0
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
