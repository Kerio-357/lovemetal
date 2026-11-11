"""LOVEMETAL - hand-tracked gothic rhythm game (YOLOv8-pose best.pt + Pygame).

Flow:  MAIN MENU (pick 1 of 3 HIM tracks, pick pace)  ->  PLAY  ->  RESULTS  ->  MENU

    python main.py                      # webcam + best.pt
    python main.py --device cuda:0      # run the model on GPU
    python main.py --mouse              # no camera: mouse moves, left button = pinch
    python main.py --full-song          # play whole tracks instead of the first 90 s

AUDIO: the three mp3 files live in the  songs/  folder next to this file. Their file
names, BPM and beat offsets are listed in TRACKS in config.py; the files are loaded in
start_music() below. Speed and difficulty knobs are in the "Rhythm / speed" block of
config.py.
"""
from __future__ import annotations

import argparse
import dataclasses
import time
from pathlib import Path

import pygame

from config import CFG, PACES, TRACKS
from game import GameCore
from input_pipeline import CursorController
from render import Renderer, make_window_icon

# >>> AUDIO: folder that holds the song files (funeral_of_hearts.mp3, ...).
SONGS_DIR = Path(__file__).parent / "songs"


# ----------------------------------------------------------------------------- setup
def parse_args():
    p = argparse.ArgumentParser(description="Lovemetal - rhythm game controlled by hand tracking")
    p.add_argument("--mouse", action="store_true", help="use the mouse instead of the webcam")
    p.add_argument("--weights", default=None, help="path to YOLOv8-pose weights (default: best.pt)")
    p.add_argument("--device", default=None, help="inference device: cpu, cuda:0, mps ...")
    p.add_argument("--imgsz", type=int, default=None, help="inference size (320 faster, 640 = trained)")
    p.add_argument("--conf", type=float, default=None, help="min hand detection confidence")
    p.add_argument("--anchor", choices=["pinch_mid", "index_tip"], default=None,
                   help="which point drives the cursor")
    p.add_argument("--camera", type=int, default=None, help="camera index")
    p.add_argument("--smoother", choices=["ema", "kalman"], default=None)
    p.add_argument("--approach", type=float, default=None,
                   help="seconds a note is visible before its beat (bigger = easier)")
    p.add_argument("--audio-offset", type=float, default=None,
                   help="shift notes against the music, seconds (+ = later)")
    p.add_argument("--full-song", action="store_true", help="play the whole track")
    p.add_argument("--seed", type=int, default=None, help="fixed beatmap seed (reproducible runs)")
    p.add_argument("--fullscreen", action="store_true")
    return p.parse_args()


def build_config(args):
    mapping = {"weights": "weights", "device": "yolo_device", "imgsz": "yolo_imgsz",
               "conf": "yolo_conf", "anchor": "cursor_anchor", "camera": "camera_index",
               "smoother": "smoother", "approach": "approach_time",
               "audio_offset": "audio_offset_s", "seed": "rng_seed"}
    over = {cfg_key: getattr(args, arg) for arg, cfg_key in mapping.items()
            if getattr(args, arg) is not None}
    if args.full_song:
        over["max_song_seconds"] = None
    return dataclasses.replace(CFG, **over)


def start_tracker(cfg, use_mouse: bool):
    """Returns (tracker, startup_note). Falls back to the mouse if the model/camera fail."""
    note = None
    if not use_mouse:
        try:
            from hand_tracker import HandTracker
            print("[tracker] loading model and opening camera ...")
            return HandTracker(cfg).start(), None
        except Exception as e:  # noqa: BLE001 - any startup failure -> playable fallback
            print(f"[tracker] FAILED: {e}\n[tracker] falling back to mouse control.")
            note = f"Tracker unavailable ({type(e).__name__}) - using mouse"
    from mouse_input import MouseTracker
    return MouseTracker(cfg).start(), note


# ----------------------------------------------------------------------------- audio
def init_audio() -> bool:
    """Open the sound device. The game still runs (silently) if there is none."""
    try:
        pygame.mixer.init()
        return True
    except pygame.error as e:
        print(f"[audio] no audio device: {e}")
        return False


def start_music(track, cfg, audio_ok: bool) -> str | None:
    """Load and start the selected track. Returns a warning text, or None if it plays.

    >>> AUDIO FILES ARE LOADED HERE <<<
    track.file comes from TRACKS in config.py and is looked up in songs/.
    pygame.mixer.music streams the file from disk (mp3, ogg and wav all work).
    """
    if not audio_ok:
        return "No audio device - playing without sound"
    path = SONGS_DIR / track.file
    if not path.is_file():
        return f"Missing audio file songs/{track.file} - playing without sound"
    try:
        pygame.mixer.music.load(str(path))               # <-- load the song file
        pygame.mixer.music.set_volume(cfg.music_volume)
        pygame.mixer.music.play()                        # <-- playback starts at song time 0
        return None
    except pygame.error as e:
        return f"Cannot play songs/{track.file} ({e}) - playing without sound"


def stop_music(audio_ok: bool, fade_ms: int = 0) -> None:
    if not audio_ok:
        return
    try:
        if fade_ms:
            pygame.mixer.music.fadeout(fade_ms)
        else:
            pygame.mixer.music.stop()
    except pygame.error:
        pass


# ----------------------------------------------------------------------------- main
def main() -> None:
    args = parse_args()
    cfg = build_config(args)

    pygame.init()
    audio_ok = init_audio()
    icon = make_window_icon()                      # assets/heartagram.png as the window / taskbar icon
    if icon is not None:
        pygame.display.set_icon(icon)              # must be set before set_mode()
    flags = pygame.FULLSCREEN | pygame.SCALED if args.fullscreen else 0
    screen = pygame.display.set_mode((cfg.screen_w, cfg.screen_h), flags)
    pygame.display.set_caption("Lovemetal - Gothic Rhythm Game")
    clock = pygame.time.Clock()

    tracker, startup_note = start_tracker(cfg, args.mouse)
    using_mouse = args.mouse or startup_note is not None

    # The mouse does not jitter, so in mouse mode the smoothing filter is bypassed.
    controller = CursorController(
        dataclasses.replace(cfg, smoother="ema", ema_alpha=1.0) if using_mouse else cfg)
    core = GameCore(cfg)
    renderer = Renderer(screen, cfg)

    mode = "menu"                          # menu | play | results
    selected = 0                           # index into TRACKS
    pace = min(max(cfg.default_pace, 0), len(PACES) - 1)   # index into PACES
    show_camera = cfg.camera_background
    audio_note = None                      # warning if the song could not be played
    music_faded = False

    def start_game() -> str:
        nonlocal audio_note, music_faded
        track = TRACKS[selected]
        renderer.clear_effects()
        audio_note = start_music(track, cfg, audio_ok)
        # Song time 0 = the moment playback starts; notes are placed on the track's
        # beat grid (bpm + offset from config.py), every PACES[pace] beats.
        core.start(time.perf_counter(), track, PACES[pace][1])
        music_faded = False
        return "play"

    def to_menu() -> str:
        stop_music(audio_ok)
        renderer.clear_effects()
        return "menu"

    try:
        running = True
        while running:
            now = time.perf_counter()

            # ---- input: hand / mouse cursor --------------------------------
            state = tracker.get_state()
            cur = controller.update(state, now)
            click_pos, click_t = None, None
            if cur.click_time is not None:               # pinch (or left button in --mouse mode)
                click_pos, click_t = cur.pos, cur.click_time

            # ---- input: keyboard / window ----------------------------------
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    running = False
                elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1 and not using_mouse:
                    if mode != "play":                   # the mouse always works in the menus
                        click_pos, click_t = e.pos, now
                elif e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_ESCAPE:
                        if mode == "menu":
                            running = False
                        else:
                            mode = to_menu()
                    elif e.key == pygame.K_c:
                        show_camera = not show_camera
                    elif e.key == pygame.K_SPACE and cur.visible:
                        click_pos, click_t = cur.pos, now    # keyboard "pinch" at the cursor
                    elif mode == "menu":
                        if e.key in (pygame.K_UP, pygame.K_w):
                            selected = (selected - 1) % len(TRACKS)
                        elif e.key in (pygame.K_DOWN, pygame.K_s):
                            selected = (selected + 1) % len(TRACKS)
                        elif e.key in (pygame.K_TAB, pygame.K_LEFT, pygame.K_RIGHT):
                            step = -1 if e.key == pygame.K_LEFT else 1
                            pace = (pace + step) % len(PACES)
                        elif e.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                            mode = start_game()
                    elif mode == "results" and e.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                        mode = to_menu()

            # ---- update + draw ---------------------------------------------
            renderer.begin(state, show_camera)           # full-screen webcam background

            if mode == "menu":
                if click_pos is not None:
                    for i, rect in enumerate(renderer.track_rects[:len(TRACKS)]):
                        if rect.collidepoint(click_pos):
                            selected = i                 # choose a song
                    if renderer.btn_pace.collidepoint(click_pos):
                        pace = (pace + 1) % len(PACES)
                    elif renderer.btn_start.collidepoint(click_pos):
                        mode = start_game()              # launch the game loop
                if mode == "menu":
                    renderer.draw_menu(TRACKS, selected, PACES[pace][0], PACES[pace][1], cur)

            elif mode == "play":
                if click_pos is not None:                # judge clicks first, then expire misses
                    renderer.add_click_ring(click_pos, now)
                    ev = core.click(click_pos, click_t)  # judged at the moment of the pinch
                    if ev:
                        renderer.add_hit(ev)
                for ev in core.update(now):
                    renderer.add_hit(ev)
                renderer.draw_notes(core, now)
                renderer.draw_hud(core, now)
                if core.finished:
                    if not music_faded:
                        stop_music(audio_ok, fade_ms=1500)
                        music_faded = True
                    if core.song_time(now) > core.end_time:
                        mode = "results"

            elif mode == "results":
                renderer.draw_results(core, cur)
                if click_pos is not None and renderer.btn_return.collidepoint(click_pos):
                    mode = to_menu()

            renderer.draw_effects(now)
            renderer.draw_cursor(cur)

            # Status line: tracker problems > audio problems > startup note > "show your hand"
            msg = getattr(tracker, "error", None) or (audio_note if mode == "play" else None) \
                or startup_note
            if msg:
                renderer.draw_status(msg)
            elif not using_mouse and not cur.tracking:
                renderer.draw_status("No hand detected - show your hand to the camera", warn=False)
            renderer.draw_fps(clock.get_fps(), state.infer_ms)
            pygame.display.flip()
            clock.tick(cfg.fps)
    finally:
        stop_music(audio_ok)
        tracker.stop()
        pygame.quit()


if __name__ == "__main__":
    main()
