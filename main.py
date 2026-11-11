"""LOVEMETAL - hand-tracked gothic rhythm game (YOLOv8-pose best.pt + Pygame).

Flow:  MAIN MENU (pick 1 of 3 HIM tracks, pick pace)  ->  PLAY  ->  RESULTS  ->  MENU

    python main.py                      # webcam + best.pt
    python main.py --device cuda:0      # run the model on GPU
    python main.py --mouse              # no camera: mouse moves, left button = pinch
    python main.py --full-song          # play whole tracks instead of the first 90 s

AUDIO: the three mp3 files live in the  songs/  folder next to this file. Their file
names, BPM and beat offsets are listed in TRACKS in config.py; the files are loaded in
AdaptiveAudio.start() below. The hit sound is  assets/hit.wav  (see load_hit_sound()).
Speed and difficulty knobs are in the "Rhythm / speed" block of config.py.

ADAPTIVE AUDIO (class AdaptiveAudio below): the music reacts to how you play.
  * "stems" mode - two synchronized files per song in  songs/stems/ :
        <name>_backing.ogg  (melody, vocals, pads)  -> always plays
        <name>_drums.ogg    (drums / beat)          -> heard ONLY while you hit on time
    Each hit opens the drum stem until the next note is due; a miss mutes it.
  * "duck" mode - fallback when a song has no stems: the single file plays quietly,
    a hit restores full volume, a miss cuts it to silence for a moment.
  Create stems with  python tools/make_stems.py  (already done for the three songs).

INPUT (Meta Quest style): aim with the hand - the cursor is anchored to the knuckles,
  so it stays still while the fingers move - and "click" by tapping thumb tip and
  index tip together. That pinch is the only way to hit a note.
  HIT = cursor inside the note's hit radius + pinch + inside the note's beat window.
  Hovering does nothing. Sensitivity: pinch_close_thresh in config.py, --pinch, or the
  [ and ] keys while the game runs (P shows the pinch meter).

RHYTHM: notes come from game.note_times() - one note every N beats of the selected
track's BPM grid - and can only be hit inside their timing window (GameCore.click).
Every judged note goes through on_judgement(), which calls the three feedback
functions: play_hit_sound(), spawn_hit_particles(x, y), show_score_popup(text, x, y).
"""
from __future__ import annotations

import argparse
import dataclasses
import threading
import time
from pathlib import Path

import pygame

from config import CFG, PACES, TRACKS
from game import GREAT, LABELS, MISS, OK, GameCore, HitEvent
from input_pipeline import CursorController
from render import HIT_COLORS, Renderer, make_window_icon

# >>> AUDIO: folder that holds the song files (funeral_of_hearts.mp3, ...).
SONGS_DIR = Path(__file__).parent / "songs"

# >>> AUDIO STEMS: for the adaptive "stems" mode put two files per song here:
#       songs/stems/funeral_of_hearts_backing.ogg     songs/stems/funeral_of_hearts_drums.ogg
#       songs/stems/join_me_in_death_backing.ogg      songs/stems/join_me_in_death_drums.ogg
#       songs/stems/bury_me_deep_inside_backing.ogg   songs/stems/bury_me_deep_inside_drums.ogg
# i.e. <song file name without .mp3> + "_backing.ogg" / "_drums.ogg" (ogg or wav content,
# same length, both starting at 0:00 so they stay in sync).
STEMS_DIR = SONGS_DIR / "stems"
MUSIC_MODES = ("auto", "stems", "duck", "off")

# >>> HIT SOUND: put your own sound here as  osu_cv_project/assets/hit.wav
# (short, < 0.3 s, wav or ogg). A synthesized gothic hit-clap is already there;
# replace the file to change the sound. If the file is missing, hits are silent.
HIT_SOUND_PATH = Path(__file__).parent / "assets" / "hit.wav"

# Objects the feedback functions below need; filled in once by main().
_fx = {"renderer": None, "hit_sound": None, "audio": None, "core": None}


# ----------------------------------------------------------------------------- setup
def parse_args():
    p = argparse.ArgumentParser(description="Lovemetal - rhythm game controlled by hand tracking")
    p.add_argument("--mouse", action="store_true", help="use the mouse instead of the webcam")
    p.add_argument("--weights", default=None, help="path to YOLOv8-pose weights (default: best.pt)")
    p.add_argument("--device", default=None, help="inference device: cpu, cuda:0, mps ...")
    p.add_argument("--imgsz", type=int, default=None, help="inference size (320 faster, 640 = trained)")
    p.add_argument("--conf", type=float, default=None, help="min hand detection confidence")
    p.add_argument("--anchor", choices=["knuckles", "pinch_mid", "index_tip"], default=None,
                   help="which point drives the cursor")
    p.add_argument("--camera", type=int, default=None, help="camera index")
    p.add_argument("--smoother", choices=["ema", "kalman"], default=None)
    p.add_argument("--approach", type=float, default=None,
                   help="seconds a note is visible before its beat (bigger = easier)")
    p.add_argument("--audio-offset", type=float, default=None,
                   help="shift notes against the music, seconds (+ = later)")
    p.add_argument("--pinch", type=float, default=None, metavar="RATIO",
                   help="pinch sensitivity: close threshold of the pinch ratio (default 0.28; "
                        "higher = easier to trigger, lower = fewer accidental pinches)")
    p.add_argument("--no-frame-skip", action="store_true", help="always draw every frame")
    p.add_argument("--music", choices=list(MUSIC_MODES), default=None,
                   help="adaptive audio: stems (drums follow your hits), duck, off, auto")
    p.add_argument("--full-song", action="store_true", help="play the whole track")
    p.add_argument("--seed", type=int, default=None, help="fixed beatmap seed (reproducible runs)")
    p.add_argument("--fullscreen", action="store_true")
    return p.parse_args()


def build_config(args):
    mapping = {"weights": "weights", "device": "yolo_device", "imgsz": "yolo_imgsz",
               "conf": "yolo_conf", "anchor": "cursor_anchor", "camera": "camera_index",
               "smoother": "smoother", "approach": "approach_time",
               "audio_offset": "audio_offset_s", "seed": "rng_seed", "music": "adaptive_audio", "pinch": "pinch_close_thresh"}
    over = {cfg_key: getattr(args, arg) for arg, cfg_key in mapping.items()
            if getattr(args, arg) is not None}
    if args.full_song:
        over["max_song_seconds"] = None
    if args.pinch is not None:                 # keep the re-open threshold 1.5x above
        over["pinch_open_thresh"] = args.pinch * 1.5
    if args.no_frame_skip:
        over["frame_skip"] = False
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
        if not pygame.mixer.get_init():          # normally already opened by pygame.init()
            pygame.mixer.init()
        return True
    except pygame.error as e:
        print(f"[audio] no audio device: {e}")
        return False


def stem_paths(track) -> tuple[Path, Path]:
    """(backing, drums) stem files that belong to a track."""
    base = Path(track.file).stem
    return STEMS_DIR / f"{base}_backing.ogg", STEMS_DIR / f"{base}_drums.ogg"


class AdaptiveAudio:
    """Interactive music: what you hear depends on how you play.

    One number drives everything: `gain` (0..1) = "how alive the beat is".
      on_hit()  -> gain target = 1.0 / 0.85 / 0.6 (PERFECT / GREAT / GOOD), held until the
                   next note is due. Keep hitting and the beat never drops out.
      on_miss() -> gain target = 0 at once.
    update() glides gain towards its target every frame (short fades, no clicks) and
    applies it:
      "stems": backing stem at full volume on mixer Channel 0 (never touched);
               drum stem on Channel 1 with volume = gain  -> silent beat when you miss.
      "duck":  the single song file via pygame.mixer.music with
               volume = duck_level + (1 - duck_level) * gain, and 0 for `stumble_time`
               seconds after a miss.
      "off":   plain playback.
    The hit-clap (play_hit_sound) uses the remaining free channels.
    """

    BACKING_CHANNEL, DRUMS_CHANNEL = 0, 1

    def __init__(self, cfg, audio_ok: bool):
        self.cfg = cfg
        self.ok = audio_ok
        self.mode = "off"                 # what is active right now: stems | duck | off
        self.gain = 0.0                   # current beat level (smoothed)
        self._hit_gain = 0.0              # level granted by the last hit
        self._open_until = -1.0           # song time until which the beat stays alive
        self._stumble_until = -1.0        # "duck" mode: silent until this song time
        self._playing = False
        self._cache: dict = {}            # last loaded stems: {track.file: (backing, drums)}
        self._load_lock = threading.Lock()
        self._preloading: str | None = None
        self._ch_backing = self._ch_drums = None
        if audio_ok:
            try:
                pygame.mixer.set_reserved(2)          # channels 0+1 belong to the stems
                self._ch_backing = pygame.mixer.Channel(self.BACKING_CHANNEL)
                self._ch_drums = pygame.mixer.Channel(self.DRUMS_CHANNEL)
            except pygame.error as e:
                print(f"[audio] cannot reserve mixer channels: {e}")

    # -- which mode will a track get? --------------------------------------
    def resolve_mode(self, track, wanted: str) -> str:
        has_stems = self._ch_drums is not None and all(p.is_file() for p in stem_paths(track))
        if wanted == "auto":
            return "stems" if has_stems else "duck"
        if wanted == "stems" and not has_stems:
            return "duck"                 # asked for stems but the files are missing
        return wanted

    # -- start / stop -------------------------------------------------------
    def start(self, track, wanted: str) -> str | None:
        """Load and start the music for `track`. Returns a warning text or None.
        Song time 0 is the moment this function returns."""
        self.stop()
        self.gain, self._hit_gain = 0.0, 0.0
        self._open_until = self._stumble_until = -1.0
        self.mode = "off"
        if not self.ok:
            return "No audio device - playing without sound"
        mode = self.resolve_mode(track, wanted)
        note = None
        if wanted == "stems" and mode != "stems":
            note = "No stems for this song in songs/stems/ - using volume ducking instead"

        if mode == "stems":
            try:
                backing, drums = self._load_stems(track)
                # >>> BOTH STEMS START HERE, back to back, so they stay sample-aligned <<<
                self._ch_backing.play(backing)
                self._ch_drums.play(drums)
                self._ch_drums.set_volume(0.0)                    # beat is silent until you hit
                self._ch_backing.set_volume(self.cfg.music_volume)
                self.mode, self._playing = "stems", True
                return note
            except (pygame.error, OSError) as e:
                note = f"Cannot play stems ({e}) - using the single track"
                mode = "duck"

        # Single-file playback ("duck" fallback or "off").
        path = SONGS_DIR / track.file
        if not path.is_file():
            return f"Missing audio file songs/{track.file} - playing without sound"
        try:
            pygame.mixer.music.load(str(path))                    # <-- the full song file is loaded here
            start_volume = self.cfg.duck_level if mode == "duck" else 1.0
            pygame.mixer.music.set_volume(self.cfg.music_volume * start_volume)
            pygame.mixer.music.play()
            self.mode, self._playing = mode, True
            return note
        except pygame.error as e:
            return f"Cannot play songs/{track.file} ({e}) - playing without sound"

    def _load_stems(self, track):
        """>>> THE TWO STEM FILES ARE LOADED HERE <<< (decoded fully into memory, ~50 MB each)."""
        with self._load_lock:                 # waits here if a background preload is still running
            if track.file not in self._cache:
                backing_path, drums_path = stem_paths(track)
                self._cache = {track.file: (pygame.mixer.Sound(str(backing_path)),     # melody / vocals / pads
                                            pygame.mixer.Sound(str(drums_path)))}      # drums / beat
            return self._cache[track.file]

    def preload(self, track, wanted: str) -> None:
        """Decode the selected track's stems in a background thread while the player is
        still in the menu, so pressing START does not freeze the game."""
        if not self.ok or self._playing or self.resolve_mode(track, wanted) != "stems":
            return
        if track.file in self._cache or self._preloading == track.file:
            return
        self._preloading = track.file

        def work():
            try:
                self._load_stems(track)
            except (pygame.error, OSError) as e:
                print(f"[audio] preload failed: {e}")
            finally:
                if self._preloading == track.file:
                    self._preloading = None

        threading.Thread(target=work, name="stem-preload", daemon=True).start()

    def stop(self, fade_ms: int = 0) -> None:
        if not self.ok or not self._playing:
            return
        self._playing = False             # update() stops touching the volumes during a fade-out
        try:
            if self.mode == "stems":
                for ch in (self._ch_backing, self._ch_drums):
                    ch.fadeout(fade_ms) if fade_ms else ch.stop()
            elif fade_ms:
                pygame.mixer.music.fadeout(fade_ms)
            else:
                pygame.mixer.music.stop()
        except pygame.error:
            pass

    # -- performance events -----------------------------------------------
    def on_hit(self, points: int, open_until: float) -> None:
        """A note was hit on time: bring the beat back until `open_until` (song time
        at which the next note would be missed)."""
        c = self.cfg
        self._hit_gain = (c.drum_gain_perfect if points == GREAT else
                          c.drum_gain_great if points == OK else c.drum_gain_good)
        self._open_until = open_until
        self._stumble_until = -1.0

    def on_miss(self, song_time: float) -> None:
        """A note was skipped: the beat dies (and "duck" mode stumbles into silence)."""
        self._open_until = -1.0
        self._stumble_until = song_time + self.cfg.stumble_time

    @property
    def beat_alive(self) -> bool:
        return self.gain > 0.05

    # -- per frame ----------------------------------------------------------
    def update(self, song_time: float, dt: float) -> None:
        c = self.cfg
        target = self._hit_gain if song_time < self._open_until else 0.0
        step = dt / max(c.audio_attack_s if target > self.gain else c.audio_release_s, 1e-3)
        self.gain += max(-step, min(step, target - self.gain))
        if not self._playing:
            return
        try:
            if self.mode == "stems":
                self._ch_drums.set_volume(c.music_volume * self.gain)
            elif self.mode == "duck":
                level = c.duck_level + (1.0 - c.duck_level) * self.gain
                if song_time < self._stumble_until:
                    level = 0.0                                   # stumble: music drops out
                pygame.mixer.music.set_volume(c.music_volume * level)
        except pygame.error:
            pass


def music_label(audio: AdaptiveAudio, track, wanted: str) -> tuple[str, str]:
    """Menu button text: which adaptive-audio mode the selected track will use."""
    mode = audio.resolve_mode(track, wanted) if audio.ok else "off"
    sub = {"stems": "drums play only while you hit",
           "duck": "volume drops when you miss" if wanted != "stems" else "no stems found - ducking",
           "off": "plain playback"}[mode]
    return mode.upper() + (" (AUTO)" if wanted == "auto" else ""), sub


# ----------------------------------------------------------------------------- hit feedback
def load_hit_sound(cfg, audio_ok: bool):
    """Load assets/hit.wav once at start-up. Returns None (silent hits) if unavailable.

    >>> THE HIT SOUND FILE IS LOADED HERE <<<   (path: HIT_SOUND_PATH above)
    """
    if not audio_ok:
        return None
    if not HIT_SOUND_PATH.is_file():
        print(f"[audio] {HIT_SOUND_PATH} not found - hits will be silent")
        return None
    try:
        sound = pygame.mixer.Sound(str(HIT_SOUND_PATH))
        sound.set_volume(cfg.hit_volume)
        return sound
    except pygame.error as e:
        print(f"[audio] cannot load hit sound: {e}")
        return None


def play_hit_sound() -> None:
    """Short hit-clap on a successful hit.

    pygame.mixer.Sound plays on its own mixer channel, so it is layered on top of the
    song (which streams through pygame.mixer.music) and never interrupts it.
    """
    sound = _fx["hit_sound"]
    if sound is not None:
        try:
            sound.play()
        except pygame.error:
            pass


def spawn_hit_particles(x: float, y: float) -> None:
    """Crimson/silver spark explosion + shattering heartagram where the target was hit."""
    _fx["renderer"].spawn_particles(x, y, time.perf_counter())


def show_score_popup(text: str, x: float, y: float) -> None:
    """Floating judgement text ("PERFECT!", "GREAT!", "GOOD!", "MISS!") for ~0.5 s."""
    color = next((HIT_COLORS[pts] for pts, label in LABELS.items() if label == text),
                 HIT_COLORS[MISS])
    _fx["renderer"].add_popup(text, x, y, color, time.perf_counter())


def on_judgement(ev: HitEvent) -> None:
    """All feedback for one judged note (a hit, or a note that was skipped)."""
    audio, core = _fx["audio"], _fx["core"]
    show_score_popup(LABELS[ev.points], ev.x, ev.y)
    if ev.points != MISS:
        play_hit_sound()
        spawn_hit_particles(ev.x, ev.y)
        # ADAPTIVE AUDIO: the beat comes back and stays until the next note is due.
        audio.on_hit(ev.points, core.next_deadline())
    else:
        # ADAPTIVE AUDIO: skipped note -> the drums fall silent (text only, no clap, no sparks).
        audio.on_miss(core.song_time(ev.at))


# ----------------------------------------------------------------------------- main
def main() -> None:
    args = parse_args()
    cfg = build_config(args)

    # Ask for a small audio buffer BEFORE pygame.init() so sounds follow a pinch without lag.
    pygame.mixer.pre_init(frequency=44100, size=-16, channels=2, buffer=cfg.audio_buffer)
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
    audio = AdaptiveAudio(cfg, audio_ok)                  # stems / ducking music engine
    _fx.update(renderer=renderer, hit_sound=load_hit_sound(cfg, audio_ok), audio=audio, core=core)

    mode = "menu"                          # menu | play | results
    selected = 0                           # index into TRACKS
    pace = min(max(cfg.default_pace, 0), len(PACES) - 1)   # index into PACES
    music_mode = cfg.adaptive_audio        # auto | stems | duck | off (menu button / M key)
    show_camera = cfg.camera_background
    show_meter = False                     # pinch meter during play (P key); always on in the menu
    audio_note = None                      # warning if the song could not be played
    music_faded = False
    last_frame = time.perf_counter()

    # Frame skipping state: smoothed cost of drawing one frame, in milliseconds.
    frame_budget_ms = 1000.0 / cfg.fps
    draw_ms, skipping, frame_index = 0.0, False, 0

    def start_game() -> str:
        nonlocal audio_note, music_faded
        track = TRACKS[selected]
        renderer.clear_effects()
        renderer.draw_status("Loading audio ...", warn=False)   # decoding stems takes a moment
        pygame.display.flip()
        audio_note = audio.start(track, music_mode)       # backing + drum stems (or single file)
        # Song time 0 = the moment playback starts; notes are placed on the track's
        # beat grid (bpm + offset from config.py), every PACES[pace] beats.
        core.start(time.perf_counter(), track, PACES[pace][1])
        music_faded = False
        return "play"

    def to_menu() -> str:
        audio.stop()
        renderer.clear_effects()
        return "menu"

    def next_music_mode() -> str:
        return MUSIC_MODES[(MUSIC_MODES.index(music_mode) + 1) % len(MUSIC_MODES)]

    try:
        running = True
        while running:
            now = time.perf_counter()
            dt, last_frame = now - last_frame, now

            # ================= 1. INPUT HANDLER: hand cursor + pinch =================
            # tracker.get_state() never blocks: it returns the newest YOLO result.
            # controller.update() smooths the cursor and turns the thumb-index distance
            # into a pinch event. cur.click_time is set only in the frame where the
            # fingers CLOSE (one event per pinch) and holds the capture time of the camera
            # frame that showed it, so camera/model latency does not count against timing.
            state = tracker.get_state()
            cur = controller.update(state, now)
            pinch_pos, pinch_t = None, None
            if cur.click_time is not None:
                pinch_pos, pinch_t = cur.pos, cur.click_time
            menu_click = pinch_pos                        # menus accept a pinch or a mouse click

            # ================= 2. KEYBOARD / WINDOW EVENTS ==========================
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    running = False
                elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1 and not using_mouse:
                    menu_click = e.pos                    # mouse works in menus only, never on notes
                elif e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_ESCAPE:
                        if mode == "menu":
                            running = False
                        else:
                            mode = to_menu()
                    elif e.key == pygame.K_c:
                        show_camera = not show_camera
                    elif e.key == pygame.K_p:
                        show_meter = not show_meter
                    elif e.key in (pygame.K_LEFTBRACKET, pygame.K_RIGHTBRACKET):
                        # PINCH SENSITIVITY, live: [ = stricter (fewer accidental pinches),
                        # ] = looser (pinch registers more easily). Copy the value you like
                        # into pinch_close_thresh in config.py to keep it.
                        step = -0.02 if e.key == pygame.K_LEFTBRACKET else 0.02
                        controller.gesture.set_sensitivity(controller.gesture.close_thresh + step)
                    elif mode == "menu":
                        if e.key in (pygame.K_UP, pygame.K_w):
                            selected = (selected - 1) % len(TRACKS)
                        elif e.key in (pygame.K_DOWN, pygame.K_s):
                            selected = (selected + 1) % len(TRACKS)
                        elif e.key in (pygame.K_TAB, pygame.K_LEFT, pygame.K_RIGHT):
                            step = -1 if e.key == pygame.K_LEFT else 1
                            pace = (pace + step) % len(PACES)
                        elif e.key == pygame.K_m:
                            music_mode = next_music_mode()
                        elif e.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                            mode = start_game()
                    elif mode == "results" and e.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                        mode = to_menu()

            # ================= 3. GAME LOGIC + AUDIO (before any drawing) ============
            # Everything that makes sound runs here, right after the input was read, so
            # the delay between a pinch and its audio is not extended by rendering.
            if mode == "menu":
                audio.preload(TRACKS[selected], music_mode)   # decode stems in the background
                if menu_click is not None:
                    for i, rect in enumerate(renderer.track_rects[:len(TRACKS)]):
                        if rect.collidepoint(menu_click):
                            selected = i                 # choose a song
                    if renderer.btn_pace.collidepoint(menu_click):
                        pace = (pace + 1) % len(PACES)
                    elif renderer.btn_mode.collidepoint(menu_click):
                        music_mode = next_music_mode()
                    elif renderer.btn_start.collidepoint(menu_click):
                        mode = start_game()              # launch the game loop

            elif mode == "play":
                # TARGET HIT CONDITION - all three must hold:
                #   [pinch gesture]            pinch_pos is set only when the fingers just closed
                #   [cursor inside the note]   checked in core.click() against the hit radius
                #   [valid beat window]        checked in core.click() against pinch_t
                # -> on_judgement(): hit-clap + drum stem unmuted + particles + score pop-up.
                # A hand merely resting on a note never reaches core.click().
                if pinch_pos is not None:
                    renderer.add_click_ring(pinch_pos, now)
                    ev = core.click(pinch_pos, pinch_t)
                    if ev is not None:
                        on_judgement(ev)
                # Notes whose window passed without a pinch -> MISS, drums fall silent.
                for miss in core.update(now):
                    on_judgement(miss)
                # Adaptive audio: fade the drum stem (or song volume) towards its target.
                audio.update(core.song_time(now), dt)
                if core.finished:
                    if not music_faded:
                        audio.stop(fade_ms=1500)
                        music_faded = True
                    if core.song_time(now) > core.end_time:
                        mode = "results"

            elif mode == "results":
                if menu_click is not None and renderer.btn_return.collidepoint(menu_click):
                    mode = to_menu()

            # ================= 4. DRAW (with optional frame skipping) ================
            # If drawing alone costs more than one frame budget, draw every second frame
            # only. Steps 1-3 still run every loop, so input and audio stay responsive.
            frame_index += 1
            if cfg.frame_skip:
                skipping = draw_ms > frame_budget_ms if not skipping else draw_ms > 0.8 * frame_budget_ms
            if not (skipping and frame_index % 2):
                t_draw = time.perf_counter()
                renderer.begin(state, show_camera)       # full-screen webcam background
                if mode == "menu":
                    name, sub = music_label(audio, TRACKS[selected], music_mode)
                    renderer.draw_menu(TRACKS, selected, PACES[pace][0], PACES[pace][1], cur, name, sub)
                elif mode == "play":
                    renderer.draw_notes(core, now)
                    renderer.draw_hud(core, now, beat=(audio.mode, audio.beat_alive))
                else:
                    renderer.draw_results(core, cur)
                renderer.draw_effects(now)
                renderer.draw_cursor(cur, controller.gesture.close_thresh)
                if not using_mouse and (show_meter or mode == "menu"):
                    renderer.draw_pinch_meter(cur, controller.gesture.close_thresh,
                                              controller.gesture.open_thresh)

                # Status line: tracker problems > audio problems > startup note > "show your hand"
                msg = getattr(tracker, "error", None) or (audio_note if mode == "play" else None) \
                    or startup_note
                if msg:
                    renderer.draw_status(msg)
                elif not using_mouse and not cur.tracking:
                    renderer.draw_status("No hand detected - show your hand to the camera", warn=False)
                renderer.draw_fps(clock.get_fps(), state.infer_ms, state.track_fps, skipping)
                cost = (time.perf_counter() - t_draw) * 1000.0
                draw_ms = cost if draw_ms == 0.0 else 0.9 * draw_ms + 0.1 * cost
                pygame.display.flip()

            # ================= 5. FRAME CAP =========================================
            clock.tick(cfg.fps)                          # 60 fps cap
    finally:
        audio.stop()
        tracker.stop()
        pygame.quit()


if __name__ == "__main__":
    main()
