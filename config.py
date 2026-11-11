"""Central configuration. Tweak values here instead of hunting through the code."""
from dataclasses import dataclass


# =============================================================================
# TRACKS
# =============================================================================
@dataclass(frozen=True)
class Track:
    title: str             # name shown in the menu
    file: str              # AUDIO FILE NAME inside the songs/ folder (mp3 / ogg / wav)
    bpm: float             # tempo of the track; notes are placed on this beat grid
    offset: float          # seconds from the start of the file to a beat (grid anchor)
    duration: float        # length of the audio file in seconds
    lead_in: float = 0.0   # no notes before this time (e.g. a long quiet intro)
    # Optional hand-made beatmap: exact note times in seconds, e.g. (4.2, 5.25, 6.3).
    # Empty = notes are generated from bpm + offset (the default for all three songs).
    timestamps: tuple = ()


# >>> AUDIO: put the audio files into  osu_cv_project/songs/  with exactly these names.
# BPM / offset were measured from the three mp3 files with an onset-autocorrelation
# script. To add a song, drop the file into songs/ and add a Track(...) line here.
TRACKS = (
    Track("The Funeral of Hearts", "funeral_of_hearts.mp3",  bpm=113.99, offset=0.245, duration=270.3),
    Track("Join Me in Death",      "join_me_in_death.mp3",   bpm=113.76, offset=0.091, duration=216.8),
    Track("Bury Me Deep Inside",   "bury_me_deep_inside.mp3", bpm=85.85, offset=0.304, duration=256.3,
          lead_in=4.0),
)

# Menu "PACE" button: how many beats pass between two notes.
# At 114 BPM: 4 beats = a note every 2.1 s, 2 beats = 1.05 s, 1 beat = 0.53 s.
PACES = (("SLOW", 4), ("NORMAL", 2), ("FAST", 1))


@dataclass(frozen=True)
class Config:
    # --- Display ---------------------------------------------------------
    screen_w: int = 1280
    screen_h: int = 720
    fps: int = 60                      # render cap, enforced with clock.tick(fps)
    # Optional frame skipping: if drawing one frame takes longer than the frame budget
    # (slow laptop), only every second frame is drawn. Input, judgement and audio still
    # run every loop, so hits stay responsive; only the picture updates less often.
    frame_skip: bool = True
    audio_buffer: int = 512            # mixer buffer in samples (512 = ~12 ms; 256 = lower lag, may crackle)

    # --- Camera / tracking ----------------------------------------------
    camera_index: int = 0
    cam_w: int = 640
    cam_h: int = 480
    mirror: bool = True                # selfie view: moving hand right moves cursor right
    # The webcam picture fills the whole screen and the cursor sits exactly on your
    # hand in that picture. A margin > 0 additionally stretches the centre of the
    # picture so the screen edges are easier to reach (cursor then leads the hand).
    active_margin_x: float = 0.0
    active_margin_y: float = 0.0
    lost_grace_s: float = 0.25         # after this long without a hand: stop clicks, re-seed filter

    # --- Full-screen webcam background ----------------------------------
    camera_background: bool = True     # False = plain gothic gradient (also toggled with C)
    # Gothic filter: the picture is converted to grayscale, then each channel is
    # gray * multiplier. Lower numbers = darker; raise the first for a redder look.
    bg_tint: tuple = (0.50, 0.10, 0.14)      # (R, G, B) multipliers
    bg_vignette: int = 150             # 0..255 darkness of the screen corners
    draw_hand_skeleton: bool = True    # draw the detected keypoints on the background

    # --- YOLOv8-pose model (best.pt) ------------------------------------
    weights: str = "best.pt"           # relative paths resolve next to main.py
    # PERFORMANCE: the camera picture is shown full-screen at its own resolution, but YOLO
    # only sees a copy downscaled to yolo_imgsz. 320 is the fastest setting that still
    # tracks a hand well on a laptop CPU; use 416 or 640 if keypoints look unstable.
    yolo_imgsz: int = 320              # inference size (multiple of 32): 320 fast, 416, 640 = trained size
    yolo_half: bool = False            # half-precision inference (only useful on a CUDA GPU)
    yolo_conf: float = 0.35            # min hand-box confidence
    yolo_device: str | None = None     # None = auto (CUDA if available), or "cpu", "cuda:0", "mps"
    kpt_conf: float = 0.30             # min confidence of the keypoints we rely on
    # Cursor anchor: "pinch_mid" (midpoint of thumb+index tips, stable while pinching)
    # or "index_tip".
    cursor_anchor: str = "pinch_mid"

    # --- Smoothing -------------------------------------------------------
    smoother: str = "kalman"           # "ema" or "kalman"
    ema_alpha: float = 0.35            # 0..1, higher = more responsive, more jitter
    # Kalman process noise = expected hand acceleration^2 (px/s^2)^2. Higher = snappier,
    # lower = smoother but the cursor trails behind fast moves. (The old value 400 lagged
    # far behind a fast hand; 1e6 follows it while still reducing the jitter.)
    kalman_process_noise: float = 1.0e6
    kalman_meas_noise: float = 25.0       # pixels^2; higher = smoother (more lag)
    # Between camera frames the cursor is extrapolated along its current velocity, which
    # hides camera + model latency and makes it move at 60 fps instead of camera fps.
    cursor_predict_s: float = 0.08        # max look-ahead in seconds (0 = off)
    cursor_predict_min_speed: float = 250.0   # px/s; below this the look-ahead fades out (no jitter at rest)

    # --- PINCH GESTURE = the only way to hit a note ---------------------------
    # Measured from the YOLO keypoints every camera frame:
    #     pinch_ratio = distance(thumb tip, index tip) / distance(wrist, middle knuckle)
    # Dividing by the palm size makes it independent of how far the hand is from the
    # camera. Open hand ~0.6-1.2, fingertips touching ~0.05-0.20.
    #
    # >>> SENSITIVITY - tune these while watching the pinch meter (bottom-left, key P):
    #   pinch_close_thresh  the ratio must drop BELOW this to count as a pinch.
    #       Pinches not registering      -> raise it (0.32 ... 0.40)
    #       Accidental / phantom pinches -> lower it (0.20 ... 0.25)
    #   pinch_open_thresh   the ratio must rise ABOVE this before the next pinch can
    #       start (hysteresis). Keep it about 1.5x the close threshold; a bigger gap
    #       means fewer double triggers but you must open the fingers wider.
    # In the game the [ and ] keys change the close threshold live (open follows at 1.5x).
    pinch_close_thresh: float = 0.28
    pinch_open_thresh: float = 0.42
    # A pinch is accepted on the FIRST frame if the ratio is clearly closed (below
    # pinch_instant_ratio) - no added input lag. A borderline reading must persist for
    # pinch_min_frames camera frames, which filters single-frame keypoint glitches.
    pinch_instant_ratio: float = 0.20
    pinch_min_frames: int = 2
    click_cooldown_s: float = 0.12     # minimum time between two pinches

    # --- Rhythm / speed (THE KNOBS TO TURN IF IT FEELS TOO FAST OR SLOW) ----
    default_pace: int = 1              # index into PACES: 0 = SLOW, 1 = NORMAL, 2 = FAST
    approach_time: float = 1.6         # seconds a note is visible before its beat (reaction time)
    approach_scale: float = 3.0        # approach ring starts at note_radius * this
    lead_in_s: float = 3.0             # seconds of music before the first note
    max_song_seconds: float | None = 90.0   # play only this much of a track; None = whole song
    audio_offset_s: float = 0.0        # + = notes later, - = earlier (fix audio/video lag here)
    music_volume: float = 0.8
    # STRICT BEAT WINDOW: judgement windows are fractions of ONE BEAT of the selected
    # song (either side of the note's beat), so they tighten automatically with tempo.
    # At 114 BPM a beat is 0.526 s -> PERFECT +-0.095 s, GREAT +-0.158 s, GOOD +-0.237 s.
    # Outside the GOOD window a hit attempt does nothing and the note becomes a MISS.
    # Raise these three numbers to make timing more forgiving.
    window_beats_perfect: float = 0.18
    window_beats_great: float = 0.30
    window_beats_good: float = 0.45

    # --- Adaptive audio: the music reacts to how you play --------------------
    # "stems": backing track always plays; the DRUM stem is heard only while you hit
    #          notes on time (needs songs/stems/<name>_backing.ogg + <name>_drums.ogg).
    # "duck":  single-file fallback - the song is quiet by default, a hit restores full
    #          volume until the next note is due, a miss cuts it to silence for a moment.
    # "off":   the song just plays.
    # "auto":  "stems" when both stem files exist for the track, otherwise "duck".
    adaptive_audio: str = "auto"
    drum_gain_perfect: float = 1.0     # drum stem volume after a PERFECT / GREAT / GOOD hit
    drum_gain_great: float = 0.85
    drum_gain_good: float = 0.6
    audio_attack_s: float = 0.02       # fade-in time when the beat comes back (no clicks)
    audio_release_s: float = 0.10      # fade-out time when it is lost
    duck_level: float = 0.30           # "duck" mode: volume while you are not hitting (0..1)
    stumble_time: float = 0.40         # "duck" mode: seconds of silence after a miss

    # --- Hit feedback ---------------------------------------------------------
    hit_volume: float = 0.7            # loudness of assets/hit.wav (0..1)
    popup_time: float = 0.5            # seconds the PERFECT! / GREAT! / MISS! text stays
    particle_count: int = 26           # sparks per hit

    # --- Targets ---------------------------------------------------------
    note_radius: int = 64
    hit_forgiveness: float = 1.6       # hit radius = note_radius * this (~102 px)
    cursor_radius: int = 16
    spawn_margin_x: int = 170          # keep notes away from the screen edges ...
    spawn_margin_top: int = 190        # ... below the HUD bar ...
    spawn_margin_bottom: int = 120     # ... and above the bottom edge
    min_note_spacing: int = 180        # distance between consecutive notes: not on top
    max_note_spacing: int = 460        # ... of each other, but reachable in one move
    rng_seed: int | None = None

    # --- Health bar ("SOUL RHYTHM") - visual only, the song never fails ---
    health_gain: float = 0.04
    health_loss: float = 0.10


CFG = Config()
