# Lovemetal (Osu! CV)

Hand-tracked gothic rhythm game: YOLOv8-pose (`best.pt`, 21 hand keypoints) + OpenCV + Pygame.

    pip install -r requirements.txt
    python main.py                  # webcam + best.pt
    python main.py --device cuda:0  # GPU; --imgsz 320 for speed on CPU
    python main.py --mouse          # no camera: mouse + left button as "pinch"
    python main.py --full-song      # whole tracks instead of the first 90 s

Controls: move hand = cursor, pinch thumb+index = click.
Keyboard: UP/DOWN track, TAB pace, ENTER start, SPACE click, C camera background, ESC menu/quit.

## Songs
The mp3 files are **not** in this repository (copyrighted). Put your own files into `songs/` using the names from `TRACKS` in
`config.py` (`funeral_of_hearts.mp3`, `join_me_in_death.mp3`, `bury_me_deep_inside.mp3`); without them the game still runs, silently.
Audio files live in `songs/`. Each track's file name, BPM and beat offset are in `TRACKS`
in `config.py`; notes are placed on that beat grid. Files are loaded in `start_music()` in `main.py`.

## Logo / icon
`assets/heartagram.png` (white on transparent) is the window icon, the menu crest, the face of every target and the
backdrop when the camera is off. Swap the file to change it; if it is missing the game falls back to a drawn star.

## Making it easier / harder (`config.py`, "Rhythm / speed" block)
| Setting | Effect |
|---|---|
| `PACES`, `default_pace` | beats between notes (SLOW 4, NORMAL 2, FAST 1) |
| `approach_time` | how long a note is visible before its beat |
| `window_300/100/50` | timing tolerance |
| `note_radius`, `hit_forgiveness` | target size and hit radius |
| `audio_offset_s` | shift notes against the music if they feel early/late |
| `max_song_seconds` | length of a round (`None` = whole song) |
| `bg_tint`, `bg_vignette` | colour and darkness of the webcam background |

## Files
| File | Responsibility |
|---|---|
| `config.py` | tracks and all tunables |
| `hand_tracker.py` | threaded webcam + best.pt inference, cursor mapping, full-screen background frame |
| `smoothing.py` | EMA and Kalman cursor filters |
| `input_pipeline.py` | smoothing + pinch hysteresis -> cursor and click events |
| `game.py` | beat-synced beatmap, hit detection, scoring (no pygame) |
| `render.py` | Lovemetal-themed drawing: menu, notes, HUD, results |
| `main.py` | menu -> play -> results loop, audio loading |
| `tests/` | headless tests (`python tests/test_core.py`, `test_yolo_tracker.py`, `test_app_smoke.py`) |
