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
The mp3 files and their stems are **not** in this repository (copyrighted). Put your own files into `songs/` using the names from `TRACKS` in
`config.py` (`funeral_of_hearts.mp3`, `join_me_in_death.mp3`, `bury_me_deep_inside.mp3`), then run `python tools/make_stems.py` for the stems.
Without them the game still runs, silently.
Audio files live in `songs/`. Each track's file name, BPM and beat offset are in `TRACKS`
in `config.py`; notes are placed on that beat grid. Files are loaded in `start_music()` in `main.py`.

## Adaptive audio (the music reacts to your playing)
`AdaptiveAudio` in `main.py`. Modes (`M` key in the menu, `--music stems|duck|off|auto`, `adaptive_audio` in `config.py`):

| Mode | What you hear |
|---|---|
| `stems` | Backing stem always plays; the drum stem is audible only while you hit on time. A hit keeps the drums on until the next note is due; a miss mutes them. PERFECT / GREAT / GOOD = 100 / 85 / 60 % drum volume. |
| `duck` | Fallback for songs without stems: the song plays at 30 % volume, a hit restores full volume until the next note is due, a miss cuts it to silence for 0.4 s. |
| `off` | Plain playback. |

Stems are two files per song in `songs/stems/`: `<name>_backing.ogg` and `<name>_drums.ogg` (same length, both starting at 0:00).
`python tools/make_stems.py` creates them from the mp3s by harmonic-percussive separation (needs numpy, scipy, ffmpeg);
stems from a neural separator such as Demucs sound cleaner - save them under the same names.

## Hit feedback
Every judged note goes through `on_judgement()` in `main.py`, which calls `play_hit_sound()`, `spawn_hit_particles(x, y)` and
`show_score_popup(text, x, y)`. The hit sound is `assets/hit.wav` (replace the file to change it). Hit rule: `HIT: PINCH`
(pinch on the target) or `HIT: TOUCH` (just move the hand onto it) - menu button, `H` key or `--hit-mode touch`.

## Logo / icon
`assets/heartagram.png` (white on transparent) is the window icon, the menu crest, the face of every target and the
backdrop when the camera is off. Swap the file to change it; if it is missing the game falls back to a drawn star.

## Making it easier / harder (`config.py`, "Rhythm / speed" block)
| Setting | Effect |
|---|---|
| `PACES`, `default_pace` | beats between notes (SLOW 4, NORMAL 2, FAST 1) |
| `approach_time` | how long a note is visible before its beat |
| `window_beats_perfect/great/good` | timing tolerance as a fraction of one beat (0.18 / 0.30 / 0.45) |
| `hit_mode`, `touch_window_beats` | pinch or touch to hit; how early a touch counts |
| `drum_gain_*`, `duck_level`, `stumble_time`, `audio_attack_s/release_s` | adaptive audio behaviour |
| `hit_volume`, `popup_time`, `particle_count` | hit feedback |
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
| `main.py` | menu -> play -> results loop, adaptive audio engine, hit feedback |
| `tools/make_stems.py` | splits a song into backing + drum stems |
| `tests/` | headless tests (`python tests/test_core.py`, `test_yolo_tracker.py`, `test_app_smoke.py`) |
