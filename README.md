# Lovemetal 🜏

A gothic, osu!-style rhythm game you play **with your hand in front of a webcam**: move your hand to move the cursor, bring thumb and index finger together (a "pinch") to hit a note. Three songs by HIM are built in.

Built with Python, OpenCV, Ultralytics YOLOv8-pose (a custom `best.pt` model, 21 hand keypoints) and Pygame. This is an academic computer-vision project.

## What you need

- a Windows 10/11 PC (macOS and Linux work too, the commands are almost the same);
- a webcam (or play with the mouse, see below);
- **Python 3.10, 3.11 or 3.12** installed;
- about 3 GB of free disk space (the neural-network libraries are large);
- internet access for the installation only.

You do not need any code editor or extension (VS Code, PyCharm): the plain command line is enough.

## Quick start in 5 steps (Windows)

**1. Download the project.** On the GitHub repository page click the green **Code** button, then **Download ZIP**. Unzip it, for example onto your desktop. The model file `best.pt` is already inside.

**2. Install Python.** Download it from [python.org/downloads](https://www.python.org/downloads/). On the first installer screen **tick "Add python.exe to PATH"**, then click Install Now.

**3. Open a command line in the project folder.** Open the unzipped folder in File Explorer, click the address bar (where the path is shown), delete the path, type `cmd` and press Enter.

**4. Install the libraries** (once; it takes a few minutes):

    pip install -r requirements.txt

**5. Start the game:**

    python main.py

The menu opens. If Windows asks for camera permission, allow it.

## Play without a camera

No webcam, or you just want to try the game? Start it like this: the mouse moves the cursor and the left button acts as the pinch.

    python main.py --mouse

## How to play

1. In the menu pick a song with the **Up / Down** arrows (or point the hand cursor at it) and press **Enter**, or pinch the START button.
2. Place your hand so the whole hand is visible, about an arm's length from the camera, in good light.
3. Notes appear with a shrinking ring. Move the cursor onto a note and **pinch** (thumb and index finger together) exactly when the ring meets the note.
4. The more precise the hit, the better the grade: PERFECT, GREAT, GOOD. A miss is MISS.
5. If the song has a drum track, the drums in the music are audible only while you keep hitting on the beat.

Keys: **Up / Down** song, **Tab** note speed, **M** music mode, **Enter** start, **C** camera background on/off, **P** pinch meter, **[** and **]** pinch sensitivity, **Esc** menu or quit.

## How to add the songs

Song files cannot be published in an open repository (copyright), so the mp3 files are not included. The game runs without them but is silent. To get sound you need **your own** mp3 files (bought or from your own collection).

### The three HIM songs already in the menu

1. Open the project folder and go into the **`songs`** subfolder (next to `main.py`; create it if it is missing).
2. Put three files there and **name them exactly like this**:

| Song | File name |
|---|---|
| The Funeral of Hearts | `funeral_of_hearts.mp3` |
| Join Me in Death | `join_me_in_death.mp3` |
| Bury Me Deep Inside | `bury_me_deep_inside.mp3` |

3. Start the game again (`python main.py`). Pick a song in the menu: the music plays and the notes follow the beat.

**Common mistake.** Windows hides file extensions by default. If you rename `song.mp3` to `funeral_of_hearts.mp3` but the file is really called `funeral_of_hearts.mp3.mp3`, the game will not find it. Turn extensions on: in File Explorer open **View** → **Show** → **File name extensions**.

The recording must be the same version as ours: the tempo and the first beat are tuned to specific recordings (for Join Me in Death it is the Razorblade Mix). With another version the notes drift out of time. Then either shift the timing (`python main.py --audio-offset 0.05`, change the value as needed) or adjust the track settings as described below.

### Your own song instead of one of the three

The menu has three slots, so a new song **replaces** one of the existing ones.

1. Put your mp3 into `songs`, for example `my_song.mp3`.
2. Open `config.py` in Notepad, find the `TRACKS` block and change one of the `Track(...)` lines:

       Track("My Song", "my_song.mp3", bpm=120, offset=0.0, duration=200)

3. What the values mean:
   - `"My Song"` the title shown in the menu;
   - `"my_song.mp3"` the file name inside `songs`;
   - `bpm` the tempo in beats per minute (find it with a "tap tempo" or "song BPM" website: tap along with the music);
   - `offset` seconds from the start of the file to the first beat. Start with `0.0`; if the notes come late or early, try 0.05 to 0.3;
   - `duration` song length in seconds (check the file properties: 3:20 is 200 seconds).
4. Save the file and start the game.

### Stems: drums only on accurate hits (optional)

You can play without stems: the song just gets quieter while you miss. To have the drums as a separate track, run this once in the command line:

    pip install scipy
    python tools/make_stems.py songs/funeral_of_hearts.mp3

This needs the **ffmpeg** program (Windows: `winget install ffmpeg`, then close and reopen the command line). The script creates a `songs/stems` folder with two files per song. Repeat the command for each song. The separation is simple, so the drums may not sound perfect.

## Troubleshooting

| Problem | What to do |
|---|---|
| `python` is not found | Reinstall Python and tick "Add python.exe to PATH", then open a new command line. |
| `pip` is not found | Try `python -m pip install -r requirements.txt`. |
| Black screen or no camera | Close Zoom, Teams and other programs that hold the camera. Try `python main.py --camera 1`. |
| The game lags | Close other programs. Run `python main.py --imgsz 256`. With an NVIDIA graphics card: `python main.py --device cuda:0`. FPS and model time are shown in the corner of the screen. |
| The pinch does not register or fires by itself | Turn on the meter with **P**, add light, change sensitivity with **[** and **]**. |
| The cursor shakes | Rest your elbow on the table, keep your palm uncovered, add light. |
| Notes are out of time with the music | Change `audio_offset_s` in `config.py` or run `python main.py --audio-offset 0.05`. |
| You only want to check that the installation worked | Run `python main.py --mouse`. |

Close the game window with **Esc**. Stop the program in the command line with **Ctrl+C**.

## What is inside (short)

- `main.py` startup, menu and game loop;
- `hand_tracker.py` camera and the `best.pt` neural network;
- `input_pipeline.py` pinch and cursor, `smoothing.py` smoothing (Kalman filter);
- `game.py` notes, timing and score, `render.py` all graphics;
- `config.py` all settings: songs, tempo, sensitivity;
- `tests/` checks that need no camera or sound: `python tests/test_core.py`, `python tests/test_yolo_tracker.py`, `python tests/test_app_smoke.py`.

More details for developers below.

---

## For developers

### Technical reference

Hand-tracked gothic rhythm game: YOLOv8-pose (`best.pt`, 21 hand keypoints) + OpenCV + Pygame.

    pip install -r requirements.txt
    python main.py                  # webcam + best.pt
    python main.py --device cuda:0  # GPU; --imgsz 320 for speed on CPU
    python main.py --mouse          # no camera: mouse + left button as "pinch"
    python main.py --full-song      # whole tracks instead of the first 90 s

Controls: move hand = cursor, pinch thumb+index = click.
Keyboard: UP/DOWN track, TAB pace, M music mode, ENTER start, P pinch meter, [ ] pinch sensitivity, C camera background, ESC menu/quit.

## Songs
The mp3 files and their stems are **not** in this repository (copyrighted). Put your own files into `songs/` using the names from `TRACKS` in
`config.py` (`funeral_of_hearts.mp3`, `join_me_in_death.mp3`, `bury_me_deep_inside.mp3`), then run `python tools/make_stems.py` for the stems.
Without them the game still runs, silently.
Audio files live in `songs/`. Each track's file name, BPM and beat offset are in `TRACKS`
in `config.py`; notes are placed on that beat grid. Files are loaded in `start_music()` in `main.py`.

## Pinch input (Meta Quest style)
The cursor is anchored to the knuckles (`cursor_anchor = "knuckles"`), so it stays still while the fingers move; the ring around it
closes as thumb and index approach and snaps shut on the click. `--anchor pinch_mid` or `index_tip` restores fingertip aiming.

A note is hit only by a **pinch** (thumb tip + index tip together): cursor inside the note + pinch + inside the beat window.
Hovering does nothing, and a held pinch does not repeat. The gesture is measured as
`pinch_ratio = fingertip distance / palm size`; it fires when the ratio drops below `pinch_close_thresh` (0.28) and re-arms above
`pinch_open_thresh` (0.42). Tune with the pinch meter (`P`), the `[` / `]` keys (live), `--pinch 0.32`, or `config.py`.

## Performance
- YOLO runs on a 320 px copy of the frame (`yolo_imgsz`, `--imgsz`); the full-screen picture keeps the camera resolution.
- A grabber thread keeps only the newest camera frame, so a slow model skips frames instead of lagging behind.
- The cursor is extrapolated between camera frames (`cursor_predict_s`) and drawn at 60 fps (`clock.tick(60)`).
- If drawing is too slow, every second frame is skipped while input and audio still run every loop (`frame_skip`, `--no-frame-skip`).
- The background is built at half resolution and stretched (`bg_downscale`); texts, panels and logos are cached instead of re-rendered.
- The model leaves one CPU core free for the game loop (`yolo_threads`); stems are decoded in the background while you are in the menu.
- The HUD shows render fps, tracking rate and model time.

## Adaptive audio (the music reacts to your playing)
`AdaptiveAudio` in `main.py`. Modes (MUSIC button or `M` key in the menu, `--music stems|duck|off|auto`, `adaptive_audio` in `config.py`):

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
`show_score_popup(text, x, y)`. The hit sound is `assets/hit.wav` (replace the file to change it).

## Logo / icon
`assets/heartagram.png` (white on transparent) is the window icon, the menu crest, the face of every target and the
backdrop when the camera is off. Swap the file to change it; if it is missing the game falls back to a drawn star.

## Making it easier / harder (`config.py`, "Rhythm / speed" block)
| Setting | Effect |
|---|---|
| `PACES`, `default_pace` | beats between notes (SLOW 4, NORMAL 2, FAST 1) |
| `approach_time` | how long a note is visible before its beat |
| `window_beats_perfect/great/good` | timing tolerance as a fraction of one beat (0.18 / 0.30 / 0.45) |
| `pinch_close_thresh`, `pinch_open_thresh`, `pinch_instant_ratio` | pinch sensitivity |
| `kalman_process_noise`, `cursor_predict_s` | cursor smoothness vs. responsiveness |
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
