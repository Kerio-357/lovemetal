# Lovemetal 🜏

Готическая ритм-игра в стиле osu!, в которую играют **рукой перед веб-камерой**: двигаешь руку — двигается курсор, соединяешь большой и указательный пальцы («щипок») — попадаешь по ноте. В игре три песни HIM.

Работает на Python, OpenCV, Ultralytics YOLOv8-pose (своя модель `best.pt`, 21 точка руки) и Pygame. Это учебный проект по компьютерному зрению.

## Что нужно

- компьютер с Windows 10/11 (на macOS и Linux тоже запустится, команды ниже почти те же);
- веб-камера (без неё можно играть мышкой, см. ниже);
- установленный **Python 3.10, 3.11 или 3.12**;
- около 3 ГБ свободного места: библиотеки для нейросети тяжёлые;
- интернет только для установки.

Никаких расширений и программ для кода (VS Code, PyCharm) ставить не нужно: хватит обычной командной строки.

## Запуск за 5 шагов (Windows)

**1. Скачай проект.** Открой страницу репозитория на GitHub, нажми зелёную кнопку **Code**, затем **Download ZIP**. Распакуй архив, например на рабочий стол. Файл `best.pt` (модель) уже внутри.

**2. Установи Python.** Скачай его на [python.org/downloads](https://www.python.org/downloads/). В первом окне установщика **обязательно поставь галочку «Add python.exe to PATH»**, потом нажми Install Now.

**3. Открой командную строку в папке проекта.** Зайди в распакованную папку, кликни в адресную строку проводника (где написан путь), сотри его, напиши `cmd` и нажми Enter.

**4. Установи библиотеки** (один раз, это займёт несколько минут):

    pip install -r requirements.txt

**5. Запусти игру:**

    python main.py

Откроется меню. Если камера просит разрешение, разреши.

## Играть без камеры

Нет камеры или хочешь просто посмотреть игру? Запусти так: курсор будет мышкой, а левая кнопка заменит щипок.

    python main.py --mouse

## Как играть

1. В меню выбери песню стрелками **↑ ↓** (или наведи курсор-руку), нажми **Enter** или сделай щипок по кнопке START.
2. Поставь руку так, чтобы её было видно целиком, на расстоянии вытянутой руки, при хорошем свете.
3. На экране появляются ноты со сжимающимся кольцом. Наведи курсор на ноту и **сделай щипок** (большой и указательный пальцы вместе) ровно в момент, когда кольцо совпадёт с нотой.
4. Чем точнее удар, тем выше оценка: PERFECT, GREAT, GOOD. Промах — MISS.
5. Если есть барабанная дорожка, ударные в музыке звучат только пока ты попадаешь в такт.

Клавиши: **↑ ↓** песня, **Tab** скорость нот, **M** режим музыки, **Enter** старт, **C** камера на фоне вкл/выкл, **P** индикатор щипка, **[** и **]** чувствительность щипка, **Esc** меню или выход.

## Как добавить песни

Файлы песен нельзя выкладывать в открытый репозиторий (авторские права), поэтому mp3 в проекте нет. Игра запустится и без них, но будет беззвучной. Чтобы появился звук, нужны **твои собственные** mp3-файлы (купленные, из своей коллекции).

### Три песни HIM, которые уже есть в меню

1. Открой папку проекта и зайди в подпапку **`songs`** (рядом с `main.py`; если её нет, создай).
2. Положи туда три файла и **назови их точно так**:

| Песня | Имя файла |
|---|---|
| The Funeral of Hearts | `funeral_of_hearts.mp3` |
| Join Me in Death | `join_me_in_death.mp3` |
| Bury Me Deep Inside | `bury_me_deep_inside.mp3` |

3. Запусти игру заново (`python main.py`). В меню выбери песню: музыка пойдёт сама, ноты ложатся на ритм.

**Частая ошибка.** Windows по умолчанию прячет расширения файлов. Если ты переименуешь `song.mp3` в `funeral_of_hearts.mp3`, а на деле файл называется `funeral_of_hearts.mp3.mp3`, игра его не найдёт. Включи показ расширений: в проводнике вкладка **Вид** → **Показать** → **Расширения имён файлов**.

Версия песни должна быть той же, что у нас: темп и начало бита уже подобраны под конкретные записи (для Join Me in Death это Razorblade Mix). Если у тебя другая версия, ноты съедут. Тогда либо подбери сдвиг (`python main.py --audio-offset 0.05`, значение можно менять), либо подстрой настройки трека, как описано ниже.

### Своя песня вместо одной из трёх

В меню сейчас три места, поэтому новая песня **заменяет** одну из существующих.

1. Положи mp3 в `songs`, например `my_song.mp3`.
2. Открой `config.py` обычным Блокнотом и найди блок `TRACKS`. Измени одну из строк `Track(...)`:

       Track("Моя песня", "my_song.mp3", bpm=120, offset=0.0, duration=200)

3. Что здесь значит:
   - `"Моя песня"` название в меню;
   - `"my_song.mp3"` имя файла из папки `songs`;
   - `bpm` темп песни (ударов в минуту). Его можно узнать на сайтах «tap tempo» или «song BPM»: нажимай в такт музыке;
   - `offset` через сколько секунд от начала файла приходится первый удар. Для начала поставь `0.0`; если ноты опаздывают или спешат, меняй на 0.05–0.3;
   - `duration` длина песни в секундах (посмотри в свойствах файла: 3:20 это 200 секунд).
4. Сохрани файл и запусти игру.

### Стемы: ударные только при точных попаданиях (необязательно)

Без стемов тоже можно играть: песня просто становится тише, пока ты промахиваешься. Чтобы ударные звучали отдельно, один раз выполни в командной строке:

    pip install scipy
    python tools/make_stems.py songs/funeral_of_hearts.mp3

Для этого нужна программа **ffmpeg** (в Windows: `winget install ffmpeg`, затем закрой и открой командную строку заново). Скрипт создаст папку `songs/stems` с двумя файлами для песни. Повтори команду для каждой песни. Разделение простое, поэтому ударные могут звучать неидеально.

## Если что-то не работает

| Проблема | Что сделать |
|---|---|
| `python` не найден | Переустанови Python и поставь галочку «Add python.exe to PATH», затем открой новую командную строку. |
| `pip` не найден | Попробуй `python -m pip install -r requirements.txt`. |
| Чёрный экран или нет камеры | Закрой Zoom, Teams и другие программы, которые держат камеру. Попробуй `python main.py --camera 1`. |
| Игра лагает | Закрой лишние программы. Запусти `python main.py --imgsz 256`. Если есть видеокарта NVIDIA: `python main.py --device cuda:0`. В углу экрана видно FPS и время модели. |
| Щипок не срабатывает или срабатывает сам | Включи индикатор клавишей **P**, добавь света, меняй чувствительность клавишами **[** и **]**. |
| Курсор дрожит | Положи локоть на стол, не загораживай ладонь, добавь света. |
| Ноты идут не в такт музыке | Поправь `audio_offset_s` в `config.py` или запусти `python main.py --audio-offset 0.05`. |
| Нужно просто проверить, что всё установилось | Запусти `python main.py --mouse`. |

Окно игры закрывается клавишей **Esc**. Остановить программу в командной строке можно сочетанием **Ctrl+C**.

## Что внутри (коротко)

- `main.py` запуск, меню и игровой цикл;
- `hand_tracker.py` камера и нейросеть `best.pt`;
- `input_pipeline.py` щипок и курсор, `smoothing.py` сглаживание (фильтр Калмана);
- `game.py` ноты, тайминг и счёт, `render.py` вся графика;
- `config.py` все настройки: песни, темп, чувствительность;
- `tests/` проверки без камеры и звука: `python tests/test_core.py`, `python tests/test_yolo_tracker.py`, `python tests/test_app_smoke.py`.

Подробности для разработчиков на английском ниже.

---

## For developers (English)

### Lovemetal (Osu! CV)

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
