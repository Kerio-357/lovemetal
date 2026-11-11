"""Split a song into two synchronized stems for the adaptive-audio mode.

    python tools/make_stems.py songs/funeral_of_hearts.mp3 [more files ...]

Writes  songs/stems/<name>_backing.ogg  (melody, vocals, pads)
and     songs/stems/<name>_drums.ogg    (percussive part: drums / beat)

Method: harmonic-percussive source separation (HPSS, Fitzgerald 2010). In a
spectrogram, sustained notes form horizontal lines and drum hits vertical lines;
median filters along time / frequency estimate each part, and a soft mask splits
the mix. backing + drums adds up to the original song again.
Needs: numpy, scipy and the ffmpeg program. A neural separator (e.g. Demucs) gives
cleaner drums; save its output under the same two file names to use it instead.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import istft, stft

SR, NFFT, HOP = 44100, 2048, 512


def decode(path):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-ac", "2", "-ar", str(SR),
                          "-f", "f32le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).reshape(-1, 2).T.copy()      # (2, samples)


def encode(path, audio):
    pcm = np.clip(audio.T, -1, 1).astype(np.float32).tobytes()
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ar", str(SR), "-ac", "2",
                    "-i", "-", "-c:a", "libvorbis", "-q:a", "5", str(path)], input=pcm, check=True)


def split(audio):
    spec = [stft(ch, SR, nperseg=NFFT, noverlap=NFFT - HOP)[2].astype(np.complex64) for ch in audio]
    mag = (np.abs(spec[0]) + np.abs(spec[1])) / 2
    harm = median_filter(mag, size=(1, 31))          # smooth along time  -> sustained sounds
    perc = median_filter(mag, size=(31, 1))          # smooth along freq  -> drum hits
    mask = perc ** 2 / (perc ** 2 + harm ** 2 + 1e-12)                # soft "is percussive" mask
    drums = np.stack([istft(s * mask, SR, nperseg=NFFT, noverlap=NFFT - HOP)[1] for s in spec])
    n = min(drums.shape[1], audio.shape[1])
    drums = drums[:, :n].astype(np.float32)
    return audio[:, :n] - drums, drums               # backing = everything that is not drums


def main(paths):
    for p in map(Path, paths):
        out = p.parent / "stems"
        out.mkdir(exist_ok=True)
        audio = decode(p)
        backing, drums = split(audio)
        encode(out / f"{p.stem}_backing.ogg", backing)
        encode(out / f"{p.stem}_drums.ogg", drums)
        rms = lambda x: float(np.sqrt((x ** 2).mean()))
        print(f"{p.name}: {audio.shape[1] / SR:.1f}s  rms mix {rms(audio):.3f}  "
              f"backing {rms(backing):.3f}  drums {rms(drums):.3f}")


if __name__ == "__main__":
    main(sys.argv[1:] or sorted(Path(__file__).resolve().parents[1].glob("songs/*.mp3")))
