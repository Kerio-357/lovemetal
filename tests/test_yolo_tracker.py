"""Tests HandTracker end-to-end with a *fake* ultralytics + fake camera (no GPU/webcam needed)."""
import dataclasses, sys, time, types
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import cv2
from config import CFG
import hand_tracker as ht
from input_pipeline import CursorController

ROOT = Path(__file__).resolve().parents[1]


class FakeCam:
    def __init__(self, *a, **k): self.n = 0
    def isOpened(self): return True
    def set(self, *a): pass
    def read(self):
        time.sleep(0.005); return True, np.full((480, 640, 3), 40, np.uint8)
    def release(self): pass


class T:                      # mimics a torch tensor enough for _np()
    def __init__(self, a): self.a = np.asarray(a, dtype=np.float32)
    def detach(self): return self
    def cpu(self): return self
    def numpy(self): return self.a


def make_result(kpts, kconf=0.9, box=(0.3, 0.3, 0.7, 0.8), bconf=0.9):
    kpts = np.asarray(kpts, np.float32)
    return types.SimpleNamespace(
        boxes=types.SimpleNamespace(conf=T([bconf]), xyxyn=T([box]), __len__=None) if False else _Boxes(bconf, box),
        keypoints=types.SimpleNamespace(xyn=T([kpts]), conf=T([[kconf] * 21])))


class _Boxes:
    def __init__(self, conf, box): self.conf, self.xyxyn = T([conf]), T([box])
    def __len__(self): return 1


def empty_result():
    return types.SimpleNamespace(boxes=[], keypoints=types.SimpleNamespace(xyn=T(np.zeros((0, 21, 2))), conf=T(np.zeros((0, 21)))))


def hand(pinched, cx=0.4, cy=0.5):
    k = np.zeros((21, 2), np.float32)
    k[0] = (cx, cy + 0.25)            # wrist
    k[9] = (cx, cy)                   # middle mcp  -> palm = 0.25*480 = 120 px
    gap = 0.01 if pinched else 0.15   # thumb-index gap (fraction of width)
    k[4] = (cx - gap / 2, cy - 0.1)
    k[8] = (cx + gap / 2, cy - 0.1)
    return k


class FakeModel:
    task = "pose"
    def __init__(self, path): self.model = types.SimpleNamespace(kpt_shape=(21, 3)); self.mode = "hand"
    def predict(self, frame, **kw):
        assert kw["max_det"] == 1 and kw["verbose"] is False
        return [FakeModel.behaviour()]
    behaviour = staticmethod(lambda: make_result(hand(False)))


def install_fake(model_cls=FakeModel):
    mod = types.ModuleType("ultralytics"); mod.YOLO = model_cls
    sys.modules["ultralytics"] = mod
    cv2.VideoCapture = FakeCam


def wait_for(pred, timeout=3.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if pred(): return True
        time.sleep(0.01)
    return False


def make_tracker(**over):
    cfg = dataclasses.replace(CFG, weights=str(ROOT / "best.pt"), **over)
    return ht.HandTracker(cfg)


def test_extract_and_ratio():
    install_fake()
    tr = make_tracker()
    assert tr._extract(empty_result()) is None
    assert tr._extract(make_result(hand(False), kconf=0.1)) is None        # low keypoint confidence
    kp, box = tr._extract(make_result(hand(False)))
    assert kp.shape == (21, 2)
    open_r = ht.HandTracker.pinch_ratio(kp, 640, 480)
    pinch_r = ht.HandTracker.pinch_ratio(hand(True), 640, 480)
    assert open_r > CFG.pinch_open_thresh > CFG.pinch_close_thresh > pinch_r, (open_r, pinch_r)


def test_end_to_end_cursor_click_and_mirror():
    install_fake()
    FakeModel.behaviour = staticmethod(lambda: make_result(hand(False, cx=0.2, cy=0.5)))
    tr = make_tracker().start()
    try:
        assert wait_for(lambda: tr.get_state().detected)
        s = tr.get_state()
        # hand at x=0.2 in the raw frame -> mirrored to 0.8 -> right side of the screen
        assert s.cursor_xy[0] > CFG.screen_w * 0.6, s.cursor_xy
        assert s.frame is not None and s.frame.shape == (CFG.screen_h, CFG.screen_w, 3)
        assert s.infer_ms >= 0
        ctrl = CursorController(CFG)
        # open -> pinch -> must yield exactly one click through the whole pipeline
        clicks = 0
        FakeModel.behaviour = staticmethod(lambda: make_result(hand(True, cx=0.2, cy=0.5)))
        t_end = time.time() + 1.0
        while time.time() < t_end:
            cur = ctrl.update(tr.get_state(), time.perf_counter())
            clicks += cur.click_time is not None
            time.sleep(0.004)
        assert clicks == 1 and cur.pinched and cur.tracking
    finally:
        tr.stop()


def test_no_hand_and_errors_do_not_crash_and_cursor_stays():
    install_fake()
    FakeModel.behaviour = staticmethod(lambda: make_result(hand(False, 0.5, 0.5)))
    tr = make_tracker().start()
    try:
        ctrl = CursorController(dataclasses.replace(CFG, lost_grace_s=0.1))
        assert wait_for(lambda: tr.get_state().detected)
        for _ in range(20):
            cur = ctrl.update(tr.get_state(), time.perf_counter()); time.sleep(0.01)
        parked = cur.pos
        # 1) hand disappears
        FakeModel.behaviour = staticmethod(empty_result)
        time.sleep(0.4)
        cur = ctrl.update(tr.get_state(), time.perf_counter())
        assert cur.visible and not cur.tracking and cur.pos == parked and cur.click_time is None
        # 2) inference raises -> thread survives, error surfaced, still no crash
        def boom(): raise RuntimeError("CUDA out of memory")
        FakeModel.behaviour = staticmethod(boom)
        assert wait_for(lambda: tr.error is not None)
        assert tr._thread.is_alive() and "CUDA" in tr.error
        cur = ctrl.update(tr.get_state(), time.perf_counter())
        assert cur.pos == parked
        # 3) recovery
        FakeModel.behaviour = staticmethod(lambda: make_result(hand(False, 0.5, 0.5)))
        assert wait_for(lambda: tr.get_state().detected) and wait_for(lambda: tr.error is None)
    finally:
        tr.stop()


def test_slow_model_skips_stale_frames():
    """Model slower than the camera: old frames are dropped, results stay fresh."""
    install_fake()
    def slow(): time.sleep(0.05); return make_result(hand(False))        # 50 ms model, 5 ms camera
    FakeModel.behaviour = staticmethod(slow)
    tr = make_tracker().start()
    try:
        assert wait_for(lambda: tr.get_state().dropped > 10, timeout=4.0)
        s = tr.get_state()
        assert time.perf_counter() - s.t < 0.15                           # newest result is recent, no backlog
        assert 8 < s.track_fps < 25, s.track_fps
    finally:
        tr.stop(); FakeModel.behaviour = staticmethod(lambda: make_result(hand(False)))


def test_startup_failures_are_clear():
    install_fake()
    cfg = dataclasses.replace(CFG, weights=str(ROOT / "nope.pt"))
    try: ht.HandTracker(cfg).start(); assert False
    except FileNotFoundError as e: assert "best.pt" in str(e)

    class DetectModel(FakeModel): task = "detect"
    install_fake(DetectModel)
    try: make_tracker().start(); assert False
    except ValueError as e: assert "pose" in str(e)


def test_missing_ultralytics_message():
    sys.modules.pop("ultralytics", None)
    sys.path_hooks  # real import will fail: ultralytics isn't installed in this sandbox
    try:
        import ultralytics  # noqa
        return  # installed for real -> nothing to test here
    except ImportError:
        pass
    cv2.VideoCapture = FakeCam
    try: make_tracker().start(); assert False
    except ImportError as e: assert "pip install ultralytics" in str(e)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
