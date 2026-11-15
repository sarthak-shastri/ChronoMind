import numpy as np
import pytest
from preprocessing import video_loader
from preprocessing.video_loader import get_video_metadata, VideoLoadError

def test_missing_video_fails(tmp_path):
    try: get_video_metadata(tmp_path/'missing.mp4')
    except (VideoLoadError,RuntimeError): return
    raise AssertionError('Missing video must raise')


def test_corrupted_video_fails_clearly(tmp_path):
    path=tmp_path/'corrupt.mp4'
    path.write_bytes(b'not a valid MP4 stream')
    try: get_video_metadata(path)
    except VideoLoadError: return
    raise AssertionError('Corrupted video must raise VideoLoadError')


def test_read_frames_falls_back_to_sequential_decode_and_preserves_order(monkeypatch):
    captures=[]
    class Capture:
        def __init__(self, path):
            self.random_mode=len(captures)==0
            self.position=0
            self.released=False
            captures.append(self)
        def isOpened(self): return True
        def set(self, prop, index):
            self.position=index
            return True
        def read(self):
            if self.random_mode:
                # Simulate FFmpeg failing a distant random seek.
                return False, None
            index=self.position
            self.position+=1
            return True, np.full((2, 3, 3), index, dtype=np.uint8)
        def release(self): self.released=True
    class FakeCV2:
        CAP_PROP_POS_FRAMES=1
        COLOR_BGR2RGB=2
        VideoCapture=Capture
        @staticmethod
        def cvtColor(frame, mode): return frame.copy()
    monkeypatch.setattr(video_loader, "_cv2", lambda: FakeCV2)

    frames=video_loader.read_frames("sample.mp4", [3, 1, 3, 0])

    assert [int(frame[0, 0, 0]) for frame in frames] == [3, 1, 3, 0]
    assert all(frame.shape == (2, 3, 3) for frame in frames)
    assert len(captures) == 2
    assert all(capture.released for capture in captures)


def test_read_frames_raises_if_sequential_fallback_cannot_reach_request(monkeypatch):
    class Capture:
        def __init__(self, path): self.position=0; self.random_mode=not hasattr(Capture,"opened"); Capture.opened=True
        def isOpened(self): return True
        def set(self, prop, index): self.position=index; return True
        def read(self):
            if self.random_mode: return False,None
            if self.position==2: return False,None
            frame=np.full((1,1,3),self.position,dtype=np.uint8); self.position+=1
            return True,frame
        def release(self): pass
    class FakeCV2:
        CAP_PROP_POS_FRAMES=1; COLOR_BGR2RGB=2; VideoCapture=Capture
        @staticmethod
        def cvtColor(frame, mode): return frame
    Capture.opened=False
    monkeypatch.setattr(video_loader,"_cv2",lambda:FakeCV2)
    with pytest.raises(VideoLoadError,match="Sequential fallback failed"):
        video_loader.read_frames("sample.mp4",[2])
