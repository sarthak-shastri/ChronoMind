from pathlib import Path

class VideoLoadError(RuntimeError):
    pass

def _cv2():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError('OpenCV is required for video operations; install requirements.txt') from exc
    return cv2

def get_video_metadata(path):
    cv2=_cv2(); path=str(path); cap=cv2.VideoCapture(path)
    try:
        if not cap.isOpened(): raise VideoLoadError(f'Cannot open video: {path}')
        fps=float(cap.get(cv2.CAP_PROP_FPS) or 0.0); count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0); height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        return {'fps':fps if fps>0 else None,'frame_count':count or None,'duration_seconds':count/fps if fps>0 and count>0 else None,'width':width or None,'height':height or None}
    finally: cap.release()

def motion_scores(path, size=(64, 36)):
    """Compute per-frame low-resolution grayscale motion scores in one sequential decode."""
    import numpy as np
    cv2=_cv2(); cap=cv2.VideoCapture(str(path)); scores=[]; previous=None
    try:
        if not cap.isOpened(): raise VideoLoadError(f'Cannot open video: {path}')
        while True:
            ok, frame=cap.read()
            if not ok: break
            gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            gray=cv2.resize(gray,size,interpolation=cv2.INTER_AREA)
            scores.append(0.0 if previous is None else float(cv2.absdiff(gray,previous).mean()))
            previous=gray
    finally: cap.release()
    if not scores: raise VideoLoadError(f'No decodable frames: {path}')
    return np.asarray(scores,dtype=np.float32)

def read_frames(path,indices):
    cv2=_cv2(); requested=[int(i) for i in indices]
    if any(i < 0 for i in requested):
        raise VideoLoadError(f'Frame indices must be non-negative: {requested}')
    if not requested:
        return []
    cap=cv2.VideoCapture(str(path)); frames=[]; fast_error=None
    try:
        if not cap.isOpened(): raise VideoLoadError(f'Cannot open video: {path}')
        try:
            for idx in requested:
                if not cap.set(cv2.CAP_PROP_POS_FRAMES,idx):
                    raise VideoLoadError(f'Failed seeking to frame {idx} from {path}')
                ok,frame=cap.read()
                if not ok or frame is None:
                    raise VideoLoadError(f'Failed decoding frame {idx} from {path}')
                frames.append(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
            return frames
        except Exception as exc:
            fast_error=exc
    finally:
        cap.release()

    # Some OpenCV/FFmpeg backends report successful seeks but fail decoding
    # distant frames. Reopen independently and decode sequentially to the
    # largest requested index; never substitute a neighboring frame.
    fallback=cv2.VideoCapture(str(path))
    wanted=set(requested); decoded={}; sequential_index=0
    try:
        if not fallback.isOpened():
            raise VideoLoadError(f'Cannot reopen video for sequential decode: {path}') from fast_error
        last=max(requested)
        while sequential_index <= last:
            ok,frame=fallback.read()
            if not ok or frame is None:
                raise VideoLoadError(
                    f'Sequential fallback failed decoding requested frame '
                    f'{sequential_index} from {path} (fast seek failure: {fast_error})'
                ) from fast_error
            if sequential_index in wanted:
                decoded[sequential_index]=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            sequential_index+=1
        absent=wanted.difference(decoded)
        if absent:
            raise VideoLoadError(f'Sequential fallback did not retrieve requested frames {sorted(absent)} from {path}') from fast_error
        return [decoded[idx] for idx in requested]
    finally:
        fallback.release()
