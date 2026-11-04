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

def read_frames(path,indices):
    cv2=_cv2(); cap=cv2.VideoCapture(str(path)); frames=[]
    try:
        if not cap.isOpened(): raise VideoLoadError(f'Cannot open video: {path}')
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES,int(idx)); ok,frame=cap.read()
            if not ok or frame is None: raise VideoLoadError(f'Failed decoding frame {idx} from {path}')
            frames.append(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
    finally: cap.release()
    return frames
