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
