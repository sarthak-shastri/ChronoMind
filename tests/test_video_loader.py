from preprocessing.video_loader import get_video_metadata, VideoLoadError

def test_missing_video_fails(tmp_path):
    try: get_video_metadata(tmp_path/'missing.mp4')
    except (VideoLoadError,RuntimeError): return
    raise AssertionError('Missing video must raise')
