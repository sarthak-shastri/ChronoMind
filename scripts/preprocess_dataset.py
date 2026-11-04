import argparse, json, time
from pathlib import Path
from preprocessing.dataset import read_annotations
from preprocessing.config import resolve_video_path
from preprocessing.video_loader import get_video_metadata, read_frames
from preprocessing.frame_sampling import uniform_indices
from preprocessing.transformations import transform_frames


def main():
    p=argparse.ArgumentParser(description='Batch extract sampled frames; skips validated cache entries to resume')
    p.add_argument('annotation_file',type=Path); p.add_argument('video_root',type=Path); p.add_argument('--output',type=Path,default=Path('data/cache/preprocessed')); p.add_argument('--num-frames',type=int,default=8); p.add_argument('--limit',type=int); a=p.parse_args()
    rows=read_annotations(a.annotation_file); rows=rows[:a.limit] if a.limit else rows
    a.output.mkdir(parents=True,exist_ok=True); failures=[]; done=0; started=time.perf_counter()
    for row in rows:
        sid=str(row.get('qa_id',row.get('sample_id',row.get('id','unknown')))); target=a.output/(sid.replace('/','_')+'.npz')
        if target.exists():
            try:
                import numpy as np
                with np.load(target,allow_pickle=False) as z:
                    if z['frames'].shape[0]==a.num_frames: done+=1; continue
            except Exception: pass
        try:
            rel=row.get('video_path',row.get('file_path'))
            path=resolve_video_path(a.video_root,rel); meta=get_video_metadata(path)
            if not meta['frame_count']: raise RuntimeError('Video contains no frames')
            frames=read_frames(path,uniform_indices(meta['frame_count'],a.num_frames).tolist())
            tensor=transform_frames(frames).numpy()
            import numpy as np
            np.savez_compressed(target,frames=tensor)
            done+=1
        except Exception as exc: failures.append({'sample_id':sid,'error':str(exc)})
    elapsed=time.perf_counter()-started
    summary={'processed_or_cached':done,'failures':failures,'elapsed_seconds':elapsed,'seconds_per_success':elapsed/done if done else None}
    (a.output/'preprocess_summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8'); print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
