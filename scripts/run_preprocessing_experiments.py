import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import psutil
from preprocessing.cache_manager import FrameCache
from preprocessing.frame_sampling import adaptive_indices, uniform_indices
from preprocessing.hf_access import fetch_dataset_file
from preprocessing.transformations import transform_frames
from preprocessing.video_loader import get_video_metadata, motion_scores, read_frames

CONFIGS = [('E1',4,'uniform'),('E2',8,'uniform'),('E3',12,'uniform'),
           ('E4',16,'uniform'),('E5',32,'uniform'),('E6',8,'adaptive'),('E7',12,'adaptive')]

def run_config(name, count, strategy, paths, repo, revision, cache_root):
    cache = FrameCache(cache_root, namespace=f'{name}:{strategy}:{count}')
    process = psutil.Process()
    rss_peak = process.memory_info().rss
    failures=[]; properties=[]; pass_times=[]
    for pass_number in (1,2):
        started=time.perf_counter(); successes=0
        for rel in paths:
            try:
                video=fetch_dataset_file(repo,rel,revision=revision)
                meta=get_video_metadata(video)
                if not meta['frame_count']: raise RuntimeError('No frame count reported')
                properties.append({k:meta[k] for k in ('width','height','fps','duration_seconds','frame_count')}) if pass_number==1 else None
                key=f'{rel}|{video.stat().st_size}|{count}|{strategy}|224|imagenet'
                frames=cache.get(key)
                if frames is None:
                    if strategy=='uniform': indices=uniform_indices(meta['frame_count'],count)
                    else: indices=adaptive_indices(meta['frame_count'],count,motion_scores(video))
                    frames=read_frames(video,indices.tolist())
                    cache.put(key,frames)
                transform_frames(frames)
                successes+=1
                rss_peak=max(rss_peak,process.memory_info().rss)
            except Exception as exc:
                if pass_number==1: failures.append({'video_path':rel,'error':str(exc)})
        pass_times.append({'pass':pass_number,'successes':successes,
                           'elapsed_seconds':time.perf_counter()-started,
                           'seconds_per_success':(time.perf_counter()-started)/successes if successes else None})
    resolutions=Counter(f"{p['width']}x{p['height']}" for p in properties if p['width'] and p['height'])
    cache_files=list(Path(cache_root).glob('*.npz'))
    total=cache.hits+cache.misses
    return {'experiment':name,'frames':count,'strategy':strategy,'videos_requested':len(paths),
            'successfully_processed_videos':len(paths)-len(failures),'failed_videos':failures,
            'pass_timings':pass_times,'storage_bytes':sum(p.stat().st_size for p in cache_files),
            'cache_hits':cache.hits,'cache_misses':cache.misses,
            'cache_hit_rate':cache.hits/total if total else 0.0,
            'second_pass_hit_rate':1.0 if not failures and paths else None,
            'peak_observed_process_rss_bytes':rss_peak,
            'resolution_counts':dict(resolutions),'video_properties':properties}

def main():
    parser=argparse.ArgumentParser(description='Run E1-E7 on a small selected video list; no model evaluation.')
    parser.add_argument('--sample-report',default='reports/video_sample_analysis.json')
    parser.add_argument('--output-root',default='reports/preprocessing_runs')
    parser.add_argument('--cache-root',default='data/cache/preprocessing_experiments')
    parser.add_argument('--limit-videos',type=int,default=3)
    args=parser.parse_args()
    source=json.loads(Path(args.sample_report).read_text(encoding='utf-8'))
    paths=list(dict.fromkeys(v['video_path'] for v in source['sampled_videos']))[:args.limit_videos]
    revision=source['revision']; repo='ReVA-Benchmark/ReVA'
    run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    root=Path(args.output_root)/run_id; cache_run=Path(args.cache_root)/run_id; root.mkdir(parents=True,exist_ok=True); cache_run.mkdir(parents=True,exist_ok=True); results=[]
    for name,count,strategy in CONFIGS:
        target=cache_run/name; target.mkdir(parents=True,exist_ok=True)
        results.append(run_config(name,count,strategy,paths,repo,revision,target))
        print(f"{name}: {results[-1]['successfully_processed_videos']}/{len(paths)} videos")
    output={'dataset_repo':repo,'dataset_revision':revision,'sample_video_count':len(paths),
            'sample_video_paths':paths,'scope':'three-video pilot; not a dataset-wide benchmark',
            'experiments':results}
    out=root/'results.json'; out.write_text(json.dumps(output,indent=2)+'\n',encoding='utf-8')
    print(f'Results saved to {out}')
    print(json.dumps([{k:r[k] for k in ('experiment','frames','strategy','successfully_processed_videos','storage_bytes','cache_hit_rate','peak_observed_process_rss_bytes')} for r in results],indent=2))
if __name__=='__main__': main()
