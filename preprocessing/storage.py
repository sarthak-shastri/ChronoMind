"""Resolve local temporary and optional persistent Google Drive cache locations."""
import os
from pathlib import Path

def resolve_storage_paths(config=None,environ=None):
    """Return hub/frame caches under Drive when configured and a local temp cache.

    Configure with `storage.drive_root` or `REVA_DRIVE_ROOT` after mounting Drive.
    `storage.local_cache_root` / `REVA_LOCAL_CACHE_ROOT` controls temporary files.
    """
    config=config or {}; env=os.environ if environ is None else environ
    storage=config.get('storage',config)
    drive_value=storage.get('drive_root') or env.get('REVA_DRIVE_ROOT')
    local_value=storage.get('local_cache_root') or env.get('REVA_LOCAL_CACHE_ROOT') or 'data/cache'
    local_root=Path(local_value).expanduser()
    persistent_root=Path(drive_value).expanduser() if drive_value else local_root
    paths={'hf_cache_dir':persistent_root/'huggingface','frame_cache_dir':persistent_root/'frames',
           'local_temp_dir':local_root/'tmp','persistent_root':persistent_root}
    for path in paths.values(): path.mkdir(parents=True,exist_ok=True)
    return paths
