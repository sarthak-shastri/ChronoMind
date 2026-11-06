from preprocessing.storage import resolve_storage_paths

def test_local_storage_defaults(tmp_path):
    paths=resolve_storage_paths({"storage":{"local_cache_root":str(tmp_path/"local")}},environ={})
    assert paths["persistent_root"]==tmp_path/"local"
    assert paths["hf_cache_dir"].is_dir() and paths["frame_cache_dir"].is_dir()
    assert paths["local_temp_dir"].is_dir()

def test_drive_root_persists_caches_but_keeps_temp_local(tmp_path):
    drive=tmp_path/"mounted_drive"; local=tmp_path/"local_tmp"
    paths=resolve_storage_paths({"storage":{"drive_root":str(drive),"local_cache_root":str(local)}},environ={})
    assert paths["persistent_root"]==drive
    assert paths["hf_cache_dir"]==drive/"huggingface"
    assert paths["frame_cache_dir"]==drive/"frames"
    assert paths["local_temp_dir"]==local/"tmp"
