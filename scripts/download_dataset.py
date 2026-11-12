import argparse, json, pathlib
from huggingface_hub import HfApi, hf_hub_download

def main():
    p=argparse.ArgumentParser(description='List the dataset manifest or download one exact file')
    p.add_argument('--repo-id',default='ReVA-Benchmark/ReVA')
    p.add_argument('--filename')
    p.add_argument('--local-dir', default=None, help='Target directory destination')
    p.add_argument('--list-files',action='store_true')
    p.add_argument('--repo-type',default='dataset')
    p.add_argument('--manifest',default='manifests/reva_manifest.json')
    a=p.parse_args()
    api=HfApi()
    info=api.repo_info(a.repo_id,repo_type=a.repo_type)
    files=api.list_repo_files(a.repo_id,repo_type=a.repo_type,revision=info.sha)
    manifest={'repo_id':a.repo_id,'repo_type':a.repo_type,'revision':info.sha,'files':files}
    dest=pathlib.Path(a.manifest)
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(f'Manifest {dest}; revision={info.sha}; files={len(files)}')
    if not a.filename or a.list_files:
        for name in files: print(name)
    if a.filename:
        if a.filename not in files: raise SystemExit(f'Filename is not in manifest: {a.filename}')
        print('Downloading exact file:', a.filename)
        out_path = hf_hub_download(
            repo_id=a.repo_id,
            repo_type=a.repo_type,
            filename=a.filename,
            revision=info.sha,
            local_dir=a.local_dir,
        )
        print('Downloaded:', out_path)
if __name__=='__main__': main()
