"""Copy installed model references to the prepared D store without removing originals."""
from pathlib import Path
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/llm_goal_role_14b_activation_20261006'
SOURCE = Path.home() / '.ollama/models'
DEST = Path('D:/LLMResearch/OllamaModels14B')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    OUT.mkdir(exist_ok=False)
    assert SOURCE.is_dir() and DEST.is_dir() and SOURCE.resolve() != DEST.resolve()
    plan, blobs = [], {}
    for manifest in sorted((SOURCE / 'manifests').rglob('*')):
        if not manifest.is_file():
            continue
        assert not manifest.is_symlink()
        body = json.loads(manifest.read_text(encoding='utf-8'))
        for layer in [body['config'], *body['layers']]:
            name = layer['digest'].replace(':', '-')
            assert name.startswith('sha256-') and len(name) == 71
            blob = SOURCE / 'blobs' / name
            assert blob.is_file() and not blob.is_symlink() and blob.stat().st_size == layer['size']
            blobs[name] = layer['size']
        plan.append(manifest)
    required = sum(size for name, size in blobs.items() if not (DEST / 'blobs' / name).exists())
    assert shutil.disk_usage(DEST).free > required + 2 * 1024**3
    frozen = {str(p.relative_to(ROOT)).replace('\\', '/'): sha(p) for p in (
        ROOT / 'outputs/local_llm_selection.json',
        ROOT / 'outputs/third_round_development_status.json',
        ROOT / 'outputs/llm_goal_role_14b_runtime_20261006/completion_manifest.json',
        ROOT / 'outputs/llm_goal_role_8b_20261006/completion_manifest.json')}
    rows = []
    for name, size in sorted(blobs.items()):
        target = DEST / 'blobs' / name
        existed = target.exists()
        if not existed:
            pending = target.with_name(target.name + '.copying')
            with (SOURCE / 'blobs' / name).open('rb') as src, pending.open('xb') as dst:
                shutil.copyfileobj(src, dst, 8 * 1024 * 1024)
            assert pending.stat().st_size == size and sha(pending) == name[7:]
            assert not target.exists()
            pending.rename(target)
        else:
            assert target.stat().st_size == size and sha(target) == name[7:]
        rows.append({'name': name, 'bytes': size, 'copied': not existed, 'hash_verified': True})
        print(json.dumps(rows[-1]), flush=True)
    for manifest in plan:
        target = DEST / manifest.relative_to(SOURCE)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            assert target.read_bytes() == manifest.read_bytes()
        else:
            with target.open('xb') as f:
                f.write(manifest.read_bytes())
    report = {'at_utc': datetime.now(timezone.utc).isoformat(), 'source_store': str(SOURCE),
              'destination_store': str(DEST), 'original_files_preserved': True,
              'manifests': [p.relative_to(SOURCE).as_posix() for p in plan],
              'blobs': rows, 'copied_bytes': sum(x['bytes'] for x in rows if x['copied']),
              'frozen_sha256': frozen, 'no_symlinks_or_security_changes': True}
    (OUT / 'storage_preparation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Existing models copied and verified; original C store preserved.', flush=True)


if __name__ == '__main__':
    main()
