"""Download and verify public Ollama blobs into a separate D-drive store.

This does not start a service, change environment settings, or run inference.
"""
from pathlib import Path
import hashlib
import json
import os
import shutil
import sys
import time
import urllib.request

import auto_evacuation_campaign as c
import run_llm_output_mechanism as io

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / 'outputs/llm_goal_role_14b_runtime_20261006'
STORE = Path('D:/LLMResearch/OllamaModels14B')
REGISTRY = 'https://registry.ollama.ai/v2/library/qwen3/blobs/'


def file_hash(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def download(blob, log):
    digest = blob['digest']
    assert digest.startswith('sha256:') and len(digest) == 71
    name = digest.replace(':', '-')
    path = STORE / 'blobs' / name
    expected = digest.split(':', 1)[1]
    if path.exists():
        assert path.stat().st_size == blob['size'] and file_hash(path) == expected
        return {'name': name, 'bytes': blob['size'], 'sha256': expected, 'verified': True}
    partial = path.with_name(name + '.partial')
    for attempt in (1, 2):
        offset = partial.stat().st_size if partial.exists() else 0
        assert offset <= blob['size']
        h = hashlib.sha256()
        if offset:
            with partial.open('rb') as previous:
                for chunk in iter(lambda: previous.read(4 * 1024 * 1024), b''):
                    h.update(chunk)
        if offset == blob['size']:
            break
        headers = {'Range': f'bytes={offset}-'} if offset else {}
        req = urllib.request.Request(REGISTRY + digest, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=120) as response:
                assert not offset or response.status == 206, 'Server must honor resume range.'
                mode = 'ab' if partial.exists() else 'xb'
                with partial.open(mode) as target:
                    done, last = offset, -1
                    for chunk in iter(lambda: response.read(4 * 1024 * 1024), b''):
                        target.write(chunk)
                        h.update(chunk)
                        done += len(chunk)
                        assert done <= blob['size']
                        percent = (done * 20 // blob['size']) * 5
                        if percent != last:
                            item = {'at_utc': c.stamp(), 'blob': name, 'bytes': done, 'total': blob['size'], 'percent': percent}
                            log.write(json.dumps(item) + '\n')
                            log.flush()
                            print(json.dumps({'blob': name[:24], 'percent': percent}), flush=True)
                            last = percent
            assert done == blob['size']
            break
        except Exception as exc:
            io.save(RUNTIME / f'download_error_{name}_{attempt}.json', {'at_utc': c.stamp(), 'error': str(exc), 'offset': offset})
            if attempt == 2:
                raise
            time.sleep(1)
    assert partial.stat().st_size == blob['size']
    assert h.hexdigest() == expected, 'Downloaded blob hash mismatch.'
    os.replace(partial, path)
    return {'name': name, 'bytes': blob['size'], 'sha256': expected, 'verified': True}


def main():
    meta = c.read(RUNTIME / 'download_manifest.json')
    assert meta['model'] == 'qwen3:14b' and meta['registry_manifest_sha256'].startswith('bdbd181c33f2')
    manifest = meta['registry_manifest']
    assert STORE.resolve().as_posix() == 'D:/LLMResearch/OllamaModels14B'
    blobs = manifest['layers'] + [manifest['config']]
    required = sum(v['size'] for v in blobs)
    assert shutil.disk_usage('D:/').free >= required + 4 * 1024**3
    (STORE / 'blobs').mkdir(parents=True, exist_ok=True)
    log_path = RUNTIME / 'download_progress_001.jsonl'
    assert not log_path.exists(), 'Preserve existing download log; inspect before resuming.'
    with log_path.open('w', encoding='utf-8') as log:
        checks = [download(blob, log) for blob in blobs]
    target = STORE / 'manifests/registry.ollama.ai/library/qwen3/14b'
    target.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(meta['registry_manifest_url'], headers={'Accept': 'application/vnd.docker.distribution.manifest.v2+json'})
    with urllib.request.urlopen(req, timeout=30) as response:
        raw = response.read()
    assert hashlib.sha256(raw).hexdigest() == meta['registry_manifest_sha256']
    assert not target.exists()
    target.write_bytes(raw)
    io.save(RUNTIME / 'download_verification.json', {'completed_at_utc': c.stamp(), 'all_hashes_passed': True,
        'model': 'qwen3:14b', 'manifest_digest': meta['registry_manifest_sha256'], 'store': STORE.as_posix(),
        'manifest_path': target.as_posix(), 'verified_blobs': checks, 'bytes_downloaded': required,
        'service_started': False, 'environment_changed': False, 'model_inference': 0})
    print('Official 14B model store downloaded and hash verified; no inference.', flush=True)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
