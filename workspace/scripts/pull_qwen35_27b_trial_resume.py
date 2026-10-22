"""Download the requested official Ollama Qwen3.5 27B model; never start another service."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import shutil
import sys
import time

import psutil
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/qwen35_27b_download_resume_20261006'
MODEL = 'qwen3.5:27b'
STORE = Path('D:/LLMResearch/OllamaModels14B')
URL = 'http://127.0.0.1:11434'


def stamp():
    return datetime.now(timezone.utc).isoformat()


def save(name, value):
    p = OUT / name
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(tmp, p)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def run():
    assert not OUT.exists(), 'Preserve the previous download attempt.'
    assert STORE.is_dir()
    assert shutil.disk_usage(STORE).free > 32 * 1024**3
    OUT.mkdir()
    version = requests.get(URL + '/api/version', timeout=10).json()
    tags = requests.get(URL + '/api/tags', timeout=10).json()
    assert not any(x['name'] == MODEL for x in tags['models'])
    registry = requests.get('https://registry.ollama.ai/v2/library/qwen3.5/manifests/27b', timeout=30)
    registry.raise_for_status()
    expected = registry.headers.get('Docker-Content-Digest')
    digest = 'sha256:' + hashlib.sha256(registry.content).hexdigest()
    manifest_data = registry.json()
    assert manifest_data.get('runner') == 'llamacpp' and manifest_data.get('format') == 'gguf'
    model_layers = [x for x in manifest_data['layers'] if x['mediaType'] == 'application/vnd.ollama.image.model']
    assert len(model_layers) == 1 and model_layers[0]['digest'] == 'sha256:c7e0e920a0948849beb7074d8368cbd5ac6825fcd1f2450cf851276325d8a6a5'
    assert model_layers[0]['size'] == 16547399008
    assert expected in (None, digest)
    (OUT / 'registry_manifest.json').write_bytes(registry.content)
    save('preflight.json', {'at_utc': stamp(), 'model': MODEL, 'version': version,
        'tags_before': tags, 'expected_registry_digest': digest, 'store': str(STORE),
        'memory_before': psutil.virtual_memory()._asdict(), 'source_sha256': sha(__file__),
        'registry_manifest_sha256': sha(OUT / 'registry_manifest.json'),
        'paid_calls': 0, 'inference_requests': 0})
    start = time.monotonic()
    save('status.json', {'state': 'downloading', 'pid': os.getpid(), 'started_at_utc': stamp()})
    last_print = 0
    try:
        with requests.post(URL + '/api/pull', json={'model': MODEL, 'stream': True},
                           stream=True, timeout=(10, 180)) as r:
            r.raise_for_status()
            success = False
            with (OUT / 'progress.jsonl').open('a', encoding='utf-8') as log:
                for line in r.iter_lines():
                    if not line:
                        continue
                    item = json.loads(line)
                    if item.get('error'):
                        raise RuntimeError(item['error'])
                    now = time.monotonic()
                    if now - last_print >= 10 or item.get('status') == 'success':
                        record = {'at_utc': stamp(), 'elapsed_seconds': now-start, **item}
                        log.write(json.dumps(record, ensure_ascii=False) + '\n')
                        log.flush()
                        save('status.json', {'state': 'downloading', 'pid': os.getpid(), **record})
                        print(json.dumps(record, ensure_ascii=False), flush=True)
                        last_print = now
                    if item.get('status') == 'success':
                        success = True
            assert success, 'Pull ended without success.'
        local_manifest = STORE / 'manifests/registry.ollama.ai/library/qwen3.5/27b'
        assert local_manifest.is_file(), 'The service did not write into the expected D: store.'
        assert 'sha256:' + sha(local_manifest) == digest
        current = requests.get(URL + '/api/tags', timeout=10).json()
        model = next(x for x in current['models'] if x['name'] == MODEL)
        assert model['digest'] == digest.removeprefix('sha256:')
        show = requests.post(URL + '/api/show', json={'model': MODEL}, timeout=30).json()
        assert show['details']['quantization_level'] == 'Q4_K_M'
        count = show['model_info']['general.parameter_count']
        assert 26_000_000_000 <= count <= 28_000_000_000
        blobs = []
        manifest = json.loads(local_manifest.read_text(encoding='utf-8'))
        for part in [manifest['config'], *manifest['layers']]:
            p = STORE / 'blobs' / part['digest'].replace(':', '-')
            assert p.is_file() and p.stat().st_size == part['size']
            blobs.append({'digest': part['digest'], 'size': part['size'], 'present': True})
        save('model_identity.json', {'model': model, 'show': show, 'version': version,
            'manifest_digest': digest, 'blobs': blobs,
            'ollama_pull_reported_digest_verification': True,
            'independent_large_blob_hashes_repeated': False})
        save('status.json', {'state': 'complete', 'pid': None, 'completed_at_utc': stamp(),
            'elapsed_seconds': time.monotonic()-start, 'inference_requests': 0})
        files = [x for x in OUT.iterdir() if x.is_file()] + [Path(__file__)]
        save('completion_manifest.json', {'all_checks_passed': True, 'at_utc': stamp(),
            'files': {str(x.relative_to(ROOT)).replace('\\', '/'): sha(x) for x in files},
            'inference_requests': 0})
        print('Qwen3.5 27B Q4_K_M downloaded and identity checked; no inference yet.', flush=True)
    except Exception as exc:
        save('status.json', {'state': 'failed', 'pid': None, 'at_utc': stamp(),
                            'error': repr(exc), 'elapsed_seconds': time.monotonic()-start})
        raise


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    run()
