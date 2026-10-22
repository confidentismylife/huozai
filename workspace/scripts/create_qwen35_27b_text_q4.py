"""Create a text-only model referencing existing Q4 weights, preserving the source."""
from pathlib import Path
import json
import sys

import requests
import auto_evacuation_campaign as c
from run_llm_output_mechanism import save
from run_qwen35_27b_trial import sha

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/qwen35_27b_text_creation_20261006'
STORE = Path('D:/LLMResearch/OllamaModels14B')
SOURCE = 'qwen3.5:27b'
MODEL = 'qwen35-27b-text-q4:local'
URL = 'http://127.0.0.1:11434'


def api(route, body=None, timeout=30):
    r = requests.get(URL + route, timeout=timeout) if body is None else requests.post(URL + route, json=body, timeout=timeout)
    r.raise_for_status()
    return r.json()


def run():
    assert not OUT.exists(), 'Preserve prior creation.'
    assert not api('/api/ps')['models']
    tags = api('/api/tags')
    assert not any(m['name'] == MODEL for m in tags['models'])
    source = api('/api/show', {'model': SOURCE})
    original = STORE / 'manifests/registry.ollama.ai/library/qwen3.5/27b'
    original_sha = sha(original)
    manifest = json.loads(original.read_text(encoding='utf-8'))
    text = [x for x in manifest['layers'] if x['mediaType'] == 'application/vnd.ollama.image.model']
    vision = [x for x in manifest['layers'] if x['mediaType'] == 'application/vnd.ollama.image.projector']
    assert len(text) == len(vision) == 1
    assert text[0]['digest'] == 'sha256:c7e0e920a0948849beb7074d8368cbd5ac6825fcd1f2450cf851276325d8a6a5'
    parameter_layer = next(x for x in manifest['layers'] if x['mediaType'] == 'application/vnd.ollama.image.params')
    parameters = json.loads((STORE / 'blobs' / parameter_layer['digest'].replace(':', '-')).read_text(encoding='utf-8'))
    OUT.mkdir()
    save(OUT / 'source_snapshot.json', {'show': source, 'manifest': manifest, 'manifest_sha256': original_sha, 'tags': tags})
    payload = {'model': MODEL, 'files': {'qwen35-27b-text-q4.gguf': text[0]['digest']},
        'template': source['template'], 'renderer': 'qwen3.5', 'parser': 'qwen3.5',
        'parameters': parameters, 'license': source['license'],
        'capabilities': [v for v in source['capabilities'] if v != 'vision'], 'stream': False}
    save(OUT / 'create_payload.json', payload)
    result = api('/api/create', payload, timeout=180)
    save(OUT / 'create_response.json', result)
    assert result.get('status') == 'success'
    show = api('/api/show', {'model': MODEL})
    new_manifest_path = STORE / 'manifests/registry.ollama.ai/library/qwen35-27b-text-q4/local'
    new_manifest = json.loads(new_manifest_path.read_text(encoding='utf-8'))
    new_text = [x for x in new_manifest['layers'] if x['mediaType'] == 'application/vnd.ollama.image.model']
    assert len(new_text) == 1 and new_text[0]['digest'] == text[0]['digest']
    assert not any(x['mediaType'] == 'application/vnd.ollama.image.projector' for x in new_manifest['layers'])
    assert 'vision' not in show['capabilities'] and not show.get('projector_info')
    assert show['details']['quantization_level'] == 'Q4_K_M'
    assert show['model_info'] == source['model_info']
    assert show['template'] == source['template'] and show['parameters'] == source['parameters']
    assert 'RENDERER qwen3.5' in show['modelfile'] and 'PARSER qwen3.5' in show['modelfile']
    assert sha(original) == original_sha
    save(OUT / 'text_model_identity.json', {'model': MODEL, 'show': show, 'manifest': new_manifest,
        'manifest_digest': sha(new_manifest_path), 'text_weights_unchanged': True,
        'projector_absent': True, 'same_template_parser_renderer_parameters': True,
        'source_model_preserved': True, 'text_weight_bytes': text[0]['size'],
        'excluded_projector_bytes': vision[0]['size'], 'weights_requantized': False,
        'new_weights_downloaded': False, 'new_inference': 0})
    modelfile = show['modelfile']
    (OUT / 'Modelfile').write_text(modelfile, encoding='utf-8')
    paths = [p for p in OUT.iterdir() if p.is_file()] + [Path(__file__)]
    save(OUT / 'completion_manifest.json', {'all_checks_passed': True, 'created_at_utc': c.stamp(),
        'files': {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}, 'new_inference': 0})
    print(json.dumps({'created': MODEL, 'text_weight_bytes': text[0]['size'],
         'excluded_projector_bytes': vision[0]['size'], 'capabilities': show['capabilities'],
         'quantization': show['details']['quantization_level'], 'inference': 0}, ensure_ascii=False))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    run()
