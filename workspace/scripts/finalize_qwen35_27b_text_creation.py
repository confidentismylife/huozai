"""Verify the registered text-only model without regenerating any weights."""
from pathlib import Path
import json
import sys

import requests
import auto_evacuation_campaign as c
from run_llm_output_mechanism import save
from run_qwen35_27b_trial import sha
from finalize_auto_evacuation_campaign_review import ledger_check

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/qwen35_27b_text_creation_20261006'
STORE = Path('D:/LLMResearch/OllamaModels14B')
MODEL = 'qwen35-27b-text-q4:latest'


def parameters(text):
    return {line.split(None, 1)[0]: json.loads(line.split(None, 1)[1]) for line in text.splitlines() if line.strip()}


def run():
    assert not (OUT / 'completion_manifest.json').exists()
    source = c.read(OUT / 'source_snapshot.json')
    r = requests.post('http://127.0.0.1:11434/api/show', json={'model': MODEL}, timeout=20)
    r.raise_for_status()
    show = r.json()
    p = STORE / 'manifests/registry.ollama.ai/library/qwen35-27b-text-q4/latest'
    manifest = json.loads(p.read_text(encoding='utf-8'))
    text = [x for x in manifest['layers'] if x['mediaType'] == 'application/vnd.ollama.image.model']
    assert len(text) == 1 and text[0]['digest'] == 'sha256:c7e0e920a0948849beb7074d8368cbd5ac6825fcd1f2450cf851276325d8a6a5'
    assert not any(x['mediaType'] == 'application/vnd.ollama.image.projector' for x in manifest['layers'])
    assert 'vision' not in show['capabilities'] and not show.get('projector_info')
    assert show['model_info'] == source['show']['model_info']
    assert show['template'] == source['show']['template']
    assert parameters(show['parameters']) == parameters(source['show']['parameters'])
    assert show['thinking'] == source['show']['thinking']
    assert show['details']['quantization_level'] == 'Q4_K_M'
    assert 'RENDERER qwen3.5' in show['modelfile'] and 'PARSER qwen3.5' in show['modelfile']
    original = STORE / 'manifests/registry.ollama.ai/library/qwen3.5/27b'
    assert sha(original) == source['manifest_sha256']
    save(OUT / 'text_model_identity.json', {'model': MODEL, 'show': show, 'manifest': manifest,
        'manifest_digest': sha(p), 'text_weights_unchanged': True, 'projector_absent': True,
        'same_template_parser_renderer_parameters': True, 'same_thinking_settings': True,
        'source_model_preserved': True, 'text_weight_bytes': text[0]['size'],
        'excluded_projector_bytes': 931145984, 'weights_requantized': False,
        'new_weights_downloaded': False, 'new_inference': 0,
        'validation_note': 'Parameter display order can vary; compare decoded name/value mappings, not string order.'})
    save(OUT / 'registration_repairs.json', {'first_tag': 'qwen35-27b-text-q4:local',
        'first_show_http_status': 404, 'show_error': c.read(OUT / 'first_show_failure.json'),
        'verified_tag': MODEL, 'second_validation_issue': 'parameter string display ordering only',
        'same_parameter_mappings_verified': True, 'text_weight_digest_unchanged': True,
        'source_model_unchanged': True, 'inference_requests': 0})
    (OUT / 'Modelfile').write_text(show['modelfile'], encoding='utf-8')
    files = [x for x in OUT.iterdir() if x.is_file()] + [Path(__file__), ROOT / 'scripts/create_qwen35_27b_text_q4.py']
    save(OUT / 'completion_manifest.json', {'all_checks_passed': True, 'at_utc': c.stamp(),
        'files': {x.relative_to(ROOT).as_posix(): sha(x) for x in files}, 'new_inference': 0})
    ledger_check(OUT / 'completion_manifest.json')
    print(json.dumps({'model': MODEL, 'no_projector': True, 'same_q4_text_weights': True,
        'text_weight_bytes': text[0]['size'], 'new_inference': 0}, ensure_ascii=False))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    run()
