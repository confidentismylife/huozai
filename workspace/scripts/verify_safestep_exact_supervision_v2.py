"""Path-type adapter for the frozen verifier; all numerical checks are unchanged."""
import hashlib
import json
from pathlib import Path
import verify_safestep_exact_supervision as frozen

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/safestep_exact_supervision_20261015'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


if __name__ == '__main__':
    original_path = ROOT/'scripts/verify_safestep_exact_supervision.py'
    plan = json.loads((OUT/'frozen_manifest.json').read_text(encoding='utf-8'))
    assert sha(original_path) == plan['code_sha256']['scripts/verify_safestep_exact_supervision.py']
    repair = {'original_frozen_verifier_sha256': sha(original_path),
        'adapter_sha256': sha(__file__),
        'failure': 'Original sha helper assumed a Path; the final sha(__file__) metadata call passed a str and raised AttributeError before verification.json was written.',
        'change': 'Coerce hash-helper argument to pathlib.Path. Re-run all original numerical checks. Bind verifier metadata to this adapter; original script remains unchanged.',
        'model_labels_training_scores_changed': False}
    path = OUT/'verification_repair.json'
    payload = json.dumps(repair, ensure_ascii=False, indent=2)+'\n'
    if path.exists():
        assert path.read_text(encoding='utf-8') == payload
    else:
        path.write_text(payload, encoding='utf-8')
    frozen.sha = sha
    frozen.__file__ = str(Path(__file__).resolve())
    frozen.main()
