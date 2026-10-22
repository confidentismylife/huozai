"""Correct only the desktop app's saved model path; never restart processes."""
from pathlib import Path
from datetime import datetime, timezone
import json
import os
import sqlite3
import sys
import urllib.request


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    root = Path(__file__).resolve().parents[1]
    out = root / 'outputs/llm_goal_role_14b_app_path_fix_20261006'
    out.mkdir(exist_ok=False)
    target = Path('D:/LLMResearch/OllamaModels14B')
    old_expected = str(Path.home() / '.ollama/models')
    db = Path(os.environ['LOCALAPPDATA']) / 'Ollama/db.sqlite'
    for model, tag in [('qwen3', '14b'), ('qwen3', '8b'), ('qwen3', '4b'), ('nomic-embed-text', 'latest')]:
        assert (target / 'manifests/registry.ollama.ai/library' / model / tag).is_file()
    def api(path):
        with urllib.request.urlopen('http://127.0.0.1:11434' + path, timeout=5) as response:
            return json.load(response)
    assert not api('/api/ps')['models'], 'Preserve active inference.'
    c = sqlite3.connect(db, timeout=5)
    try:
        c.execute('BEGIN IMMEDIATE')
        old = c.execute('SELECT models FROM settings WHERE id=1').fetchone()
        assert old is not None and old[0] == old_expected, 'Unexpected existing setting.'
        backup = {'at_utc': datetime.now(timezone.utc).isoformat(), 'field': 'settings.models',
                  'old_value': old[0], 'new_value': str(target), 'existing_service_restart_attempted': False,
                  'chat_tables_not_read_or_modified': True}
        (out / 'model_path_before.json').write_text(json.dumps(backup, ensure_ascii=False, indent=2), encoding='utf-8')
        result = c.execute('UPDATE settings SET models=? WHERE id=1 AND models=?', (str(target), old[0]))
        assert result.rowcount == 1
        assert c.execute('SELECT models FROM settings WHERE id=1').fetchone()[0] == str(target)
        c.commit()
    except BaseException:
        c.rollback()
        raise
    finally:
        c.close()
    check = sqlite3.connect(db.as_uri() + '?mode=ro', uri=True, timeout=5)
    try:
        after = check.execute('SELECT models FROM settings WHERE id=1').fetchone()[0]
    finally:
        check.close()
    assert after == str(target)
    status = {**backup, 'saved_value_verified': after, 'inference_calls': 0,
              'state': 'app_model_path_fixed_awaiting_user_restart', 'service_tags': api('/api/tags')}
    (out / 'path_fix_status.json').write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in status.items() if k != 'service_tags'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
