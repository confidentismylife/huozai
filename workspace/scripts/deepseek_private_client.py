"""Read a Windows-user DPAPI credential; never log the credential or headers."""
from __future__ import annotations
import argparse
import json
import shutil
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def credential():
    command = (
        "$ErrorActionPreference='Stop'; "
        "$cipher=Get-Content -LiteralPath '.runtime/secrets/deepseek-key.dpapi' -Raw; "
        "$secret=ConvertTo-SecureString $cipher.Trim(); "
        "[System.Net.NetworkCredential]::new('', $secret).Password"
    )
    shell = shutil.which('pwsh')
    if shell is None:
        raise RuntimeError('PowerShell 7 is required to load this private credential')
    result = subprocess.run([shell, '-NoProfile', '-Command', command], cwd=ROOT,
        capture_output=True, check=True)
    value = result.stdout.decode('ascii').strip()
    if not value.startswith('sk-') or not 20 <= len(value) <= 200:
        raise ValueError('The private credential could not be loaded correctly')
    return value


def request(path, payload=None, timeout=30):
    key = credential()
    body = json.dumps(payload, ensure_ascii=False).encode('utf-8') if payload is not None else None
    req = urllib.request.Request('https://api.deepseek.com' + path, data=body,
        headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode('utf-8', errors='replace').replace(key, '[REDACTED]')
        raise RuntimeError(f'Official DeepSeek endpoint HTTP {exc.code}: {detail[:1000]}') from None


def probe():
    out = ROOT / 'outputs/deepseek_access_probe_20261014'
    out.mkdir(exist_ok=True)
    try:
        response = request('/models')
        result = {'authenticated': True, 'model_ids': [r['id'] for r in response['data']],
            'endpoint': 'https://api.deepseek.com/models', 'generation_requests': 0}
    except Exception as exc:
        result = {'authenticated': False, 'error': str(exc),
            'endpoint': 'https://api.deepseek.com/models', 'generation_requests': 0}
    (out / 'access_result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['probe'])
    parser.parse_args()
    probe()
