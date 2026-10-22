"""Fetch public documentation only; never authenticate or execute software."""
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/bristol_source_semantics_audit_20261014"
SOURCES = [
    ("readme_current.txt", "https://data.bristol.ac.uk/datasets/3kn82h2aamsfw2eplor4v2ewk0/README.txt", "official dataset directory"),
    ("directory_current.html", "https://data.bristol.ac.uk/datasets/3kn82h2aamsfw2eplor4v2ewk0/", "official dataset directory"),
    ("historical_game_link.html", "https://uob-my.sharepoint.com/:f:/g/personal/pm19344_bristol_ac_uk/EiqZJtDYl6dKhyWmekUju_oBvYc_TWOVn3o-fjmW3QOqA", "manually transcribed from information_sheet.pdf; case-sensitive token transcription may be imperfect")
]


def main():
    OUT.mkdir(exist_ok=True)
    log = []
    for name, url, provenance in SOURCES:
        target = OUT / name
        started = datetime.now(timezone.utc).isoformat()
        result = subprocess.run(["curl.exe", "--silent", "--show-error", "--location", "--max-time", "20",
            "--max-redirs", "4", "--output", str(target), "--write-out",
            '%{http_code}\n%{url_effective}\n%{content_type}', url], capture_output=True, text=True, timeout=25)
        fields = result.stdout.splitlines()
        entry = {"url": url, "provenance": provenance, "started_at_utc": started, "returncode": result.returncode,
            "status": fields[0] if fields else None, "effective_url": fields[1] if len(fields) > 1 else None,
            "content_type": fields[2] if len(fields) > 2 else None, "error": result.stderr.strip()[:300]}
        if target.exists():
            data = target.read_bytes()
            entry.update(file=name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
        log.append(entry)
        print(json.dumps({k: v for k, v in entry.items() if k != "effective_url"}), flush=True)
    (OUT / "public_source_fetch_log.json").write_text(json.dumps(log, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
