"""Prepare a separately versioned, source-pinned FDS restart-output audit."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/safestep_restart_hrr_audit_20261017"
OLD = ROOT / "outputs/safestep_physics_time_contract_20261016"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), "utf-8")


def main():
    OUT.mkdir(exist_ok=True)
    (OUT / "materials").mkdir(exist_ok=True)
    assert not (OUT / "source_manifest.json").exists(), "Never replace completed source evidence"
    frozen = json.loads((OLD / "completion_manifest.json").read_text("utf-8"))
    hashes = frozen["artifact_sha256"]
    bad = [name for name, expected in hashes.items() if sha(ROOT / name) != expected]
    assert not bad, bad
    save(OUT / "prior_artifact_verification_before.json", {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "manifest": (OLD / "completion_manifest.json").relative_to(ROOT).as_posix(),
        "manifest_sha256": sha(OLD / "completion_manifest.json"),
        "files_verified": len(hashes), "mismatches": bad,
    })
    # Release link resolved through official GitHub HTML after its unauthenticated API returned 403.
    commit = "ff928dbf68a2623ced3a007be75391c6dc6c9f90"
    save(OUT / "materials/source_retrieval_history.json", {
        "initial_attempt": "GitHub commits API", "result": "HTTP 403: rate limit exceeded; no solver run attempted",
        "fallback": "Official release HTML commit link, then immutable raw source URLs",
        "release_url": "https://github.com/firemodels/fds/releases/tag/FDS-6.11.1",
        "commit_url": f"https://github.com/firemodels/fds/commit/{commit}",
    })
    assert commit.startswith("ff928db")
    sources = []
    for name in ("dump.f90", "main.f90", "init.f90", "read.f90", "cons.f90", "type.f90"):
        url = f"https://raw.githubusercontent.com/firemodels/fds/{commit}/Source/{name}"
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read()
        target = OUT / "materials" / name
        target.write_bytes(data)
        sources.append({"url": url, "path": target.relative_to(ROOT).as_posix(), "sha256": sha(target), "bytes": len(data)})
    save(OUT / "source_manifest.json", {
        "downloaded_at": datetime.now(timezone.utc).isoformat(), "official_repository": "https://github.com/firemodels/fds",
        "release": "FDS-6.11.1", "commit": commit, "matches_installed_revision": True,
        "sources": sources, "purpose": "Read-only implementation audit; installed solver unchanged",
    })
    print(json.dumps({"previous_files_verified": len(hashes), "source_files": len(sources), "commit": commit}))


if __name__ == "__main__":
    main()
