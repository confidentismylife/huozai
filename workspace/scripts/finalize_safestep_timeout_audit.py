"""Record completion of an interrupted-run audit, never training completion."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/safestep_paper_stepwise_timeout_audit_20261015"
REPORT = ROOT / "reports/SafeStep正文训练流程_超时与中途策略审计_2026-10-15.md"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    destination = OUT / "completion_manifest.json"
    assert not destination.exists()
    audit = read(OUT / "audit_summary.json")
    diagnosis = read(OUT / "cached_failure_diagnosis/summary.json")
    assert audit["status"] == "INTERRUPTED_TRAINING_PREFIX_AUDIT_PASS"
    assert diagnosis["status"] == "READ_ONLY_CACHED_DIAGNOSIS_COMPLETE"
    assert audit["completed_12000_episode_runs"] == 0
    assert not audit["final_12000_episode_results_available"]
    assert diagnosis["training_updates"] == diagnosis["LLM_requests"] == 0
    assert not diagnosis["old_final48_crowd_scenarios_accessed"]
    assert digest(ROOT / "scripts/audit_safestep_paper_timeout.py") == audit["auditor_sha256"]
    assert digest(ROOT / "scripts/verify_safestep_paper_stepwise.py") == audit["verifier_sha256"]
    assert digest(ROOT / "scripts/diagnose_safestep_stepwise_cache.py") == diagnosis["script_sha256"]
    for path, expected in diagnosis["source_sha256"].items():
        assert digest(ROOT / path) == expected
    frozen = read(ROOT / "outputs/safestep_paper_stepwise_20261015/frozen_manifest.json")
    for path, expected in frozen["files"].items():
        assert digest(ROOT / path) == expected, path
    for report in audit["reports"]:
        directory = ROOT / report["selection"]["source_directory"]
        assert not (directory / "training_complete.json").exists()
        assert not (directory / "final_model.pth").exists()
        assert not (directory / "summary.json").exists()
    for result in diagnosis["reports"]:
        assert sum(result["trajectory_categories"].values()) == 61
        assert result["verified_prefix_training"]["episodes"] == result["intermediate_checkpoint_episode"]
    counts = {key: sum(r["trajectory_categories"].get(key, 0) for r in diagnosis["reports"])
              for key in ("exit_arrival", "blocked_self_loop", "multi_cell_cycle")}
    assert counts == dict(exit_arrival=26, blocked_self_loop=146, multi_cell_cycle=11)
    links = re.findall(r"\]\(([^)]+)\)", REPORT.read_text(encoding="utf-8"))
    local = [link for link in links if not link.startswith(("https://", "http://"))]
    for link in local:
        assert Path(link).is_absolute() and Path(link).is_file(), link
    selected = [ROOT / r["selection"]["checkpoint"] for r in audit["reports"]]
    csvs = [ROOT / r["selection"]["source_directory"] / "episodes.csv" for r in audit["reports"]]
    artifacts = {p for p in OUT.rglob("*") if p.is_file()}
    artifacts.update(selected + csvs + [REPORT, Path(__file__).resolve(),
        ROOT / "schemas/safestep_stepwise_cached_diagnosis_20261015.json",
        ROOT / "scripts/diagnose_safestep_stepwise_cache.py",
        ROOT / "scripts/audit_safestep_paper_timeout.py",
        ROOT / "scripts/verify_safestep_paper_stepwise.py",
        ROOT / "schemas/safestep_paper_stepwise_20261015.json",
        ROOT / "outputs/safestep_paper_stepwise_20261015/campaign_status.json",
        ROOT / "outputs/safestep_paper_stepwise_20261015/frozen_manifest.json"])
    manifest = dict(
        status="INTERRUPTED_TRAINING_AUDIT_AND_CACHE_DIAGNOSIS_COMPLETE",
        created_at=datetime.now(timezone.utc).isoformat(),
        training_status="ALL_THREE_TIMEOUT_INCOMPLETE",
        completed_12000_episode_runs=0,
        final_training_acceptance="NOT_EVALUATED",
        interim_trajectory_categories=counts,
        frozen_source_hashes_rechecked=len(frozen["files"]),
        report_local_links_checked=len(local),
        earlier_584_artifacts_verified_unchanged_during_prefix_audit=True,
        automatic_training_restart=False,
        training_updates_in_cache_diagnosis=0,
        LLM_requests=0,
        old_final48_crowd_scenarios_accessed=False,
        artifact_sha256={str(p.relative_to(ROOT)): digest(p) for p in sorted(artifacts)},
    )
    destination.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    loaded = read(destination)
    for path, expected in loaded["artifact_sha256"].items():
        assert digest(ROOT / path) == expected, path
    print(json.dumps({k: v for k, v in manifest.items() if k != "artifact_sha256"}, indent=2))
    print("Verified artifact hashes:", len(loaded["artifact_sha256"]))


if __name__ == "__main__":
    main()
