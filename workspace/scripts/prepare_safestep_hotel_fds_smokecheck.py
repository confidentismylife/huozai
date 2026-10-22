"""Freeze a bounded hotel FDS diagnostic and create its first initialization input."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT/"schemas/safestep_hotel_fds_smokecheck_20261016.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text("utf-8-sig"))


def save(path, value):
    assert not path.exists(), path
    path.write_text(value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2), "utf-8")


def main():
    import os
    import pymupdf
    config = load(CONFIG)
    out = ROOT/config["output_directory"]
    assert not out.exists(), out
    previous = ROOT/config["source_candidate_directory"]
    complete = load(previous/"completion_manifest.json")
    protected = {**load(previous/"source_manifest.json")["protected_previous_sha256"],
                 **complete["artifact_sha256"],
                 (previous/"completion_manifest.json").relative_to(ROOT).as_posix(): digest(previous/"completion_manifest.json")}
    for name, sha in protected.items():
        assert digest(ROOT/name) == sha, name
    env = os.environ.copy()
    settings = config["process_environment"]
    env["PATH"] = os.pathsep.join(settings["prepend_PATH"]+[env["PATH"]])
    for key in ("I_MPI_ROOT", "OMP_NUM_THREADS"):
        env[key] = settings[key]
    r = subprocess.run([config["executable"], "-v"], env=env, capture_output=True, timeout=20)
    assert r.returncode == 0, r.stderr
    version = (r.stdout+r.stderr).decode("utf-8", errors="replace")
    assert "FDS-6.11.1" in version, version
    guide = Path("C:/Program Files/firemodels/FDS6/Documentation/Guides_and_Release_Notes/FDS_User_Guide.pdf")
    with pymupdf.open(guide) as doc:
        guide_text = "\n".join(page.get_text() for page in doc)
    assert "FDS-6.11.1-0-gff928db" in guide_text[:2000]
    out.mkdir(parents=True)
    save(out/"frozen_diagnostic_plan.json", config)
    save(out/"source_manifest.json", {"created_at": datetime.now(timezone.utc).isoformat(),
         "protected_previous_sha256": protected,
         "input_sha256": {CONFIG.relative_to(ROOT).as_posix(): digest(CONFIG)},
         "installed_executable": config["executable"], "installed_executable_sha256": digest(Path(config["executable"])),
         "installed_guide": guide.as_posix(), "installed_guide_sha256": digest(guide)})
    save(out/"FDS_6.11.1_local_guide_text.txt", guide_text)
    save(out/"version_query.json", {"command": [config["executable"], "-v"], "returncode": r.returncode,
         "stdout": r.stdout.decode(errors="replace"), "stderr": r.stderr.decode(errors="replace"),
         "initial_direct_launch_observed_error": "Windows0xC0000135(DLLnotfound); resolved by adding bundledbin/mpi to subprocessPATH only",
         "global_environment_modified": False})
    shutil.copyfile(previous/"FD-Gen_LICENSE.md", out/"FD-Gen_LICENSE.md")
    text = (previous/"safestep_hotel_candidate.fds.in").read_text("utf-8")
    text = text.replace("{{REQUIRED_HCN_YIELD_KG_PER_KG}}", "0.01")
    text = text.replace("CHID='safestep_hotel_candidate'", "CHID='hotel_init_v1'")
    text = text.replace("T_END=900.0", "T_END=0.0")
    text = text.replace("TITLE='Unvalidated reconstruction; required HCN yield is unresolved'", "TITLE='SOFTWARE DIAGNOSTIC ONLY; arbitrary HCN yield; not original paper fire'")
    header = "! EXPLORATORY SOFTWARE DIAGNOSTIC 2026-10-16. HCN_YIELD=0.01 is arbitrary.\n! All fire-effect and FED claims excluded; frozen original remains unchanged.\n"
    case = out/"runs/hotel_init_v1"
    case.mkdir(parents=True)
    save(case/"hotel_init_v1.fds", header+text)
    save(case/"derivation.json", {"parent": (previous/"safestep_hotel_candidate.fds.in").relative_to(ROOT).as_posix(),
         "changes": {"CHID": "hotel_init_v1", "T_END": 0, "HCN_YIELD": 0.01},
         "HCN_parameter_status": config["temporary_parameters"]["status"],
         "changes_to_geometry": False})
    print(json.dumps({"output": out.as_posix(), "version": version.strip(), "protected_previous": len(protected)}, ensure_ascii=True))


if __name__ == "__main__":
    main()
