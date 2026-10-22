"""Create a separate revision after FDS6.11.1 rejects legacy comma IDs."""
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"outputs/safestep_hotel_fds_smokecheck_20261016"


def main():
    parent = OUT/"runs/hotel_init_v1/hotel_init_v1.fds"
    text = parent.read_text("utf-8")
    renamed = {}
    for m in re.finditer(r"&DEVC\b[^/]*?\bID\s*=\s*'([^']*)'", text, re.S):
        old = m.group(1)
        if "," in old:
            renamed[old] = f"LEGACY_TC_{len(renamed)+1:03d}"
    assert len(renamed) == 232, len(renamed)
    for old, new in renamed.items():
        text = text.replace("'"+old+"'", "'"+new+"'")
    text = text.replace("CHID='hotel_init_v1'", "CHID='hotel_init_v2'")
    path = OUT/"runs/hotel_init_v2"
    path.mkdir(exist_ok=False)
    (path/"hotel_init_v2.fds").write_text(text, "utf-8")
    (path/"derivation.json").write_text(json.dumps({"parent": str(parent.relative_to(ROOT)),
        "trigger": "ERROR899 DEVC should not contain comma; FDS returned0 despite error", "geometry_or_physics_changed": False,
        "renamed_IDs": renamed}, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps({"case": "hotel_init_v2", "renamed_ID_count": len(renamed)}, ensure_ascii=True))


if __name__ == "__main__":
    main()
