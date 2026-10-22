"""Join, verify and safely restore the full workspace. Python 3.10+; stdlib only."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--volumes",default="backups/2026-10-22")
    ap.add_argument("--output",default="restored_backup")
    ap.add_argument("--verify-only",action="store_true")
    args=ap.parse_args();base=Path(args.volumes);out=Path(args.output)
    info=json.loads((base/"VOLUMES.json").read_text(encoding="utf-8"))
    out.mkdir(parents=True,exist_ok=True);archive=out/"workspace_backup.zip"
    if archive.exists():raise SystemExit(f"Refusing to overwrite {archive}; choose a new output directory.")
    whole=hashlib.sha256()
    with archive.open("wb") as dest:
        for i,part in enumerate(info['parts'],1):
            src=base/part['name'];h=hashlib.sha256();size=0
            with src.open("rb") as f:
                for block in iter(lambda:f.read(4*1024*1024),b""):
                    dest.write(block);whole.update(block);h.update(block);size+=len(block)
            if size!=part['bytes'] or h.hexdigest()!=part['sha256']:
                raise SystemExit(f"Checksum mismatch: {src.name}")
            print(f"Verified volume {i}/{len(info['parts'])}",flush=True)
    if whole.hexdigest()!=info['sha256']:raise SystemExit("Combined archive checksum mismatch")
    with zipfile.ZipFile(archive) as z:
        bad=z.testzip()
        if bad:raise SystemExit(f"ZIP CRC mismatch: {bad}")
        if len(z.infolist())!=info['files']:raise SystemExit("Archive file count mismatch")
        print(f"ZIP CRC verified for {len(z.infolist())} files",flush=True)
        if not args.verify_only:
            target=out.resolve()
            for entry in z.infolist():
                resolved=(target/entry.filename).resolve()
                if not resolved.is_relative_to(target):raise SystemExit("Unsafe ZIP member path")
            z.extractall(out)
            manifest=out/"BACKUP_MANIFEST.jsonl"
            n=0
            with manifest.open(encoding="utf-8") as f:
                for line in f:
                    row=json.loads(line);p=out/row['path'];h=hashlib.sha256()
                    with p.open('rb') as source:
                        for block in iter(lambda:source.read(4*1024*1024),b''):h.update(block)
                    if p.stat().st_size!=row['export_bytes'] or h.hexdigest()!=row['sha256']:
                        raise SystemExit(f"Restored file verification failed: {row['path']}")
                    n+=1
            print(f"Verified {n} restored workspace files",flush=True)
    print("BACKUP_VERIFIED" if args.verify_only else "BACKUP_RESTORED_AND_VERIFIED",flush=True)


if __name__=='__main__':main()
