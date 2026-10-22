"""Independent byte/hash audit of staged files, with recursive archive credential checks."""
import collections
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from build_complete_github_backup import PATTERNS, Sanitizer


def main():
    stage=Path(sys.argv[1]);san=Sanitizer();manifest=stage/'BACKUP_MANIFEST.jsonl'
    rows=[json.loads(line) for line in manifest.read_text(encoding='utf-8').splitlines()]
    by_path={row['path']:row for row in rows}
    files=sorted(x for x in stage.rglob('*') if x.is_file())
    fixes=[];total=0;last=time.monotonic()
    # Only scan large opaque streams for patterns when their distinguishing byte prefix exists.
    needles=(b'sk-',b'ghp_',b'gho_',b'ghu_',b'ghs_',b'ghr_',b'github_pat_',b'AIza',b'AKIA',b'PRIVATE KEY',b'eyJ')
    for i,p in enumerate(files,1):
        rel=p.relative_to(stage).as_posix();data=p.read_bytes();total+=len(data)
        if rel in by_path:
            row=by_path[rel]
            if len(data)!=row['export_bytes'] or hashlib.sha256(data).hexdigest()!=row['sha256']:
                raise RuntimeError('Staged hash mismatch: '+rel)
        # All archive members were inspected during initial sanitization. Reinspect valid ZIP/GZIP
        # to verify containers rather than trusting their compressed raw bytes.
        container=data[:4] in (b'PK\x03\x04',b'PK\x05\x06') or data[:2]==b'\x1f\x8b' or rel.endswith('.tar')
        lower=data.lower()
        text_candidate=any(x in data for x in needles) or any(x in lower for x in (b'bearer',b'api_key',b'api-key',b'apikey',b'access_token',b'access-token',b'accesstoken',b'refresh_token',b'refresh-token',b'client_secret',b'client-secret',b'authorization',b'password'))
        del lower
        if container or text_candidate:
            before=san.counts.copy();clean=san.clean(data,rel)
            if clean!=data:
                p.write_bytes(clean)
                fix={'path':rel,'redactions':dict(san.counts-before)};fixes.append(fix)
                if rel in by_path:
                    row=by_path[rel];row['export_bytes']=len(clean);row['sha256']=hashlib.sha256(clean).hexdigest()
                    row['redactions']=dict(collections.Counter(row.get('redactions',{}))+collections.Counter(fix['redactions']))
                # A second pass must be stable before this file can be archived.
                if san.clean(clean,rel)!=clean:raise RuntimeError('Sanitization is not idempotent: '+rel)
        if i%3000==0 or time.monotonic()-last>30:
            print(json.dumps({'phase':'verify_stage','done':i,'total':len(files),'bytes':total,'fixed_files':len(fixes)}),flush=True);last=time.monotonic()
    manifest.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows),encoding='utf-8')
    summary=json.loads((stage/'BACKUP_SUMMARY.json').read_text(encoding='utf-8'))
    summary['independent_audit']={'files_checked':len(files),'bytes_checked':total,'extra_redacted_files':fixes,'redactions':dict(san.counts),'manifest_hashes_checked':len(rows),'status':'passed'}
    summary['totals']['export_bytes']=sum(r['export_bytes'] for r in rows)
    summary['totals']['modified_files']=sum(bool(r.get('redactions')) for r in rows)
    (stage/'SANITIZATION_NOTES.json').write_text(json.dumps(san.notices,ensure_ascii=False,indent=2),encoding='utf-8')
    summary['archive_fixture_notes']=len(san.notices)
    (stage/'BACKUP_SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'phase':'independent_audit_passed','files':len(files),'bytes':total,'fixed_files':len(fixes),'redactions':dict(san.counts)}),flush=True)


if __name__=='__main__':main()
