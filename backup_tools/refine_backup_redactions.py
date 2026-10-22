"""Re-read only previously changed originals with strict credential-format patterns."""
import collections
import hashlib
import json
from pathlib import Path
import sys

from build_complete_github_backup import Sanitizer

source,stage=map(Path,sys.argv[1:3]);manifest=stage/'BACKUP_MANIFEST.jsonl'
rows=[json.loads(x) for x in manifest.read_text(encoding='utf-8').splitlines()]
old_counts=collections.Counter()
for row in rows:old_counts.update(row.get('redactions',{}))
san=Sanitizer();changed=0;restored=0
for row in rows:
    if not row.get('redactions'):continue
    rel=Path(row['path']).relative_to('workspace');original=(source/rel).read_bytes()
    before=san.counts.copy();clean=san.clean(original,rel.as_posix())
    (stage/row['path']).write_bytes(clean)
    row['export_bytes']=len(clean);row['sha256']=hashlib.sha256(clean).hexdigest()
    row['redactions']=dict(san.counts-before)
    changed+=1;restored+=int(clean==original)
manifest.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows),encoding='utf-8')
summary=json.loads((stage/'BACKUP_SUMMARY.json').read_text(encoding='utf-8'))
summary['totals']['export_bytes']=sum(r['export_bytes'] for r in rows)
summary['totals']['modified_files']=sum(bool(r.get('redactions')) for r in rows)
c=collections.Counter()
for row in rows:c.update(row.get('redactions',{}))
summary['workspace_redactions']=dict(c)
overall=collections.Counter(summary.get('redactions',{}));overall.subtract(old_counts);overall.update(c)
summary['redactions']={k:v for k,v in overall.items() if v>0}
summary['redaction_refinement']={'rechecked_files':changed,'restored_false_positive_files':restored,'policy':'strict provider key formats; private-key blocks and credential assignments'}
(stage/'BACKUP_SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'phase':'redaction_refined','rechecked_files':changed,'restored_false_positive_files':restored,'remaining_workspace_redactions':dict(c)}),flush=True)
