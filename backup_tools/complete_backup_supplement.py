import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).parent))
from build_complete_github_backup import Sanitizer

stage=Path(sys.argv[1])
san=Sanitizer()
sources=[
    (Path(r'D:\UserFolders\Downloads\Exit choice.xlsx'),Path('provided_materials/Exit choice.xlsx')),
    (Path(r'D:\UserFolders\Downloads\Exit choice ppt.pptx'),Path('provided_materials/Exit choice ppt.pptx')),
    (Path(r'D:\UserFolders\Desktop\火灾行人轨迹疏散论文呢'),Path('provided_materials/papers')),
    (Path(r'C:\Users\摩天轮大盗\Documents\ChatGPT\LLM探索\artifacts\llm_ppo_paper_package_20261020'),Path('workspace_path_supplement/artifacts/llm_ppo_paper_package_20261020')),
    (Path(r'C:\Users\摩天轮大盗\Documents\ChatGPT\LLM探索\scripts\build_llm_ppo_paper_package.py'),Path('workspace_path_supplement/scripts/build_llm_ppo_paper_package.py')),
]
rows=[]
for source,dest in sources:
    if not source.exists():continue
    files=[source] if source.is_file() else sorted(x for x in source.rglob('*') if x.is_file() and '.git' not in x.parts)
    for p in files:
        rel=dest if source.is_file() else dest/p.relative_to(source)
        if san.excluded(rel):continue
        data=p.read_bytes();clean=san.clean(data,rel.as_posix());q=stage/rel
        q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(clean)
        rows.append({'path':rel.as_posix(),'source_bytes':len(data),'export_bytes':len(clean),'source_sha256':hashlib.sha256(data).hexdigest(),'sha256':hashlib.sha256(clean).hexdigest()})
(stage/'SUPPLEMENT_MANIFEST.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows),encoding='utf-8')
print(json.dumps({'supplement_files':len(rows),'bytes':sum(x['export_bytes'] for x in rows),'redactions':dict(san.counts),'notices':san.notices}),flush=True)
