"""Push complete archive volumes in bounded commits; verify remote HEAD afterward."""
import json
from pathlib import Path
import subprocess
import sys
import time

repo=Path(sys.argv[1]);volumes=repo/'backups'/'2026-10-22'
info=json.loads((volumes/'VOLUMES.json').read_text(encoding='utf-8'))


def run(args,**kwargs):
    return subprocess.run(['git',*args],cwd=repo,check=True,**kwargs)


def push():
    for attempt in range(1,4):
        print(json.dumps({'phase':'push','attempt':attempt}),flush=True)
        result=subprocess.run(['git','push','origin','main'],cwd=repo)
        if result.returncode==0:return
        if attempt<3:time.sleep(5)
    raise SystemExit('Push failed after three attempts; local commits remain intact.')


run(['config','core.autocrlf','false'])
run(['config','core.compression','0'])
run(['config','pack.window','0'])
run(['config','pack.threads','2'])
run(['add','--all','--',':(exclude)backups/2026-10-22/workspace_backup.zip.part*'])
changes=subprocess.run(['git','diff','--cached','--quiet'],cwd=repo)
if changes.returncode==1:
    run(['commit','-m','Add complete workspace backup documentation, conversation and browsable sources'])
    push()
elif changes.returncode!=0:raise SystemExit('Unable to inspect staged metadata')

names=[p['name'] for p in info['parts']]
for start in range(0,len(names),8):
    group=names[start:start+8]
    run(['add','--',*[f'backups/2026-10-22/{n}' for n in group]])
    changed=subprocess.run(['git','diff','--cached','--quiet'],cwd=repo)
    if changed.returncode==1:
        run(['commit','-m',f'Add full compressed backup volumes {start+1}-{start+len(group)} of {len(names)}'])
    elif changed.returncode!=0:raise SystemExit('Unable to inspect staged archive volumes')
    push()
    print(json.dumps({'phase':'uploaded_volumes','count':start+len(group),'total':len(names)}),flush=True)

local=run(['rev-parse','HEAD'],capture_output=True,text=True).stdout.strip()
remote=run(['ls-remote','origin','refs/heads/main'],capture_output=True,text=True).stdout.split()[0]
if local!=remote:raise SystemExit('Remote HEAD differs from local HEAD')
print(json.dumps({'phase':'remote_verified','commit':local,'volumes':len(names)}),flush=True)
