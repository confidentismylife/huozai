"""Read append-only progress without opening the supervisor's replace target."""
import json
from pathlib import Path

OUT=Path(__file__).resolve().parents[1]/'outputs/safestep_continuous450_20261017'
state={}
path=OUT/'progress.jsonl'
if path.exists():
    with path.open('rb') as f:
        f.seek(max(0,path.stat().st_size-32768));rows=f.read().splitlines()
    for row in reversed(rows):
        try:state=json.loads(row);break
        except (ValueError,UnicodeError):pass
for name,key in [('run/run_result.json','run_result'),('analysis_results.json','analysis'),
                 ('analysis_failure.json','analysis_failure'),('completion_manifest.json','manifest')]:
    path=OUT/name
    if path.exists():
        try:state[key]=json.loads(path.read_text('utf-8'))
        except (ValueError,OSError):pass
if 'manifest' in state:
    state['status']=state.pop('manifest')['status']
    state['report']='reports/SafeStep连续450秒候选火灾与四条路线FED_2026-10-17.md'
elif 'analysis_failure' in state:state['status']='ANALYSIS_FAILED_RAW_OUTPUTS_PRESERVED'
elif 'run_result' in state:state['status']='SIMULATION_COMPLETE_ANALYSIS_PENDING' if state['run_result']['passed'] else 'SIMULATION_INCOMPLETE'
if 'analysis' in state:
    state['analysis_summary']={k:v for k,v in state.pop('analysis').items() if k in ['status','final_native_fire_s','route_FED_computed','candidate_routes','route_rankings']}
print(json.dumps(state))
