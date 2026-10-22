import sys
from pathlib import Path
from collections import defaultdict
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))


@pytest.mark.parametrize('delivered', [True, False])
def test_delivery_guard_runs_after_four_attempts(monkeypatch, tmp_path, delivered):
    from run_qwen35_timestamped_live_v3 import CorrectedRun
    from stable_baselines3 import PPO
    import fire_evac.timestamped_orchestrator as orchestration
    import run_timestamped_pipeline_followup as followup
    monkeypatch.setattr(followup, 'verify', lambda: None)
    monkeypatch.setattr(PPO, 'load', lambda *args, **kwargs: object())
    call_ids = []
    runner = object.__new__(CorrectedRun)
    runner.rows = defaultdict(list)
    runner.out = tmp_path
    def request(stage, req):
        call_ids.append(req['request_id'])
        if delivered:
            runner.rows[stage].append({'request_id': req['request_id']})
        return {'message': {'content': '{}'}}
    runner.request = request
    def episode(env, model, transport):
        transport(env.snapshot)
        return {'functional_fixture': True}
    monkeypatch.setattr(orchestration, 'execute_episode', episode)
    if delivered:
        runner.proposals()
    else:
        with pytest.raises(RuntimeError, match='incomplete'):
            runner.proposals()
    assert len(call_ids) == len(set(call_ids)) == 4
    assert len(list((tmp_path / 'live_checks').glob('*.json'))) == 4
