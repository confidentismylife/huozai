"""Additive v2: unchanged original live stages; robust same-model pipeline."""
import sys
from run_qwen35_timestamped_live import Run, MODEL, strict_json
from run_timestamped_development_campaign import ROOT, PREP, read, save


class CorrectedRun(Run):
    def proposals(self):
        from run_timestamped_pipeline_followup import verify
        from stable_baselines3 import PPO
        import torch
        from fire_evac.behavior import RuleBehavior
        from fire_evac.dynamic_conflicts import DynamicConflictScenario
        from fire_evac.timestamped_pipeline import TimestampedControlEnv, ADVICE_PREFIX, CANDIDATE_SCHEMA
        from fire_evac.timestamped_orchestrator import execute_episode
        import json
        verify()
        if len(self.rows['live_proposals']) != 4:
            raise RuntimeError('Live proposal responses incomplete; functional fallbacks recorded separately.')
        torch.set_num_threads(1)
        plan = read(PREP / 'manifest.json')
        model = PPO.load(plan['ppo_models']['601']['raw_llm_ppo'], device='cpu')
        for i, record in enumerate(plan['scenarios'][:4]):
            scene = DynamicConflictScenario(**record)
            env = TimestampedControlEnv(RuleBehavior(), (scene,))
            env.reset(seed=1407200000 + i, options={'scenario': scene})
            def transport(observation):
                req = {'request_id': f'LIVE_{scene.scenario_id}', 'payload': {
                    'model': MODEL, 'messages': [{'role': 'user', 'content': ADVICE_PREFIX + json.dumps(observation.public_data(), separators=(',', ':'))}],
                    'format': CANDIDATE_SCHEMA, 'think': False, 'stream': False, 'keep_alive': 0,
                    'options': {'temperature': 0, 'num_ctx': 1024, 'num_batch': 32, 'num_thread': 8, 'num_gpu': 0, 'num_predict': 192}}}
                response = self.request('live_proposals', req)
                return strict_json(response['message']['content'])
            result = execute_episode(env, model, transport)
            save(self.out / 'live_checks' / f'{scene.scenario_id}.json', result)
            env.close()
        verify()


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    from run_timestamped_pipeline_followup import verify
    verify()
    CorrectedRun().execute()
