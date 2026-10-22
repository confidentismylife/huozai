"""Replay one fixed complete training run; this is not another performance seed."""
from __future__ import annotations
from pathlib import Path
from datetime import datetime,timezone
import json
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
import run_mappo_six_hour_extension as core
from fire_evac.campaign_extension import write_json
from fire_evac.mappo import FactorizedMAPPO,MAPPOConfig


def main():
    original=core.OUT/'mechanism'/'shared_raw'/'seed_20261301'/core.RAW
    target=core.OUT/'reproducibility_replay'/'seed_20261301'/core.RAW
    while not (original/'summary.json').exists():time.sleep(10)
    scenarios=core.base.make_domain_randomized_trust_bank(24,20261021,'DSG')
    core.TRAIN_SCENARIO_IDS=tuple(s.scenario_id for s in scenarios[:12])
    core.base.make_env=core.factory
    config=MAPPOConfig(hidden_size=64,update_epochs=2,batch_size=64,
        route_entropy_coefficient=.01,flow_entropy_coefficient=.03,plan_imitation_coefficient=0)
    core.recorded_seeded_train_run(target,core.RAW,20261301,config,
        scenarios[:12],scenarios[12:],150000,30000)
    expected=sorted(original.glob('checkpoint_*.pt'))
    comparisons=[]
    for path in expected:
        a,ma=FactorizedMAPPO.load(path)
        b,mb=FactorizedMAPPO.load(target/path.name)
        tensors_a=a.network.state_dict();tensors_b=b.network.state_dict()
        delta=max(float((tensors_a[k]-tensors_b[k]).abs().max()) for k in tensors_a)
        comparisons.append({'checkpoint':path.name,'maximum_absolute_tensor_difference':delta,
                            'same_validation':ma['validation']==mb['validation']})
    pd.testing.assert_frame_equal(pd.read_csv(original/'episode_results.csv'),
                                  pd.read_csv(target/'episode_results.csv'),check_exact=True)
    passed=len(comparisons)==5 and all(c['maximum_absolute_tensor_difference']==0 and c['same_validation'] for c in comparisons)
    result={'passed':passed,'training_seed':20261301,'timesteps':150000,
        'checkpoints':comparisons,'evaluation_CSV_identical':True,
        'scope':'same runtime/CPU, one raw34 full training run; does not prove bitwise portability to other hardware',
        'performance_analysis_inclusion':False,'completed_at_utc':datetime.now(timezone.utc).isoformat()}
    write_json(core.OUT/'full_training_reproducibility_result.json',result)
    if not passed:raise RuntimeError('The full deterministic replay did not reproduce all checkpoints')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
