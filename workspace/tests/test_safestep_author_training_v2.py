import json
from pathlib import Path
from types import SimpleNamespace
from copy import deepcopy
import torch
from fire_evac.safestep_author_numerics import load_author_inputs
from fire_evac.safestep_author_training_v2 import AuditV2

ROOT=Path(__file__).resolve().parents[1]


def test_hundredth_episode_logs_both_training_and_episode_status(tmp_path):
    cfg=json.loads((ROOT/"schemas/safestep_author_training_20261015.json").read_text(encoding="utf-8"))
    inputs=load_author_inputs(ROOT/cfg["source_directory"])
    audit=AuditV2(tmp_path/"run",inputs,cfg,905)
    audit.env=SimpleNamespace(episode=100,k=4,reset_seed=1,start=(5,1),last_status=SimpleNamespace(name="WIN"),
        last_fed=.1,total_reward=-2.,updates_at_reset=0)
    audit.updates=1;audit.losses=[.2]
    audit.save_hook(SimpleNamespace(policy_dqn=torch.nn.Linear(1,1)))
    progress=json.loads((audit.out/"progress.json").read_text(encoding="utf-8"))
    assert progress["status"]=="training"
    assert progress["episode_status"]=="WIN"
    audit.finish()
