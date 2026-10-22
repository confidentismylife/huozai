from copy import deepcopy
import json
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import torch

from fire_evac.safestep_author_numerics import load_author_inputs
from fire_evac.safestep_author_training import (
    Audit,DirectAuthorEnvironment,instrument_original,load_original_modules,tensor_fingerprint)

ROOT=Path(__file__).resolve().parents[1]
CFG=json.loads((ROOT/"schemas/safestep_author_training_20261015.json").read_text(encoding="utf-8"))
SOURCE=ROOT/CFG["source_directory"]


def prepare(out):
    maze,algorithm=load_original_modules(SOURCE)
    inputs=load_author_inputs(SOURCE)
    audit=Audit(out,inputs,CFG,905)
    env=DirectAuthorEnvironment(maze,inputs,audit)
    return maze,algorithm,inputs,audit,env


def test_reset_schedule_repeats_without_forcing_a_single_start(tmp_path):
    maze,algorithm,inputs,a,ea=prepare(tmp_path/"a")
    _,_,_,b,eb=prepare(tmp_path/"b")
    starts=[]
    for _ in range(20):
        xa,_=ea.reset();xb,_=eb.reset()
        assert np.array_equal(xa,xb)
        assert ea.reset_seed==eb.reset_seed
        starts.append(tuple(xa))
    assert len(set(starts))>1
    a.finish();b.finish()


def test_replay_audit_does_not_change_samples_or_random_state(tmp_path):
    maze,algorithm,inputs,audit,env=prepare(tmp_path/"replay")
    original=algorithm.ReplayMemory(1000)
    instrument_original(maze,algorithm,audit)
    recorded=algorithm.ReplayMemory(1000)
    env.reset()
    for i in range(90):
        env.k=2
        transition=(inputs["empty"][i%61],maze.Action.MOVE_LEFT,
                    inputs["empty"][(i+1)%61],-.1*i,maze.Status.PLAYING)
        original.append(transition);recorded.append(transition)
    random.seed(42);expected=original.sample(64);expected_rng=random.getstate()
    random.seed(42);actual=recorded.sample(64);actual_rng=random.getstate()
    assert expected==actual
    assert expected_rng==actual_rng
    assert audit.sample_ids[0]==[list(original.memory).index(x) for x in expected]
    audit.finish()


def test_recorded_optimizer_preserves_exact_original_weight_update(tmp_path):
    torch.set_num_threads(1)
    maze,algorithm,inputs,audit,env=prepare(tmp_path/"update")
    base=algorithm.DQL_NetworkModel(env)
    original_network=algorithm.DQN
    recorded_class=instrument_original(maze,algorithm,audit)
    recorded=recorded_class(env)
    torch.manual_seed(17)
    p1=original_network(61,61,4);t1=original_network(61,61,4)
    p2=deepcopy(p1);t2=deepcopy(t1)
    base.optimizer=torch.optim.Adam(p1.parameters(),lr=.01)
    recorded.optimizer=torch.optim.Adam(p2.parameters(),lr=.01)
    sample=[(inputs["empty"][0],maze.Action.MOVE_RIGHT,inputs["empty"][1],-.2,maze.Status.PLAYING),
            (inputs["empty"][1],maze.Action.MOVE_DOWN,(6,4),0.,maze.Status.WIN),
            (inputs["empty"][2],maze.Action.MOVE_LEFT,inputs["empty"][2],-4.,maze.Status.LOSE)]
    loss1=base.optimize(.95,sample,p1,t1)
    loss2=recorded.optimize(.95,sample,p2,t2)
    assert loss1.item()==loss2.item()
    assert tensor_fingerprint(p1.state_dict())==tensor_fingerprint(p2.state_dict())
    assert tensor_fingerprint(t1.state_dict())==tensor_fingerprint(t2.state_dict())
    assert audit.updates==1
    audit.finish()
