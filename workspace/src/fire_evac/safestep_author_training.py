"""Instrumentation around the unchanged pinned author DQL.train and optimize.

The only stochastic adaptation is deterministic per-episode reset seeds. Neural
updates, replay draws, action choices, target sync and convergence checks remain
inside the original source methods. Saved files never touch author artifacts.
"""
from __future__ import annotations
import csv
import hashlib
import json
import random
import sys
import time
import types
from collections import deque
from pathlib import Path

import numpy as np
import torch

from .safestep_author_numerics import import_reviewed_source, load_author_inputs, author_rollout


TRANSITION_DTYPE = np.dtype([("episode","<i4"),("step","<i2"),("state","<i2"),
    ("action","i1"),("next_state","<i2"),("reward","<f8"),("status","i1"),("FED","<f8")])


def tensor_fingerprint(state_dict):
    h=hashlib.sha256()
    for key,value in state_dict.items():
        h.update(key.encode());h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def load_original_modules(directory):
    directory=Path(directory)
    maze=import_reviewed_source(directory/"maze.py","author_training_maze")
    previous=sys.modules.get("maze")
    sys.modules["maze"]=maze
    try:
        algorithm=import_reviewed_source(directory/"algorithm_DQL.py","author_training_algorithm")
    finally:
        if previous is None: del sys.modules["maze"]
        else: sys.modules["maze"]=previous
    return maze,algorithm


class Audit:
    def __init__(self,out,inputs,cfg,seed):
        self.out=Path(out);self.out.mkdir(exist_ok=False)
        (self.out/"transitions").mkdir()
        self.cfg=cfg;self.seed=seed;self.inputs=inputs;self.started=time.monotonic()
        self.coordinates=inputs["empty"]+list(inputs["exits"])
        self.index={xy:i for i,xy in enumerate(self.coordinates)}
        self.buffer=np.empty(cfg["transition_chunk_size"],dtype=TRANSITION_DTYPE)
        self.used=0;self.chunk=0;self.transitions=0;self.updates=0
        self.sample_ids=[];self.losses=[];self.syncs=[];self.checks=[];self.initial=[]
        self.episode_rows=[];self.env=None
        self.handle=(self.out/"episodes.csv").open("w",newline="",encoding="utf-8")
        self.writer=csv.DictWriter(self.handle,fieldnames=["episode","reset_seed","start_x","start_y","steps","status","FED","reward","loss","updates","epsilon","elapsed_s"])
        self.writer.writeheader()

    def flush(self):
        if self.used:
            np.savez_compressed(self.out/"transitions"/f"chunk_{self.chunk:05d}.npz",records=self.buffer[:self.used])
            self.chunk+=1;self.used=0

    def append(self,transition):
        state,action,next_state,reward,status=transition
        global_id=self.transitions
        self.buffer[self.used]=(self.env.episode,self.env.k-1,self.index[state],int(action),
            self.index[next_state],reward,int(status.value),float(self.env.last_fed))
        self.used+=1;self.transitions+=1
        if self.used==len(self.buffer):self.flush()
        return global_id

    def save_hook(self,model):
        env=self.env
        row=dict(episode=env.episode,reset_seed=env.reset_seed,start_x=env.start[0],start_y=env.start[1],
            steps=env.k-1,status=env.last_status.name,FED=env.last_fed,reward=env.total_reward,
            loss=self.losses[-1] if self.updates>env.updates_at_reset else 0.0,
            updates=self.updates,epsilon=self.cfg["common_train_arguments"]["exploration_rate"]*
            self.cfg["common_train_arguments"]["exploration_decay"]**self.updates,
            elapsed_s=time.monotonic()-self.started)
        self.writer.writerow(row);self.episode_rows.append(row)
        if env.episode % self.cfg["checkpoint_every_episodes"]==0:
            torch.save(model.policy_dqn.state_dict(),self.out/f"checkpoint_{env.episode:05d}.pth")
        if env.episode % self.cfg["progress_every_episodes"]==0:
            self.handle.flush()
            progress=dict(status="training",seed=self.seed,**row,transitions=self.transitions)
            (self.out/"progress.json").write_text(json.dumps(progress,indent=2),encoding="utf-8")

    def finish(self):
        self.flush();self.handle.flush();self.handle.close()
        samples=np.asarray(self.sample_ids,dtype=np.int64)
        np.savez_compressed(self.out/"optimizer_audit.npz",sample_ids=samples,losses=np.asarray(self.losses))
        (self.out/"target_syncs.json").write_text(json.dumps(self.syncs,indent=2),encoding="utf-8")
        (self.out/"convergence_checks.json").write_text(json.dumps(self.checks,indent=2),encoding="utf-8")


class DirectAuthorEnvironment:
    def __init__(self,maze_module,inputs,audit):
        self.module=maze_module;self.audit=audit;self.episode=0;self.k=1
        self.unwrapped=self
        self.maze_environment=maze_module.Maze(inputs["grid"].copy(),inputs["streams"],inputs["patterns"])
        self.reset_rng=np.random.default_rng(100000+audit.seed)
        audit.env=self

    def reset(self,seed=None,options=None):
        self.episode+=1;self.k=1
        self.reset_seed=int(self.reset_rng.integers(0,2**32,dtype=np.uint64))
        obs=self.maze_environment.reset(seed=self.reset_seed)
        self.start=tuple(int(v) for v in obs)
        self.total_reward=0.0;self.updates_at_reset=self.audit.updates
        self.last_fed=float(self.maze_environment._Maze__FED)
        self.last_status=self.module.Status.PLAYING
        return obs,{}

    def step(self,action):
        obs,reward,status,_,fed=self.maze_environment.step(self.module.Action(action),self.k)
        self.k+=1;self.total_reward+=reward;self.last_status=status;self.last_fed=float(fed)
        return obs,reward,status,False,{"FED value":fed}


def instrument_original(maze,algorithm,audit):
    original_memory=algorithm.ReplayMemory
    class RecordedMemory(original_memory):
        def __init__(self,maxlen):
            super().__init__(maxlen);self.ids=deque(maxlen=maxlen)
        def append(self,transition):
            super().append(transition);self.ids.append(audit.append(transition))
        def sample(self,sample_size):
            sample=super().sample(sample_size)
            lookup={id(t):i for t,i in zip(self.memory,self.ids)}
            audit.sample_ids.append([lookup[id(t)] for t in sample])
            return sample
    algorithm.ReplayMemory=RecordedMemory
    original_network=algorithm.DQN
    class RecordedDQN(original_network):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self.role="policy" if len(audit.initial)==0 else "target"
            audit.initial.append(dict(role=self.role,fingerprint=tensor_fingerprint(self.state_dict())))
        def load_state_dict(self,*args,**kwargs):
            result=super().load_state_dict(*args,**kwargs)
            if self.role=="target":
                audit.syncs.append(dict(episode=audit.env.episode,transitions=audit.transitions,updates=audit.updates))
            return result
    algorithm.DQN=RecordedDQN
    class RecordedModel(algorithm.DQL_NetworkModel):
        def save(self,filename):audit.save_hook(self)
        def optimize(self,*args,**kwargs):
            loss=super().optimize(*args,**kwargs)
            audit.updates+=1;audit.losses.append(float(loss.detach()))
            return loss
    return RecordedModel


def run_training(source,cfg,arm,seed,out,episodes_override=None):
    torch.set_num_threads(cfg["torch_threads_per_process"])
    torch.use_deterministic_algorithms(True)
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    inputs=load_author_inputs(source,cfg["start_time_seconds"])
    maze,algorithm=load_original_modules(source)
    maze.Maze.reward_exit=arm["reward_exit"]
    audit=Audit(out,inputs,cfg,seed)
    env=DirectAuthorEnvironment(maze,inputs,audit)
    model_class=instrument_original(maze,algorithm,audit)
    model=model_class(env)
    original_check=env.maze_environment.check_win_all
    def checked(instance,current_model):
        result=original_check(current_model)
        audit.checks.append(dict(episode=env.episode,all_win=bool(result[0]),win_rate=float(result[1]),
            last20_loss_below_half=bool(all(loss<.5 for loss in audit.losses[-20:]))))
        return result
    env.maze_environment.check_win_all=types.MethodType(checked,env.maze_environment)
    args=dict(cfg["common_train_arguments"])
    args.update(episodes=episodes_override or arm["episodes"],learning_rate=arm["learning_rate"],
                stop_at_convergence=False if episodes_override else arm["stop_at_convergence"])
    h,w,l,episodes,duration=model.train(**args)
    model.policy_dqn.eval()
    torch.save(model.policy_dqn.state_dict(),audit.out/"final_model.pth")
    training=dict(status="trained",arm=arm["id"],seed=seed,episodes=episodes,requested_episodes=args["episodes"],
        transitions=audit.transitions,optimizer_updates=audit.updates,
        wall_seconds=time.monotonic()-audit.started,initial_networks=audit.initial,
        final_weight_fingerprint=tensor_fingerprint(model.policy_dqn.state_dict()),
        early_stopped=episodes<args["episodes"],settings=args,reward_exit=arm["reward_exit"],
        coordinate_order=audit.coordinates,original_train_sha256=hashlib.sha256(
            Path(source,"algorithm_DQL.py").read_bytes()).hexdigest())
    audit.finish()
    # All checkpoints/model initialization are locked before final scoring.
    (audit.out/"training_complete.json").write_text(json.dumps(training,indent=2),encoding="utf-8")
    reference=json.loads(Path(source).parent.joinpath("numeric_run/results.json").read_text(encoding="utf-8"))
    evaluations=[]
    for start in inputs["empty"]:
        run=author_rollout(maze,inputs,model,start)
        run["start"]=start
        evaluations.append(run)
    paper=[]
    for expected in reference["stored_weight_runs"]:
        run=next(r for r in evaluations if list(r["start"])==expected["path"][0])
        paper.append(dict(start=run["start"],status=run["status"],steps=run["steps"],FED=run["FED"],
            path_match=run["path"]==[tuple(x) for x in expected["path"]],
            FED_rounding_match=round(run["FED"],4)==expected["paper_FED"],time_s=run["time_s"]))
    metrics=dict(training=training,paper_starts=paper,
        both_paper_routes_match=all(r["path_match"] and r["FED_rounding_match"] for r in paper),
        all_start_success=sum(r["status"]=="WIN" for r in evaluations),all_start_count=len(evaluations),
        invalid_steps=sum(r["invalid_steps"] for r in evaluations),
        revisits=sum(r["revisits"] for r in evaluations),
        mean_FED=float(np.mean([r["FED"] for r in evaluations])),
        max_FED=max(r["FED"] for r in evaluations),
        over_03=sum(r["FED"]>=.3 for r in evaluations))
    (audit.out/"evaluation_traces.json").write_text(json.dumps(evaluations,indent=2),encoding="utf-8")
    (audit.out/"summary.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
    (audit.out/"progress.json").write_text(json.dumps(training|{"status":"complete"},indent=2),encoding="utf-8")
    return metrics
