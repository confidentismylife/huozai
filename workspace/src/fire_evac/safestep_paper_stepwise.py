"""Paper Table2/Eq9 training flow on the unedited original hotel environment."""
from __future__ import annotations
from collections import deque
from copy import deepcopy
import csv
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from .safestep_author_numerics import AuthorDQN,author_rollout,load_author_inputs,import_reviewed_source
from .safestep_author_training import TRANSITION_DTYPE,tensor_fingerprint


def selected_td_loss(policy,target,identity,states,actions,next_states,rewards,terminal,gamma):
    """Terminal samples never query a nonexistent exit observation."""
    q=policy(identity[states]).gather(1,actions[:,None]).squeeze(1)
    with torch.no_grad():
        labels=rewards.clone()
        active=~terminal
        if active.any():
            labels[active]+=gamma*target(identity[next_states[active]]).max(1).values
    loss=torch.mean((q-labels)**2)
    return loss,q,labels


class Journals:
    def __init__(self,out,cfg):
        self.out=Path(out);self.out.mkdir(exist_ok=False)
        for name in ("transitions","optimizer","update_contexts"):(self.out/name).mkdir()
        self.cfg=cfg;self.transitions=0;self.updates=0;self.tc=0;self.oc=0
        self.tbuf=np.empty(cfg["transition_chunk"],dtype=TRANSITION_DTYPE);self.tused=0
        n=cfg["optimizer_chunk"];b=cfg["batch_size"]
        self.obuf={"ids":np.empty((n,b),dtype=np.uint32),"q":np.empty((n,b),dtype=np.float32),
                   "targets":np.empty((n,b),dtype=np.float32),"loss":np.empty(n,dtype=np.float64),
                   "global_step":np.empty(n,dtype=np.uint32),"episode":np.empty(n,dtype=np.uint32),
                   "target_last_sync_step":np.empty(n,dtype=np.uint32)}
        self.oused=0
        self.handle=(self.out/"episodes.csv").open("w",encoding="utf-8",newline="")
        self.writer=csv.DictWriter(self.handle,fieldnames=["episode","reset_seed","start_x","start_y","steps",
             "episode_status","FED","reward","mean_update_loss","updates","epsilon","global_steps","elapsed_s"])
        self.writer.writeheader()

    def transition(self,row):
        self.tbuf[self.tused]=row;self.tused+=1;self.transitions+=1
        if self.tused==len(self.tbuf):self.flush_transitions()

    def update(self,ids,q,labels,loss,global_step,episode,sync):
        i=self.oused
        self.obuf["ids"][i]=ids
        self.obuf["q"][i]=q.detach().numpy()
        self.obuf["targets"][i]=labels.detach().numpy()
        self.obuf["loss"][i]=loss
        self.obuf["global_step"][i]=global_step
        self.obuf["episode"][i]=episode
        self.obuf["target_last_sync_step"][i]=sync
        self.oused+=1;self.updates+=1
        if self.oused==len(self.obuf["loss"]):self.flush_optimizer()

    def flush_transitions(self):
        if self.tused:
            np.savez_compressed(self.out/"transitions"/f"chunk_{self.tc:05d}.npz",records=self.tbuf[:self.tused])
            self.tc+=1;self.tused=0

    def flush_optimizer(self):
        if self.oused:
            np.savez_compressed(self.out/"optimizer"/f"chunk_{self.oc:05d}.npz",**{k:v[:self.oused] for k,v in self.obuf.items()})
            self.oc+=1;self.oused=0

    def close(self):
        self.flush_transitions();self.flush_optimizer();self.handle.flush();self.handle.close()


class GreedyPolicy:
    def __init__(self,network,empty,action_class):
        self.network=network;self.index={xy:i for i,xy in enumerate(empty)}
        self.identity=torch.eye(len(empty));self.action_class=action_class

    def predict(self,state):
        with torch.no_grad():
            action=int(self.network(self.identity[self.index[tuple(state)]]).argmax())
        return self.action_class(action)


def run(source,cfg,seed,out,episodes=None):
    started=time.monotonic()
    torch.set_num_threads(cfg["torch_threads"]);torch.use_deterministic_algorithms(True)
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    inputs=load_author_inputs(source,cfg["start_time_s"])
    maze=import_reviewed_source(Path(source)/"maze.py","stepwise_original_maze")
    maze.Maze.reward_exit=cfg["reward_exit"]
    environment=maze.Maze(inputs["grid"].copy(),inputs["streams"],inputs["patterns"])
    coordinates=inputs["empty"]+list(inputs["exits"])
    index={xy:i for i,xy in enumerate(coordinates)}
    identity=torch.eye(61)
    policy=AuthorDQN(61);target=AuthorDQN(61);target.load_state_dict(policy.state_dict())
    initial=tensor_fingerprint(policy.state_dict())
    optimizer=torch.optim.Adam(policy.parameters(),lr=cfg["learning_rate"])
    journal=Journals(out,cfg)
    ring=np.empty(cfg["replay_capacity"],dtype=TRANSITION_DTYPE)
    memory=deque(maxlen=cfg["replay_capacity"])
    reset_rng=np.random.default_rng(100000+seed)
    epsilon=cfg["epsilon_start"];last_sync=0;syncs=[];audited=[]
    total_episodes=episodes or cfg["episodes"]
    audit_indices=set(cfg["audit_update_indices"])
    for ep in range(1,total_episodes+1):
        reset_seed=int(reset_rng.integers(0,2**32,dtype=np.uint64))
        obs=environment.reset(seed=reset_seed)
        start=tuple(int(v) for v in obs);state=index[start]
        reward_sum=0.;loss_sum=0.;ep_updates=0
        for step in range(1,302):
            if np.random.random()<epsilon:action=int(random.choice(maze.Maze.actions))
            else:
                with torch.no_grad():action=int(policy(identity[state]).argmax())
            obs,reward,status,_,fed=environment.step(maze.Action(action),step)
            next_state=index[tuple(int(v) for v in obs)]
            row=(ep,step,state,action,next_state,reward,int(status.value),float(fed))
            global_id=journal.transitions
            ring[global_id%cfg["replay_capacity"]]=row;memory.append(global_id);journal.transition(row)
            global_step=journal.transitions
            reward_sum+=reward
            if len(memory)>cfg["batch_size"]:
                ids=np.asarray(random.sample(memory,cfg["batch_size"]),dtype=np.uint32)
                batch=ring[ids%cfg["replay_capacity"]]
                states=torch.from_numpy(batch["state"].astype(np.int64))
                actions=torch.from_numpy(batch["action"].astype(np.int64))
                next_states=torch.from_numpy(batch["next_state"].astype(np.int64))
                rewards=torch.from_numpy(batch["reward"].astype(np.float32))
                terminal=torch.from_numpy(batch["status"]!=maze.Status.PLAYING.value)
                loss,q,labels=selected_td_loss(policy,target,identity,states,actions,next_states,rewards,terminal,cfg["gamma"])
                update_number=journal.updates+1
                context=None
                if update_number in audit_indices:
                    context=dict(update=update_number,global_step=global_step,ids=ids.tolist(),
                        policy_before=deepcopy(policy.state_dict()),target=deepcopy(target.state_dict()),
                        optimizer_before=deepcopy(optimizer.state_dict()),loss=float(loss.detach()),
                        q=q.detach().clone(),labels=labels.detach().clone())
                optimizer.zero_grad();loss.backward();optimizer.step()
                loss_value=float(loss.detach())
                journal.update(ids,q,labels,loss_value,global_step,ep,last_sync)
                if context is not None:
                    context["policy_after"]=deepcopy(policy.state_dict())
                    torch.save(context,journal.out/"update_contexts"/f"update_{update_number:07d}.pt")
                    audited.append(update_number)
                loss_sum+=loss_value;ep_updates+=1
            if global_step%cfg["target_sync_global_steps"]==0:
                target.load_state_dict(policy.state_dict());last_sync=global_step
                syncs.append(global_step)
            state=next_state
            if status!=maze.Status.PLAYING:break
        record=dict(episode=ep,reset_seed=reset_seed,start_x=start[0],start_y=start[1],steps=step,
            episode_status=status.name,FED=float(fed),reward=reward_sum,
            mean_update_loss=loss_sum/ep_updates if ep_updates else 0.,updates=journal.updates,
            epsilon=epsilon,global_steps=journal.transitions,elapsed_s=time.monotonic()-started)
        journal.writer.writerow(record)
        if ep%cfg["checkpoint_every_episodes"]==0:
            torch.save(policy.state_dict(),journal.out/f"checkpoint_{ep:05d}.pth")
        if ep%cfg["progress_every_episodes"]==0:
            journal.handle.flush()
            (journal.out/"progress.json").write_text(json.dumps(record|{"status":"training","seed":seed},indent=2),encoding="utf-8")
        epsilon*=cfg["epsilon_decay_per_episode"]
    journal.close()
    torch.save(policy.state_dict(),journal.out/"final_model.pth")
    (journal.out/"target_syncs.json").write_text(json.dumps(syncs),encoding="utf-8")
    training=dict(status="trained",seed=seed,episodes=total_episodes,transitions=journal.transitions,
        optimizer_updates=journal.updates,initial_policy_fingerprint=initial,
        final_policy_fingerprint=tensor_fingerprint(policy.state_dict()),
        wall_seconds=time.monotonic()-started,coordinate_order=coordinates,audited_update_indices=audited,
        reward_exit=cfg["reward_exit"],early_stopped=False)
    (journal.out/"training_complete.json").write_text(json.dumps(training,indent=2),encoding="utf-8")
    policy.eval();greedy=GreedyPolicy(policy,inputs["empty"],maze.Action)
    traces=[]
    for start in inputs["empty"]:
        record=author_rollout(maze,inputs,greedy,start);record["start"]=start;traces.append(record)
    reference=json.loads((Path(source).parent/"numeric_run/results.json").read_text(encoding="utf-8"))
    paper=[]
    for expected in reference["stored_weight_runs"]:
        record=next(r for r in traces if list(r["start"])==expected["path"][0])
        paper.append(dict(start=record["start"],status=record["status"],steps=record["steps"],FED=record["FED"],
            path_match=record["path"]==[tuple(x) for x in expected["path"]],
            FED_rounding_match=round(record["FED"],4)==expected["paper_FED"],time_s=record["time_s"]))
    summary=dict(training=training,paper_starts=paper,
        both_paper_routes_match=all(r["path_match"] and r["FED_rounding_match"] for r in paper),
        all_start_success=sum(r["status"]=="WIN" for r in traces),all_start_count=61,
        invalid_steps=sum(r["invalid_steps"] for r in traces),revisits=sum(r["revisits"] for r in traces),
        mean_FED=float(np.mean([r["FED"] for r in traces])),max_FED=max(r["FED"] for r in traces),
        over_03=sum(r["FED"]>=.3 for r in traces))
    (journal.out/"evaluation_traces.json").write_text(json.dumps(traces,indent=2),encoding="utf-8")
    (journal.out/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    (journal.out/"progress.json").write_text(json.dumps(training|{"status":"complete"},indent=2),encoding="utf-8")
    return summary
