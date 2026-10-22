"""Paired update-frequency diagnostic retaining the public algorithm's loss.

Both arms share all implementation choices. Only update scheduling differs.
"""
from collections import deque
from copy import deepcopy
import csv
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from .safestep_author_numerics import AuthorDQN, author_rollout, load_author_inputs, import_reviewed_source
from .safestep_author_training import TRANSITION_DTYPE, tensor_fingerprint
from .safestep_paper_stepwise import GreedyPolicy


def schedule(arm, global_step, episode_finished, replay_length, last_sync, batch_size=64):
    ready = replay_length > batch_size
    update = ready and (arm == "every_step" or episode_finished)
    decay = ready and episode_finished
    sync = decay and global_step - last_sync > 50
    return update, decay, sync


def public_vector_loss(policy, target, identity, states, actions, next_states, rewards, wins, gamma):
    current = policy(identity[states])
    with torch.no_grad():
        labels = rewards.clone()
        active = ~wins
        if active.any():
            labels[active] += gamma * target(identity[next_states[active]]).max(1).values
        anchored = target(identity[states]).clone()
        anchored[torch.arange(len(states)), actions] = labels
    loss = torch.mean((current - anchored) ** 2)
    chosen = current.gather(1, actions[:, None]).squeeze(1)
    selected_sse = torch.sum((chosen - labels) ** 2)
    anchor_sse = torch.sum((current - anchored) ** 2) - selected_sse
    return loss, chosen, labels, anchor_sse


class Journals:
    def __init__(self, out, cfg):
        self.out = Path(out)
        self.out.mkdir(exist_ok=False)
        for name in ("transitions", "optimizer", "update_contexts"):
            (self.out / name).mkdir()
        self.tbuf = np.empty(cfg["transition_chunk"], dtype=TRANSITION_DTYPE)
        n, b = cfg["optimizer_chunk"], cfg["batch_size"]
        self.obuf = {"ids": np.empty((n, b), dtype=np.uint32),
                     "q": np.empty((n, b), dtype=np.float32),
                     "targets": np.empty((n, b), dtype=np.float32),
                     "anchor_sse": np.empty(n, dtype=np.float64),
                     "loss": np.empty(n, dtype=np.float64),
                     "global_step": np.empty(n, dtype=np.uint32),
                     "episode": np.empty(n, dtype=np.uint32),
                     "target_last_sync_step": np.empty(n, dtype=np.uint32)}
        self.tused = self.oused = self.tc = self.oc = self.transitions = self.updates = 0
        self.handle = (self.out / "episodes.csv").open("w", encoding="utf-8", newline="")
        self.writer = csv.DictWriter(self.handle, fieldnames=["episode", "reset_seed", "start_x", "start_y",
            "steps", "episode_status", "environment_status", "FED", "reward", "global_steps", "updates",
            "epsilon_before", "epsilon_after", "elapsed_s"])
        self.writer.writeheader()

    def transition(self, row):
        self.tbuf[self.tused] = row
        self.tused += 1
        self.transitions += 1
        if self.tused == len(self.tbuf):
            self.flush_transitions()

    def update(self, ids, q, labels, anchor, loss, step, ep, sync):
        i = self.oused
        self.obuf["ids"][i] = ids
        self.obuf["q"][i] = q.detach().numpy()
        self.obuf["targets"][i] = labels.detach().numpy()
        for key, value in (("anchor_sse", anchor), ("loss", loss), ("global_step", step),
                           ("episode", ep), ("target_last_sync_step", sync)):
            self.obuf[key][i] = value
        self.oused += 1
        self.updates += 1
        if self.oused == len(self.obuf["loss"]):
            self.flush_optimizer()

    def flush_transitions(self):
        if self.tused:
            np.savez_compressed(self.out / "transitions" / f"chunk_{self.tc:05d}.npz", records=self.tbuf[:self.tused])
            self.tc += 1
            self.tused = 0

    def flush_optimizer(self):
        if self.oused:
            np.savez_compressed(self.out / "optimizer" / f"chunk_{self.oc:05d}.npz",
                                **{k: v[:self.oused] for k, v in self.obuf.items()})
            self.oc += 1
            self.oused = 0

    def close(self):
        self.flush_transitions()
        self.flush_optimizer()
        self.handle.flush()
        self.handle.close()


def run(source, cfg, arm, seed, out, budget_override=None):
    started = time.monotonic()
    assert arm in cfg["arms"]
    torch.set_num_threads(cfg["torch_threads"])
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    gate_rng = np.random.RandomState(seed)
    action_rng = random.Random(200000 + seed)
    replay_rng = random.Random(300000 + seed)
    reset_rng = np.random.default_rng(100000 + seed)
    inputs = load_author_inputs(source)
    maze = import_reviewed_source(Path(source) / "maze.py", f"frequency_maze_{arm}_{seed}")
    maze.Maze.reward_exit = cfg["reward_exit"]
    env = maze.Maze(inputs["grid"].copy(), inputs["streams"], inputs["patterns"])
    coordinates = inputs["empty"] + list(inputs["exits"])
    index = {xy: i for i, xy in enumerate(coordinates)}
    identity = torch.eye(61)
    policy, target = AuthorDQN(61), AuthorDQN(61)
    target.load_state_dict(policy.state_dict())
    initial = tensor_fingerprint(policy.state_dict())
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg["learning_rate"])
    journal = Journals(out, cfg)
    ring = np.empty(cfg["replay_capacity"], dtype=TRANSITION_DTYPE)
    memory = deque(maxlen=cfg["replay_capacity"])
    epsilon = cfg["epsilon_start"]
    last_sync = completed = ep = 0
    syncs = [0]
    budget = budget_override or cfg["control_step_budget"]
    audited = []
    finished = True
    for global_step in range(1, budget + 1):
        if finished:
            ep += 1
            reset_seed = int(reset_rng.integers(0, 2**32, dtype=np.uint64))
            obs = env.reset(seed=reset_seed)
            start = tuple(map(int, obs))
            state = index[start]
            step = 0
            total_reward = 0.
            epsilon_before = epsilon
        step += 1
        if gate_rng.random_sample() < epsilon:
            action = int(action_rng.choice(maze.Maze.actions))
        else:
            with torch.no_grad():
                action = int(policy(identity[state]).argmax())
        obs, reward, status, _, fed = env.step(maze.Action(action), step)
        next_state = index[tuple(map(int, obs))]
        finished = status != maze.Status.PLAYING
        row = (ep, step, state, action, next_state, reward, int(status.value), float(fed))
        global_id = global_step - 1
        ring[global_id % cfg["replay_capacity"]] = row
        memory.append(global_id)
        journal.transition(row)
        total_reward += reward
        do_update, decay, do_sync = schedule(arm, global_step, finished, len(memory), last_sync, cfg["batch_size"])
        if do_update:
            ids = np.asarray(replay_rng.sample(memory, cfg["batch_size"]), dtype=np.uint32)
            batch = ring[ids % cfg["replay_capacity"]]
            states = torch.from_numpy(batch["state"].astype(np.int64))
            actions = torch.from_numpy(batch["action"].astype(np.int64))
            next_states = torch.from_numpy(batch["next_state"].astype(np.int64))
            rewards = torch.from_numpy(batch["reward"].astype(np.float32))
            wins = torch.from_numpy(batch["status"] == maze.Status.WIN.value)
            loss, q, labels, anchor = public_vector_loss(policy, target, identity, states, actions,
                                                        next_states, rewards, wins, cfg["gamma"])
            update_number = journal.updates + 1
            context = None
            if update_number in cfg["audit_update_indices"]:
                context = dict(update=update_number, global_step=global_step, ids=ids.tolist(),
                    policy_before=deepcopy(policy.state_dict()), target=deepcopy(target.state_dict()),
                    optimizer_before=deepcopy(optimizer.state_dict()), loss=float(loss.detach()),
                    q=q.detach().clone(), labels=labels.detach().clone())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            journal.update(ids, q, labels, float(anchor.detach()), float(loss.detach()), global_step, ep, last_sync)
            if context is not None:
                context["policy_after"] = deepcopy(policy.state_dict())
                torch.save(context, journal.out / "update_contexts" / f"update_{update_number:07d}.pt")
                audited.append(update_number)
        if decay:
            epsilon *= cfg["epsilon_decay_per_completed_episode"]
        if do_sync:
            target.load_state_dict(policy.state_dict())
            last_sync = global_step
            syncs.append(global_step)
        state = next_state
        if finished or global_step == budget:
            completed += int(finished)
            journal.writer.writerow(dict(episode=ep, reset_seed=reset_seed, start_x=start[0], start_y=start[1],
                steps=step, episode_status=status.name if finished else "BUDGET_CUT", environment_status=status.name,
                FED=float(fed), reward=total_reward, global_steps=global_step, updates=journal.updates,
                epsilon_before=epsilon_before, epsilon_after=epsilon, elapsed_s=time.monotonic() - started))
        if global_step in cfg["checkpoints_at_steps"]:
            torch.save(policy.state_dict(), journal.out / f"checkpoint_step{global_step:06d}.pth")
        if global_step % cfg["progress_every_steps"] == 0:
            journal.handle.flush()
            (journal.out / "progress.json").write_text(json.dumps(dict(status="training", arm=arm, seed=seed,
                global_steps=global_step, completed_episodes=completed, updates=journal.updates,
                elapsed_s=time.monotonic() - started), indent=2), encoding="utf-8")
    journal.close()
    torch.save(policy.state_dict(), journal.out / "budget_final_model.pth")
    training = dict(status="fixed_control_budget_completed", arm=arm, seed=seed, control_steps=budget,
        episodes_started=ep, completed_episodes=completed, last_episode_budget_cut=not finished,
        optimizer_updates=journal.updates, epsilon=epsilon, initial_policy_fingerprint=initial,
        final_policy_fingerprint=tensor_fingerprint(policy.state_dict()), wall_seconds=time.monotonic() - started,
        audited_update_indices=audited, coordinate_order=coordinates,
        paper12000_episode_training_completed=False)
    (journal.out / "target_syncs.json").write_text(json.dumps(syncs), encoding="utf-8")
    (journal.out / "budget_training_complete.json").write_text(json.dumps(training, indent=2), encoding="utf-8")
    # Save budget-final model and completion evidence before computing policy scores.
    policy.eval()
    greedy = GreedyPolicy(policy, inputs["empty"], maze.Action)
    traces = []
    for start in inputs["empty"]:
        t = author_rollout(maze, inputs, greedy, start)
        t["start"] = start
        traces.append(t)
    reference = json.loads((Path(source).parent / "numeric_run/results.json").read_text(encoding="utf-8"))
    paper = []
    for expected in reference["stored_weight_runs"]:
        t = next(t for t in traces if t["start"] == tuple(expected["path"][0]))
        paper.append(dict(start=t["start"], status=t["status"], steps=t["steps"], FED=t["FED"],
            path_match=t["path"] == list(map(tuple, expected["path"]))))
    summary = dict(training=training, success=sum(t["status"] == "WIN" for t in traces), starts=61,
        invalid_steps=sum(t["invalid_steps"] for t in traces), revisits=sum(t["revisits"] for t in traces),
        paper_starts=paper, both_paper_routes_match=all(p["path_match"] for p in paper))
    (journal.out / "evaluation_traces.json").write_text(json.dumps(traces, indent=2), encoding="utf-8")
    (journal.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (journal.out / "progress.json").write_text(json.dumps(training | dict(status="complete"), indent=2), encoding="utf-8")
    return summary
