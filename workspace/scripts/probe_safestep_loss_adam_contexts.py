"""Exploratory common-context one-step probes, not new policy training runs.

All 30 predeclared baseline contexts are used. Every branch starts from the
same baseline policy, target, batch AND baseline Adam moment history. This is
not the Adam history of a model trained from scratch with another loss.
"""
from datetime import datetime, timezone
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fire_evac.safestep_author_numerics import AuthorDQN
from fire_evac.safestep_author_training import tensor_fingerprint
from fire_evac.safestep_loss_ablation import variant_loss


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(1)
    cfg = read(ROOT / "schemas/safestep_loss_ablation_20261015.json")
    out = ROOT / cfg["output_directory"] / "common_context_adam_probe_v2.json"
    assert not out.exists()
    reuse = read(out.parent / "baseline_reuse_before_training.json")
    rows, hashes = [], {}
    for ref in reuse["references"]:
        directory = ROOT / ref["source_directory"]
        cache = {}
        paths = sorted((directory / "update_contexts").glob("*.pt"))
        assert len(paths) == len(cfg["audit_update_indices"])
        for path in paths:
            assert digest(path) == ref["files"][str(path.relative_to(ROOT))]
            hashes[str(path.relative_to(ROOT))] = digest(path)
            context = torch.load(path, weights_only=True, map_location="cpu")
            def moment_fingerprint(state):
                tensors = {f"{i}:{key}": value for i, entry in state["state"].items()
                           for key, value in entry.items() if isinstance(value, torch.Tensor)}
                return tensor_fingerprint(tensors)
            initial_moments = moment_fingerprint(context["optimizer_before"])
            samples = []
            for idx in context["ids"]:
                chunk = idx // cfg["transition_chunk"]
                if chunk not in cache:
                    with np.load(directory / "transitions" / f"chunk_{chunk:05d}.npz") as z:
                        cache[chunk] = z["records"]
                samples.append(cache[chunk][idx % cfg["transition_chunk"]])
            batch = np.asarray(samples)
            args = (torch.eye(61), torch.tensor(batch["state"].astype(int)),
                    torch.tensor(batch["action"].astype(int)), torch.tensor(batch["next_state"].astype(int)),
                    torch.from_numpy(batch["reward"].astype(np.float32)), torch.tensor(batch["status"] == 0), cfg["gamma"])
            branches, gradients = {}, {}
            for arm in cfg["arms"]:
                policy, target = AuthorDQN(61), AuthorDQN(61)
                policy.load_state_dict(context["policy_before"])
                target.load_state_dict(context["target"])
                optimizer = torch.optim.Adam(policy.parameters(), lr=cfg["learning_rate"])
                optimizer.load_state_dict(deepcopy(context["optimizer_before"]))
                assert moment_fingerprint(optimizer.state_dict()) == initial_moments
                before = torch.cat([p.detach().flatten().clone() for p in policy.parameters()])
                loss, q, labels, _ = variant_loss(arm, policy, target, *args)
                assert torch.equal(q.detach(), context["q"]) and torch.equal(labels, context["labels"])
                optimizer.zero_grad()
                loss.backward()
                gradient = torch.cat([p.grad.flatten().clone() for p in policy.parameters()])
                gradients[arm] = gradient
                optimizer.step()
                after = torch.cat([p.detach().flatten() for p in policy.parameters()])
                branches[arm] = dict(loss=float(loss.detach()), gradient_L2=float(gradient.norm()),
                                     parameter_step_L2=float((after - before).norm()))
                if arm == "full_anchor":
                    assert loss.item() == context["loss"]
                    assert tensor_fingerprint(policy.state_dict()) == tensor_fingerprint(context["policy_after"])
            torch.testing.assert_close(gradients["selected"], 4 * gradients["selected_quarter"], rtol=1e-6, atol=1e-6)
            assert moment_fingerprint(context["optimizer_before"]) == initial_moments
            denominator = branches["selected_quarter"]["parameter_step_L2"]
            rows.append(dict(seed=ref["seed"], update=context["update"], global_step=context["global_step"],
                branches=branches, selected_to_quarter_parameter_step_ratio=branches["selected"]["parameter_step_L2"] / denominator))
    ratios = [r["selected_to_quarter_parameter_step_ratio"] for r in rows]
    result = dict(status="COMMON_CONTEXT_ONE_STEP_PROBE_COMPLETE", created_at=datetime.now(timezone.utc).isoformat(),
        version="2-independent-optimizer-moment-copies",
        v1_probe_superseded="The exploratory first probe reused optimizer moment tensors in memory; its cross-branch step ratios are invalid. Formal training and verifier do not share optimizer histories across branches and are unaffected",
        exploratory_supplement_not_preregistered_primary_outcome=True,
        source="All30 saved baseline contexts; same baseline Adam moment history in each branch",
        one_step_probe_branches=len(rows) * 3, exactly_reproduced_baseline_updates=len(rows),
        parameter_step_ratio_min=float(np.min(ratios)), parameter_step_ratio_median=float(np.median(ratios)),
        parameter_step_ratio_max=float(np.max(ratios)), results=rows,
        caveat="One-step interventions from baseline Adam moments do not reproduce the from-scratch moments or full trajectories of the alternative losses",
        formal_training_runs_added=0, source_sha256=hashes, script_sha256=digest(Path(__file__)))
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ("results", "source_sha256")}, indent=2))


if __name__ == "__main__":
    main()
