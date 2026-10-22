"""Bounded direction hint without injecting advice into PPO's release features.

Experimental adapter only. It does not imply safety qualification or a gain
over the unchanged PPO. All actions still obey the shared frozen mask.
"""
import numpy as np


def direction_hint_action(model, no_advice_observation, mask, proposal=None, decision=None, strength=.25):
    import torch
    with torch.no_grad():
        tensor, _ = model.policy.obs_to_tensor(no_advice_observation)
        probs = model.policy.get_distribution(tensor).distribution.probs.cpu().numpy()[0]
    scores = np.log(np.maximum(probs, 1e-30))
    raw = int(np.argmax(scores))
    scores = np.where(mask, scores, -np.inf)
    base = int(np.argmax(scores))
    if proposal is None or decision is None or not decision.accepted:
        return base, raw
    target = proposal.preferred_branch
    if base != 8 and base // 4 == target:
        return base, raw
    # Uniform within the target direction: preserve the PPO release ranking.
    for a in range(target * 4, target * 4 + 4):
        if mask[a]:
            scores[a] += strength * proposal.confidence
    return int(np.argmax(scores)), raw
