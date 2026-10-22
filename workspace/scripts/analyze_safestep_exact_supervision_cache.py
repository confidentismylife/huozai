"""Describe value scale and ranking ties after the frozen evaluation; no training."""
import hashlib
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import numpy as np
import torch
from fire_evac.safestep_shared_observation import build_network

OUT = ROOT/'outputs/safestep_exact_supervision_20261015'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    cfg = json.loads((ROOT/'schemas/safestep_exact_supervision_20261015.json').read_text(encoding='utf-8'))
    diagnostics = {}
    torch.set_num_threads(1)
    with np.load(OUT/'training_dataset.npz') as stored:
        obs, truth, legal = stored['obs'], stored['targets'], stored['legal']
    near = obs[:, 48] > 96/128
    diagnostics['training_scope'] = {'states': len(obs), 'ordinary_states': int((~near).sum()),
        'near_deadline_states': int(near.sum()),
        'mean_absolute_legal_target_ordinary': float(np.abs(truth[~near][legal[~near]]).mean()),
        'mean_absolute_legal_target_near_deadline': float(np.abs(truth[near][legal[near]]).mean())}
    diagnostics['models'] = {}
    for seed in cfg['training']['seeds']:
        model = build_network(cfg['training']['hidden_layers'])
        model.load_state_dict(torch.load(OUT/'models'/f'seed{seed}/policy.pt', weights_only=True, map_location='cpu'))
        model.eval()
        with torch.inference_mode():
            q = np.vstack([model(torch.from_numpy(obs[i:i+2048])).numpy() for i in range(0, len(obs), 2048)])
        squared = (q-truth)**2
        diagnostics['models'][str(seed)] = {
            'training_ordinary_legal_Q_MAE': float(np.abs(q[~near]-truth[~near])[legal[~near]].mean()),
            'training_near_deadline_legal_Q_MAE': float(np.abs(q[near]-truth[near])[legal[near]].mean()),
            'near_deadline_share_of_training_squared_error': float(squared[near][legal[near]].sum()/squared[legal].sum())}
    with np.load(OUT/'validation_state_grid.npz') as data:
        truth, legal, obs = data['targets'], data['legal'], data['obs']
    best = np.where(legal, truth, -np.inf).max(axis=1)
    best_count = ((best[:, None]-truth <= cfg['ranking_tolerance']) & legal).sum(axis=1)
    unique_choice = (legal.sum(axis=1) > 1) & (best_count == 1)
    sorted_truth = np.sort(np.where(legal, truth, -np.inf), axis=1)
    gap = sorted_truth[:, -1]-sorted_truth[:, -2]
    diagnostics['new_state_grid'] = {'states': len(obs),
        'single_legal_action_states': int((legal.sum(axis=1) == 1).sum()),
        'multiple_optimal_action_states': int((best_count > 1).sum()),
        'states_with_unique_optimum_and_multiple_legal_actions': int(unique_choice.sum()),
        'median_best_to_second_best_gap_on_unique_choice_states': float(np.median(gap[unique_choice])),
        'caution': 'Frozen overall accuracy includes forced actions and true ties. These supplementary counts do not replace the frozen acceptance rules.'}
    with np.load(OUT/'validation_Q_predictions.npz') as values:
        diagnostics['new_unique_choice_accuracy'] = {}
        for method in values.files:
            q = values[method]
            action = np.where(legal, q, -np.inf).argmax(axis=1)
            correct = best-truth[np.arange(len(obs)), action] <= cfg['ranking_tolerance']
            diagnostics['new_unique_choice_accuracy'][method] = float(correct[unique_choice].mean())
    diagnostics.update({'status': 'posthoc_cached_scale_and_tie_description', 'new_training': 0, 'new_rollouts': 0,
        'not_a_causal_loss_weighting_experiment': True, 'analysis_script_sha256': sha(__file__),
        'source_summary_sha256': sha(OUT/'summary.json')})
    payload = json.dumps(diagnostics, ensure_ascii=False, indent=2)+'\n'
    path = OUT/'cached_scale_and_ranking_diagnosis.json'
    if path.exists():
        assert path.read_text(encoding='utf-8') == payload
    else:
        path.write_text(payload, encoding='utf-8')
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
