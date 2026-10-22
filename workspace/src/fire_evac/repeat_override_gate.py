"""A bounded public-state consensus gate, fitted only to archived evidence.

No scene, population, checkpoint, behavior identifier or hidden simulator field
is read by the online signature. Unsupported or conflicting cells abstain.
This is a development lookup experiment, not a calibrated safety predictor.
"""
import json
import math

SIGNATURE_VERSION = 'public_counts5_smoke005_history5_v1'


def public_signature(current, previous, base, candidate, offers_before, overrides_before):
    counts = [round(float(v) * 240) for v in current['occupancy_fraction']]
    smoke = [float(v) for v in current['smoke_increment_per_step']]
    capacities = list(map(int, current['exit_capacity_persons_per_step']))
    assert len(counts) == len(smoke) == 5 and len(capacities) == 2
    assert all(math.isfinite(v) and v >= 0 for v in smoke)
    before = counts if previous is None else [round(float(v)*240) for v in previous['occupancy_fraction']]
    # Discretization is fixed in physical units before fresh confirmation.
    features = [int(base), int(candidate), min(int(offers_before), 1), int(overrides_before > 0),
                int(current['step']), int(current['actual_sign']), int(current['sign_hold_remaining_steps']),
                *capacities, int(current['forecast_source'] == 'south'),
                max(-20, min(20, int(current['forecast_onset_step'])-int(current['step']))),
                *[n//5 for n in counts], *[math.floor(v/.005) for v in smoke],
                *[round((n-p)/5) for n,p in zip(counts,before)]]
    return json.dumps(features, separators=(',', ':'))


def evidence_class(incap_delta, fed_delta):
    if incap_delta <= -.001:
        return 'beneficial'
    if incap_delta >= .001:
        return 'harmful'
    if fed_delta <= -.001:
        return 'beneficial'
    if fed_delta >= .001:
        return 'harmful'
    return 'small_or_neutral'


def fit_consensus(samples):
    """Offline fit only: every known future must support benefit without harm.

Repeated timing-mode copies of the same archived state are deduplicated. Any
increase in incapacitation, neutral evidence or disagreement forces abstention.
The output stores no test labels; no fitting occurs during confirmation.
"""
    cells, seen = {}, {}
    for sample in samples:
        identity = sample['offline_identity']
        signature, outcomes = sample['signature'], sample['paired_outcomes']
        assert outcomes
        duplicate_key = json.dumps([signature,outcomes],sort_keys=True)
        if identity in seen:
            assert seen[identity] == duplicate_key, 'inconsistent duplicate evidence'
            continue
        seen[identity] = duplicate_key
        cell = cells.setdefault(signature, dict(states=0,futures=0,classes={},all_supported=True))
        cell['states'] += 1
        for outcome in outcomes:
            incap, fed = float(outcome['incap_delta']), float(outcome['fed_delta'])
            assert math.isfinite(incap) and math.isfinite(fed)
            label = evidence_class(incap,fed)
            cell['classes'][label] = cell['classes'].get(label,0)+1
            cell['futures'] += 1
            cell['all_supported'] &= label == 'beneficial' and incap <= 1e-12
    return dict(version=SIGNATURE_VERSION,cells=cells,unique_archived_states=len(seen),
                acceptance='Every archived paired future in the public cell is beneficial and none increases incapacitation; unseen cells reject.')


def consensus_decision(fitted, signature):
    assert fitted['version'] == SIGNATURE_VERSION
    cell = fitted['cells'].get(signature)
    if cell is None:
        return False, 'unsupported_public_state'
    if not cell['all_supported']:
        return False, 'archived_evidence_not_consistently_beneficial'
    return True, 'archived_consensus_benefit'
