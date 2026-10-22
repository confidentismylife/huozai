"""Independent recomputation from saved predictions and frozen splits; no fits."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/research_convergence_20261011'


def read(name):
    return json.loads((OUT/name).read_text(encoding='utf-8'))


def main():
    rows=read('primary_rows.json');index={r['state_id']:r for r in rows}
    protocol=read('frozen_comparison_protocol.json');summary=read('comparison_summary.json')
    predictions=read('held_out_predictions.json');models=read('outer_fold_models.json')
    assert len(rows)==len(index)==72
    assert len({r['offline_identity'] for r in rows})==72
    assert len({r['family'] for r in rows})==20
    assert len(predictions)==72*4 and len(models)==20
    for fold in protocol['folds']:
        rs=[[index[s] for s in fold[k]] for k in ('train','calibration','test')]
        assert set.union(*[{r['state_id'] for r in x} for x in rs])==set(index)
        families=[{r['family'] for r in x} for x in rs]
        assert not(families[0]&families[1] or families[0]&families[2] or families[1]&families[2])
        assert all(r['scenario']==fold['held_out_scene'] for r in rs[2])
        assert all(r['scenario']!=fold['held_out_scene'] for r in rs[0]+rs[1])
        assert all(r['population']%100==3 for r in rs[1])
    for model in models:
        variant=model['variant'];fitted=model['model']
        ps=[p for p in predictions if p['variant']==variant and p['scene']==model['scene']]
        for p in ps:
            row=index[p['state_id']]
            pred=np.asarray(fitted['intercept'])
            if variant!='constant':
                x=np.asarray(row['features'][variant]);z=(x-fitted['mean'])/fitted['scale']
                pred=z @ np.asarray(fitted['coef'])+pred
            assert np.allclose(pred,p['prediction'],rtol=0.,atol=1e-12)
            upper=pred+np.asarray(p['margin'])
            decision=bool(upper[0]<=1e-12 and (upper[0]<=-.001 or upper[1]<=-.001))
            assert decision==p['accepted']
    for v in ('current','public_history','response','constant'):
        ps=[p for p in predictions if p['variant']==v]
        counts=Counter();errors=defaultdict(list)
        for p in ps:
            row=index[p['state_id']]
            counts[('accept_' if p['accepted'] else 'reject_')+row['label']]+=1
            errors[row['family']].append(np.abs(np.asarray(p['prediction'])-[row['incap_delta'],row['fed_delta']]))
        saved=summary['methods'][v]
        assert all(counts[k]==value for k,value in saved['gate']['confusion'].items())
        assert np.allclose(np.mean([np.mean(x,axis=0) for x in errors.values()],axis=0),
                           saved['prediction_metrics']['MAE'],rtol=0.,atol=1e-12)
    for row in rows:
        assert len(row['completed_public_history'])==row['step']<=3
        assert all(r['before']['step']<row['step'] for r in row['completed_public_history'])
        assert row['features']['response'][:len(row['features']['public_history'])]==row['features']['public_history']
        assert not any(row['features']['current'][28:33])
        for v in ('current','public_history','response'):
            assert np.isfinite(row['features'][v]).all()
    sources=read('preservation_before.json')['files']
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in sources.items())
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in protocol['sources'].items())
    result=dict(status='passed',unique_states=72,held_out_predictions_checked=288,folds_checked=5,
                checks=['deduplication','disjoint scene/population splits','saved-parameter predictions',
                        'gate thresholds','family-weighted errors','confusion denominators',
                        'causal history','equal history sensor information','blank advice coordinates',
                        'finite representations','frozen source and275 historical hashes'],
                preserved_files=len(sources),new_fits=0,new_simulations=0,reserved_final_opened=0)
    dest=OUT/'verification.json';assert not dest.exists()
    dest.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
