"""Check repaired artifact consistency; this is not a scientific peer review."""
from pathlib import Path
import hashlib
import json
import re
import pandas as pd
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/llm_ppo_paper_v2_20261021'
data=OUT/'data'
checks={}
validation=json.loads((OUT/'validation.json').read_text(encoding='utf-8'))
for relative,digest in validation['source_sha256'].items():
    assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==digest, relative
checks['source_hashes_still_match']=True

links=[]
for path in OUT.glob('*.md'):
    content=path.read_text(encoding='utf-8')
    for url in re.findall(r'\]\(([^)]+)\)',content):
        if url.startswith(('https://','http://','#')): continue
        assert (path.parent/url.split('#')[0]).is_file(), (path.name,url)
        links.append([path.name,url])
checks['local_markdown_links']=len(links)

fixed=pd.read_csv(data/'replay_endpoints_corrected.csv')
original=pd.read_csv(ROOT/'outputs/final_mechanism_attribution_20261020/replay_diagnostics.csv')
assert fixed.run_id.equals(original.run_id)
for column in original:
    if column not in ('evacuated_50_seconds','evacuated_90_seconds'):
        pd.testing.assert_series_equal(fixed[column],original[column],check_exact=False,rtol=1e-14,atol=1e-14)
assert fixed.evacuated_90_seconds.isna().sum()==309
assert fixed.evacuated_50_seconds.isna().sum()==2
checks['non_time_fields_preserved']=True

summary=pd.read_csv(data/'endpoint_attainment_summary.csv')
p=summary[(summary.scope=='calibrated_logit/none')&(summary.threshold_percent==90)]
assert p.set_index('method').reached.to_dict()=={'trained_llm_advice_ppo':221,'trained_pure_ppo':220,'trained_rule_advice_ppo':211}
checks['primary_attainment_counts_match']=True

oldtable=pd.read_csv(ROOT/'artifacts/llm_ppo_paper_package_20261020/论文主表_统一结果.csv')
table=pd.read_csv(data/'论文主表_修订.csv')
assert len(table)==len(oldtable)-2
assert not table['指标'].isin(['50%撤离时间','90%撤离时间']).any()
assert table['来源文件'].notna().all()
original_kept=oldtable[~oldtable['指标'].isin(['50%撤离时间','90%撤离时间'])]
for column in ['效应值','95%CI下限','95%CI上限']:
    assert np.allclose(table[column],original_kept[column],rtol=1e-12,atol=1e-14)
checks['unaffected_effect_estimates_preserved']=True

oof=pd.read_csv(data/'prediction_without_identity_oof.csv')
assert len(oof)==768
assert not oof.duplicated(['scenario_id','behavior_model','noise_level','baseline','feature_set']).any()
assert (oof.groupby(['baseline','feature_set','scenario_id']).fold.nunique()==1).all()
assert oof.predicted_probability.between(0,1).all()
assert pd.read_csv(data/'prediction_without_identity_metrics.csv').meets_original_reference_gate.sum()==0
checks['grouped_oof_predictions_complete']=True
for image in (OUT/'figures').glob('*.png'): assert image.stat().st_size>10000
assert len(list((OUT/'figures').glob('*.png')))==5
assert len(list((OUT/'figures').glob('*.pdf')))==5
checks['figure_pairs']=5
payload=dict(status='passed_scoped_checks',checks=checks,
    limits='No external citation verification, scientific validity certification, or independent replication.',
    checked_document_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob('*.md')})
(OUT/'document_validation.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(checks,ensure_ascii=False))
