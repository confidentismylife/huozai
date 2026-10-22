"""Additional independent reconstruction of frozen predictions, never fits."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/evacuation_auto_campaign_20261005'


def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def kernel_prediction(x,m):
    zz=(x-m['mean'])/m['std'];d=np.sum((m['train']-zz)**2,axis=1);k=np.exp(-float(m['gamma'])*d)
    return (k-np.dot(k,m['weights'])-m['row']+float(m['grand']))@m['coefficient']+m['intercept']
def physical(x):
    g,on,bg=x[21],x[22],x[5];cn,cs=x[23],x[24]
    north=240/max(8*cn,1)/120;south=240/max(8*cs,1)/120;both=240/max(8*(cn+cs),1)/120
    return np.array([g,on,bg,x[14],x[15],cn,cs,x[18],x[25],x[26],x[27],x[19],x[20],
        north,south,both,g*north,g*south,bg*both,max(0,120*both-20*on)/120*g,cn*cs,cn-cs])


def main():
    frozen=OUT/'models/frozen_models.json'
    if not frozen.exists():print('Models not ready; no prediction check performed.');return
    record=json.loads(frozen.read_text(encoding='utf-8'));direct=load(OUT/'models/direct.npz')
    absolute=[load(OUT/'models'/f'absolute_{seed}.npz') for seed in record['bootstrap_seeds']];mean=load(OUT/'models/mean.npz')['gains'];checked=0
    for block in ('calibration','test'):
        for p in sorted((OUT/(block+'_choices')).glob('*.json')):
            ch=json.loads(p.read_text(encoding='utf-8'));x=load(OUT/(block+'_planning')/(p.stem+'.npz'))['observations'][0].astype(float);x[28:33]=0
            predictions={'direct':np.r_[0,kernel_prediction(x,direct)],'mean':mean,'zero':np.zeros(5)}
            ll=np.array([kernel_prediction(physical(x),m) for m in absolute]);loss=np.clip(ll.mean(axis=0).reshape(2,5),0,1)
            predictions['absolute']=(loss[:,0,None]-loss).min(axis=0)
            for i in range(3):
                loss=np.clip(ll[i].reshape(2,5),0,1);predictions[f'absolute_seed{i+1}']=(loss[:,0,None]-loss).min(axis=0)
            for name,v in predictions.items():np.testing.assert_allclose(v,ch['predictions'][name],atol=1e-10,rtol=0)
            checked+=1
    result={'all_checks_passed':True,'frozen_predictions_independently_reconstructed_families':checked,'expected_final_families':72,
        'complete_check':checked==72,'models_record_sha256':hashlib.sha256(frozen.read_bytes()).hexdigest(),
        'verifier_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'new_model_fits':0,'new_simulations':0}
    target=OUT/f'additional_prediction_verification_{checked:03d}.json'
    if not target.exists():target.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
