"""Small numerical models and instrumented frozen-policy continuations."""
import copy
import numpy as np

OUTCOME_FIELDS = ('final_incapacitation','final_fed','final_evacuation','final_remaining','additional_congestion',
                  'steps','mean_release','wait_requests','hold_block_rate','first_requested_action','first_actual_joint_action')
TRACE_FIELDS = ('elapsed','requested_joint_action','actual_joint_action','actual_sign_action','actual_release_fraction','hold_blocked') + tuple(
    f'{prefix}_{zone}' for prefix in ('observed_smoke_pre','physical_smoke_pre','occupancy_pre','occupancy_post','active_mean_fed_pre','active_mean_fed_post')
    for zone in ('room_a','room_b','junction','north','south')) + ('released_people','new_incapacitated','new_evacuated','fed_increment_sum')


def labels(risks):
    """Population x world x policy x future x metric, pooling before world scores."""
    a=np.asarray(risks,float)
    if a.ndim!=5 or a.shape[1]!=2 or a.shape[2]<2 or a.shape[-1]!=11 or not np.isfinite(a).all():raise ValueError('invalid tensor')
    l=a[...,0].mean(axis=(0,3)); r=l[:,0].max()-l.max(axis=0);m=(l[:,0,None]-l).min(axis=0)
    r[0]=m[0]=0.;return l,np.stack((r,m))


def physics_features(raw):
    a=np.asarray(raw,float)
    if a.shape!=(49,) or not np.isfinite(a).all():raise ValueError('finite raw49 required')
    g=a[21];on=a[22];bg=a[5];cn,cs=a[23:25];ln,ls=a[14:16]
    clearance_n=240/max(8*cn,1)/120;clearance_s=240/max(8*cs,1)/120
    clearance_total=240/max(8*(cn+cs),1)/120
    # Semantic fields and fixed interactions; no true source, type or latent state.
    return np.r_[g,on,bg,ln,ls,cn,cs,a[18],a[25:28],a[19:21],
                 clearance_n,clearance_s,clearance_total,
                 g*clearance_n,g*clearance_s,bg*clearance_total,
                 max(0,120*clearance_total-20*on)/120*g,cn*cs,cn-cs]


def fit_kernel(features,targets,seed=None):
    x,y=np.asarray(features,float),np.asarray(targets,float);n=len(x)
    if x.ndim!=2 or y.ndim!=2 or len(y)!=n:raise ValueError('matrix rows required')
    weights=np.full(n,1/n)
    if seed is not None:weights=np.bincount(np.random.default_rng(seed).integers(n,size=n),minlength=n)/n
    mean=weights@x;std=np.sqrt(weights@((x-mean)**2));std=np.where(std<.001,1.,std)
    z=(x-mean)/std;intercept=weights@y;active=max(1,int((np.var(z,axis=0)>1e-12).sum()));gamma=1/(2*active)
    distance=np.maximum(0,(z*z).sum(axis=1)[:,None]+(z*z).sum(axis=1)[None,:]-2*z@z.T)
    k=np.exp(-gamma*distance);row=k@weights;grand=weights@row;centered=k-row[:,None]-row[None,:]+grand;root=np.sqrt(weights)
    matrix=root[:,None]*centered*root[None,:]+.01*np.eye(n);rhs=root[:,None]*(y-intercept)
    dual=np.linalg.solve(matrix,rhs);coefficient=root[:,None]*dual
    return dict(mean=mean,std=std,train=z,weights=weights,row=row,grand=np.array(grand),gamma=np.array(gamma),
                coefficient=coefficient,intercept=intercept,seed=np.array(-1 if seed is None else seed),
                equation_residual=np.array(np.abs(matrix@dual-rhs).max()))


def predict_kernel(raw,model):
    x=(np.atleast_2d(np.asarray(raw,float))-model['mean'])/model['std'];z=model['train']
    d=np.maximum(0,(x*x).sum(axis=1)[:,None]+(z*z).sum(axis=1)[None,:]-2*x@z.T);k=np.exp(-float(model['gamma'])*d)
    centered=k-(k@model['weights'])[:,None]-model['row'][None,:]+float(model['grand'])
    return centered@model['coefficient']+model['intercept']


def gains(raw,modelset):
    f=np.asarray(raw,float).copy();f[28:33]=0
    direct=np.r_[0.,predict_kernel(f,modelset['direct'])[0]]
    individual=[predict_kernel(physics_features(raw),model)[0] for model in modelset['absolute']]
    loss=np.clip(np.mean(individual,axis=0).reshape(2,5),0,1)
    absolute=(loss[:,0,None]-loss).min(axis=0);absolute[0]=0
    result={'direct':direct,'absolute':absolute,'mean':modelset['mean'].copy(),'zero':np.zeros(5)}
    for i,pred in enumerate(individual):
        l=np.clip(pred.reshape(2,5),0,1);y=(l[:,0,None]-l).min(axis=0);y[0]=0.;result[f'absolute_seed{i+1}']=y
    return result


def choose(prediction,margin=0):
    a=np.asarray(prediction,float);j=int(a.argmax())
    return j if j!=0 and a[j]>max(.001,margin) else 0


def calibrated_margin(predictions,labels_,prior):
    p,y=np.asarray(predictions,float),np.asarray(labels_,float)
    residual=(p[:,1:]-y[:,1:]).max(axis=1);q=float(np.quantile(residual,.95,method='higher'))
    return max(.001,prior,q),residual


def mean_active_fed(domain):
    return np.array([np.mean([domain.people[i].fed for i in ids]) if ids else 0. for ids in domain.zones])


def continuation(env,observation,actor,future_seed,full_trace=False):
    """Same execution as original evaluate_continuation, optionally richer observation."""
    import gymnasium as gym
    x=np.asarray(observation).copy();clone=copy.deepcopy(env);traces=[]
    try:
        domain=clone.unwrapped.domain_env;domain.rng=np.random.default_rng(future_seed);layer=clone
        while isinstance(layer,gym.Wrapper):
            if 'rng' in vars(layer):layer.rng=np.random.default_rng(future_seed+9_000_001)
            layer=layer.env
        start=domain._info();elapsed=waits=blocks=0;release=0.;done=False;first=actualfirst=-1
        while not done:
            action=actor.action(x)
            if full_trace:
                sensed=x[5:10].copy();hazard=domain._hazard().copy();before=np.array([len(ids) for ids in domain.zones]);fed_before=mean_active_fed(domain)
                inc_before=domain.incapacitated;evac_before=domain.evacuated;fed_sum_before=domain.cumulative_fed
            x,_,term,trunc,info=clone.step(action);done=term or trunc
            if elapsed==0:first=action;actualfirst=int(info['actual_joint_action'])
            release+=float(info['actual_release_fraction']);waits+=action==8;blocks+=bool(info['action_blocked_by_sign_hold'])
            if full_trace:
                after=np.array([len(ids) for ids in domain.zones]);fed_after=mean_active_fed(domain)
                traces.append(np.r_[elapsed,action,info['actual_joint_action'],info['actual_sign_action'],info['actual_release_fraction'],int(info['action_blocked_by_sign_hold']),
                    sensed,hazard,before,after,fed_before,fed_after,domain.last_room_release_count,
                    domain.incapacitated-inc_before,domain.evacuated-evac_before,domain.cumulative_fed-fed_sum_before])
            elapsed+=1
            if elapsed>domain.config.max_steps:raise RuntimeError('episode exceeded boundary')
        result=np.array([info['incapacitation_rate'],info['mean_cumulative_fed'],info['evacuation_rate'],info['remaining']/domain.config.population,
            info['cumulative_congestion']-start['cumulative_congestion'],elapsed,release/elapsed,waits,blocks/elapsed,first,actualfirst],float)
        return result,np.asarray(traces,float).reshape(-1,len(TRACE_FIELDS))
    finally:clone.close()


class AdviceActor:
    """Frozen34 actor preserving its advice inputs for the paired LLM ablation."""
    def __init__(self,model):self.model=model
    def action(self,observation):
        action,_=self.model.predict(np.asarray(observation,dtype=np.float32)[:34],deterministic=True)
        return int(np.asarray(action).item())
