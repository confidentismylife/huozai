"""Fixed proposal-specific ridge models; no hidden scenario inputs."""
from dataclasses import dataclass
import numpy as np

ALPHA=.01


def inputs(observation,plan):
    x=np.asarray(observation,float).copy()
    if x.shape!=(49,) or not np.isfinite(x).all() or plan not in range(17):raise ValueError("finite observable49 and plan index required")
    x[28:33]=0
    return np.r_[x,np.eye(17)[plan]]


def family_weights(families):
    families=np.asarray(families);unique,counts=np.unique(families,return_counts=True)
    if not len(unique):raise ValueError("nonempty families required")
    lookup=dict(zip(unique,counts));return np.array([1/(len(unique)*lookup[f]) for f in families])


def targets(risks,supported):
    risks=np.asarray(risks,float);supported=np.asarray(supported)
    if risks.ndim!=4 or risks.shape[:2]!=(2,2) or min(risks.shape[2:])<1 or supported.shape!=(2,) or supported.dtype!=bool or not supported.any() or not np.isfinite(risks).all():
        raise ValueError("finite two-world two-policy risks required")
    means=risks[supported,:,:,0].mean(axis=2)
    robust=float(means[:,0].max()-means[:,1].max())
    minimum=float((means[:,0]-means[:,1]).min())
    return np.array([robust,minimum])


def kernel(x,z,gamma):
    distance=np.maximum(0,(x*x).sum(axis=-1)[:,None]+(z*z).sum(axis=-1)[None,:]-2*x@z.T)
    return np.exp(-gamma*distance)


@dataclass
class PairGainModel:
    kind: str
    mean: np.ndarray
    std: np.ndarray
    intercept: np.ndarray
    coefficients: np.ndarray
    train: np.ndarray
    weights: np.ndarray
    kernel_row_mean: np.ndarray
    kernel_grand_mean: np.ndarray
    gamma: np.ndarray

    def predict(self,x):
        x=np.asarray(x,float)
        if x.shape[-1]!=66 or not np.isfinite(x).all():raise ValueError("finite observable/proposal66 required")
        scalar=x.ndim==1;tx=np.atleast_2d((x-self.mean)/self.std)
        if self.kind=="linear":pred=tx@self.coefficients+self.intercept
        elif self.kind=="kernel":
            k=kernel(tx,self.train,float(self.gamma));center=k-(k@self.weights)[:,None]-self.kernel_row_mean[None,:]+float(self.kernel_grand_mean)
            pred=center@self.coefficients+self.intercept
        else:raise ValueError("unknown model")
        return pred[0] if scalar else pred

    def save(self,p):
        np.savez_compressed(p,**{k:np.asarray(v) for k,v in vars(self).items()})

    @classmethod
    def load(cls,p):
        with np.load(p,allow_pickle=False) as z:
            d={k:z[k].copy() for k in z.files};d["kind"]=str(d["kind"])
        return cls(**d)


def fit(x,y,families,kind,forbidden_families=()):
    x,y,families=np.asarray(x,float),np.asarray(y,float),np.asarray(families)
    if x.shape!=(len(y),66) or len(families)!=len(y) or not np.isfinite(x).all() or not np.isfinite(y).all() or set(families).intersection(forbidden_families):
        raise ValueError("aligned training rows only required")
    w=family_weights(families);mean=w@x;std=np.sqrt(w@((x-mean)**2));std=np.where(std<.001,1.,std)
    tx=(x-mean)/std;b=np.asarray(w@y)
    if kind=="linear":
        coefficient=np.linalg.solve(tx.T@(w[:,None]*tx)+ALPHA*np.eye(66),tx.T@(w*(y-b)))
        model=PairGainModel(kind,mean,std,b,coefficient,np.empty((0,66)),w,np.empty(0),np.asarray(0.),np.asarray(0.))
        gradient=tx.T@(w*(model.predict(x)-y))+ALPHA*coefficient
    elif kind=="kernel":
        active=max(1,int((np.var(tx,axis=0)>1e-12).sum()));gamma=np.asarray(1/(2*active));k=kernel(tx,tx,float(gamma))
        row=k@w;grand=np.asarray(w@row);center=k-row[:,None]-row[None,:]+grand
        root=np.sqrt(w);dual=np.linalg.solve(root[:,None]*center*root[None,:]+ALPHA*np.eye(len(y)),root*(y-b))
        coefficient=root*dual
        model=PairGainModel(kind,mean,std,b,coefficient,tx,w,row,grand,gamma)
        gradient=(root[:,None]*center*root[None,:]+ALPHA*np.eye(len(y)))@dual-root*(y-b)
    else:raise ValueError("kind must be linear or kernel")
    return model,{"states":len(y),"families":len(set(families)),"alpha":ALPHA,"kind":kind,
        "training_families":sorted(set(families.tolist())),"max_abs_normal_equation_residual":float(np.max(np.abs(gradient))),
        "gamma":float(model.gamma),"training_mae":float(w@np.abs(model.predict(x)-y))}


def calibrated_margin(prediction,empirical,families,prior):
    prediction,empirical,families=np.asarray(prediction,float),np.asarray(empirical,float),np.asarray(families)
    if prediction.shape!=empirical.shape or len(prediction)!=len(families) or not np.isfinite(prediction).all() or not np.isfinite(empirical).all() or not np.isfinite(prior) or prior<0:
        raise ValueError("aligned finite calibration inputs required")
    values={str(f):float(np.max((prediction-empirical)[families==f])) for f in sorted(set(families))}
    q=float(np.quantile(list(values.values()),.8,method="higher")) if values else None
    return max(prior,.001,q if q is not None else .001),values,q


def accepted(prediction,margin,worth_scores,seen=True,full=True):
    prob=np.asarray(worth_scores,float)
    if not np.isfinite(prediction) or not np.isfinite(margin) or margin<0 or prob.shape!=(3,) or np.any(prob<0) or not np.isclose(prob.sum(),1):
        raise ValueError("finite prediction/margin and class scores required")
    return bool(seen and prediction-margin>.001 and (not full or (prob[2]>=.8 and prob[0]<=.1)))
