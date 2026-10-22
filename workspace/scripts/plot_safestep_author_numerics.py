"""Plot original hotel topology, stored-weight paths and reproduced FED results."""
import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from fire_evac.safestep_author_numerics import load_author_inputs

out=ROOT/"outputs/safestep_paper_numerics_20261015/numeric_run"
j=json.loads((out/"results.json").read_text(encoding="utf-8"))
inputs=load_author_inputs(ROOT/"outputs/safestep_paper_numerics_20261015/author_repository")
fig,axes=plt.subplots(1,3,figsize=(15,5.2),gridspec_kw={"width_ratios":[1,1,1.1]},layout="constrained")
for ax,policy,alternative in zip(axes[:2],j["stored_weight_runs"],j["fixed_routes"][1::2]):
    ax.imshow(inputs["grid"],cmap="Greys",vmin=0,vmax=1,alpha=.8)
    path=np.array(policy["path"])
    alt=np.array(alternative["path"])
    ax.plot(alt[:,0],alt[:,1],"--",color="#c26b29",label="Alternative Exit A",linewidth=2)
    ax.plot(path[:,0],path[:,1],"o-",color="#176a9b",label="Author DQN to Exit B",markersize=3,linewidth=2)
    ax.scatter(*path[0],color="#bd2334",marker="*",s=110,zorder=4,label="Start")
    for xy,name in zip(inputs["exits"],["A","B"]):
        ax.scatter(*xy,marker="s",color="#148055",s=90,zorder=4)
        ax.text(xy[0]+.6,xy[1]+.6,name,color="#148055",weight="bold")
    ax.set_title(f"Start {1 if policy['steps']==10 else 2}: {policy['steps']} steps, {policy['time_s']:.1f} s\nFED {policy['FED']:.4f}",fontsize=11)
    ax.set_xticks([]);ax.set_yticks([])
axes[0].legend(loc="lower right",fontsize=7)
rows=j["fixed_route_comparisons"]
y=np.arange(4)
axes[2].barh(y+.16,[r["paper_FED"] for r in rows],height=.30,color="#b8c6ce",label="Paper (4 decimals)")
axes[2].barh(y-.16,[r["reproduced_FED"] for r in rows],height=.30,color="#176a9b",label="Recomputed from author data")
axes[2].set_yticks(y,["Start 1 / B","Start 1 / A","Start 2 / B","Start 2 / A"])
axes[2].invert_yaxis()
axes[2].set_xlabel("Personal FED")
axes[2].axvline(.3,color="#a63434",linestyle=":",linewidth=1,label="FED 0.3 reference")
axes[2].legend(fontsize=7,loc="upper center",bbox_to_anchor=(.5,-.16),frameon=False)
axes[2].set_title("All four match at published precision",fontsize=11)
fig.suptitle("SafeStep hotel case: original data + original stored weights, no retraining",fontsize=13)
fig.savefig(out/"hotel_numerical_reproduction.png",dpi=180)
plt.close(fig)
print(out/"hotel_numerical_reproduction.png")
