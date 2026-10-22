"""I/O-only repair for the frozen v1 audit hook's duplicate status keyword."""
import json
import time
import torch
from . import safestep_author_training as frozen_v1


class AuditV2(frozen_v1.Audit):
    def save_hook(self,model):
        env=self.env
        row=dict(episode=env.episode,reset_seed=env.reset_seed,start_x=env.start[0],start_y=env.start[1],
            steps=env.k-1,status=env.last_status.name,FED=env.last_fed,reward=env.total_reward,
            loss=self.losses[-1] if self.updates>env.updates_at_reset else 0.0,
            updates=self.updates,epsilon=self.cfg["common_train_arguments"]["exploration_rate"]*
            self.cfg["common_train_arguments"]["exploration_decay"]**self.updates,
            elapsed_s=time.monotonic()-self.started)
        self.writer.writerow(row);self.episode_rows.append(row)
        if env.episode % self.cfg["checkpoint_every_episodes"]==0:
            torch.save(model.policy_dqn.state_dict(),self.out/f"checkpoint_{env.episode:05d}.pth")
        if env.episode % self.cfg["progress_every_episodes"]==0:
            self.handle.flush()
            # Keep episode outcome separate from the process's running status.
            progress=row|dict(status="training",episode_status=row["status"],
                              seed=self.seed,transitions=self.transitions)
            (self.out/"progress.json").write_text(json.dumps(progress,indent=2),encoding="utf-8")


def run_training(*args,**kwargs):
    previous=frozen_v1.Audit
    frozen_v1.Audit=AuditV2
    try:
        return frozen_v1.run_training(*args,**kwargs)
    finally:
        frozen_v1.Audit=previous
