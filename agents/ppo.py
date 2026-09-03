"""Minimal PPO (clipped surrogate, GAE) with separate actor / critic networks — mirrors the
baseline paper's setup (Table II) and keeps full control over which transitions each learner
sees (needed for the region-specialised variants).

Differences from the paper that are deliberate and documented:
  * policy std is a state-independent learnable parameter (paper: sigma(s)); more stable for
    small residual actions and the usual choice in continuous-control PPO.
  * observations are normalised with running statistics (frozen at evaluation).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


class RunningMeanStd:
    def __init__(self, shape, eps: float = 1e-4):
        self.mean = np.zeros(shape, np.float64)
        self.var = np.ones(shape, np.float64)
        self.count = eps

    def update(self, x: np.ndarray):
        x = np.asarray(x, np.float64).reshape(-1, self.mean.shape[0])
        bm, bv, bc = x.mean(0), x.var(0), x.shape[0]
        delta = bm - self.mean
        tot = self.count + bc
        self.mean = self.mean + delta * bc / tot
        m_a, m_b = self.var * self.count, bv * bc
        self.var = (m_a + m_b + delta ** 2 * self.count * bc / tot) / tot
        self.count = tot

    def normalize(self, x):
        return (x - self.mean) / np.sqrt(self.var + 1e-8)

    def state(self):
        return {"mean": self.mean.copy(), "var": self.var.copy(), "count": self.count}

    def load(self, s):
        self.mean, self.var, self.count = s["mean"].copy(), s["var"].copy(), s["count"]


def mlp(inp, hidden, out, gain=0.1):
    layers, d = [], inp
    for h in hidden:
        lin = nn.Linear(d, h)
        nn.init.xavier_normal_(lin.weight, gain=gain)
        nn.init.zeros_(lin.bias)
        layers += [lin, nn.Tanh()]
        d = h
    lin = nn.Linear(d, out)
    nn.init.xavier_normal_(lin.weight, gain=gain)
    nn.init.zeros_(lin.bias)
    layers.append(lin)
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim=1, hidden=(64, 64), init_gain=0.1, log_std_init=-1.0):
        super().__init__()
        self.mu = mlp(obs_dim, hidden, act_dim, init_gain)
        self.log_std = nn.Parameter(torch.full((act_dim,), float(log_std_init)))

    def dist(self, obs):
        mu = torch.tanh(self.mu(obs))
        return torch.distributions.Normal(mu, self.log_std.exp().expand_as(mu))

    @torch.no_grad()
    def act(self, obs_np: np.ndarray, deterministic: bool = False):
        obs = torch.as_tensor(obs_np, dtype=torch.float32).unsqueeze(0)
        d = self.dist(obs)
        a = d.mean if deterministic else d.sample()
        return a.squeeze(0).numpy(), d.log_prob(a).sum(-1).item()


class Critic(nn.Module):
    def __init__(self, obs_dim, hidden=(64, 64), init_gain=0.1):
        super().__init__()
        self.v = mlp(obs_dim, hidden, 1, init_gain)

    def forward(self, obs):
        return self.v(obs).squeeze(-1)


@dataclass
class PPOConfig:
    actor_lr: float = 1e-4
    critic_lr: float = 1e-3
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.1
    batch_size: int = 1024
    n_epochs: int = 10
    hidden: tuple = (64, 64)
    init_gain: float = 0.1
    max_grad_norm: float = 0.5
    entropy_coef: float = 0.0
    log_std_init: float = -1.0


class PPOLearner:
    """One actor/critic pair + its own transition buffer."""

    def __init__(self, obs_dim: int, cfg: PPOConfig, name: str = "agent", act_dim: int = 1):
        self.cfg = cfg
        self.name = name
        self.actor = Actor(obs_dim, act_dim, cfg.hidden, cfg.init_gain, cfg.log_std_init)
        self.critic = Critic(obs_dim, cfg.hidden, cfg.init_gain)
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
        self.obs_rms = RunningMeanStd((obs_dim,))
        self.n_updates = 0
        self.n_samples = 0

    # -------------------------------------------------------------- snapshot for workers
    def snapshot(self) -> dict:
        return {"actor": copy.deepcopy(self.actor.state_dict()), "obs_rms": self.obs_rms.state()}

    # -------------------------------------------------------------- GAE on one contiguous segment
    def _gae(self, rew, val, last_val):
        T = len(rew)
        adv = np.zeros(T, np.float32)
        last = 0.0
        for t in reversed(range(T)):
            nv = last_val if t == T - 1 else val[t + 1]
            delta = rew[t] + self.cfg.gamma * nv - val[t]
            last = delta + self.cfg.gamma * self.cfg.gae_lambda * last
            adv[t] = last
        return adv, adv + val

    def prepare(self, segments: list[dict]) -> dict | None:
        """segments: list of {'obs','act','logp','rew','terminal'} contiguous in time for this
        learner. Returns a flat batch with advantages/returns."""
        if not segments:
            return None
        # Normalise with the statistics the workers used when collecting (the snapshot sent with the
        # rollout job), so that logp_old and the new log-probs refer to the same inputs; only then
        # fold the new batch into the running statistics for the *next* rollout.
        obs_all = np.concatenate([s["obs"] for s in segments])
        O, A, LP, ADV, RET = [], [], [], [], []
        with torch.no_grad():
            for s in segments:
                o = self.obs_rms.normalize(s["obs"]).astype(np.float32)
                v = self.critic(torch.as_tensor(o)).numpy()
                last_val = 0.0 if s["terminal"] else float(self.critic(
                    torch.as_tensor(self.obs_rms.normalize(s["last_obs"]).astype(np.float32)).unsqueeze(0)))
                adv, ret = self._gae(s["rew"].astype(np.float32), v, last_val)
                O.append(o); A.append(s["act"]); LP.append(s["logp"]); ADV.append(adv); RET.append(ret)
        self.obs_rms.update(obs_all)
        return {"obs": np.concatenate(O), "act": np.concatenate(A).astype(np.float32),
                "logp": np.concatenate(LP).astype(np.float32), "adv": np.concatenate(ADV),
                "ret": np.concatenate(RET)}

    def update(self, batch: dict) -> dict:
        cfg = self.cfg
        obs = torch.as_tensor(batch["obs"]); act = torch.as_tensor(batch["act"]).reshape(len(obs), -1)
        logp_old = torch.as_tensor(batch["logp"]); ret = torch.as_tensor(batch["ret"])
        adv = torch.as_tensor(batch["adv"])
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        n = len(obs)
        stats = {"pi_loss": 0.0, "v_loss": 0.0, "clipfrac": 0.0, "approx_kl": 0.0, "n": 0}
        for _ in range(cfg.n_epochs):
            perm = torch.randperm(n)
            for i in range(0, n, cfg.batch_size):
                idx = perm[i:i + cfg.batch_size]
                d = self.actor.dist(obs[idx])
                logp = d.log_prob(act[idx]).sum(-1)
                ratio = torch.exp(logp - logp_old[idx])
                s1 = ratio * adv[idx]
                s2 = torch.clamp(ratio, 1 - cfg.clip_range, 1 + cfg.clip_range) * adv[idx]
                pi_loss = -torch.min(s1, s2).mean() - cfg.entropy_coef * d.entropy().sum(-1).mean()
                self.opt_a.zero_grad(); pi_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.max_grad_norm); self.opt_a.step()
                v_loss = 0.5 * ((self.critic(obs[idx]) - ret[idx]) ** 2).mean()
                self.opt_c.zero_grad(); v_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.max_grad_norm); self.opt_c.step()
                with torch.no_grad():
                    stats["pi_loss"] += pi_loss.item(); stats["v_loss"] += v_loss.item()
                    stats["clipfrac"] += ((ratio - 1).abs() > cfg.clip_range).float().mean().item()
                    stats["approx_kl"] += (logp_old[idx] - logp).mean().item(); stats["n"] += 1
        k = max(stats.pop("n"), 1)
        self.n_updates += 1
        self.n_samples += n
        return {f"{self.name}/{kk}": v / k for kk, v in stats.items()} | {
            f"{self.name}/std": float(self.actor.log_std.exp().mean().detach()), f"{self.name}/batch": n}

    def state_dict(self):
        return {"actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
                "obs_rms": self.obs_rms.state(), "n_updates": self.n_updates, "n_samples": self.n_samples}

    def load_state_dict(self, s):
        self.actor.load_state_dict(s["actor"]); self.critic.load_state_dict(s["critic"])
        self.obs_rms.load(s["obs_rms"]); self.n_updates = s["n_updates"]; self.n_samples = s["n_samples"]


# ====================================================================== specialised actors, shared critic
class _ActorView:
    """Learner-like handle for one region of a SharedCriticPPO (snapshot / state for the trainer)."""

    def __init__(self, parent: "SharedCriticPPO", region: int, name: str):
        self.parent, self.region, self.name = parent, region, name

    def snapshot(self) -> dict:
        return {"actor": copy.deepcopy(self.parent.actors[self.region].state_dict()),
                "obs_rms": self.parent.obs_rms.state()}

    def state_dict(self):
        return self.parent.state_dict()

    def load_state_dict(self, s):
        self.parent.load_state_dict(s)


class SharedCriticPPO:
    """Region-specialised actors with ONE critic and one observation normaliser.
    GAE is computed over the whole trajectory (bootstrapping through region switches with the
    shared critic), then each actor is updated on the steps where it acted. This removes the
    truncation-at-switch bias of the independent-learner variant (`spec`)."""

    def __init__(self, obs_dim: int, cfg: PPOConfig, regions=(0, 1), names=("R2", "R3"), act_dim: int = 1):
        self.cfg = cfg
        self.regions = tuple(regions)
        self.names = dict(zip(regions, names))
        self.actors = {r: Actor(obs_dim, act_dim, cfg.hidden, cfg.init_gain, cfg.log_std_init) for r in regions}
        self.critic = Critic(obs_dim, cfg.hidden, cfg.init_gain)
        self.opt_a = {r: torch.optim.Adam(self.actors[r].parameters(), lr=cfg.actor_lr) for r in regions}
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
        self.obs_rms = RunningMeanStd((obs_dim,))
        self.n_updates = 0
        self.n_samples = 0

    def view(self, region: int) -> _ActorView:
        return _ActorView(self, region, self.names[region])

    def _gae(self, rew, val, last_val):
        T = len(rew)
        adv = np.zeros(T, np.float32)
        last = 0.0
        for t in reversed(range(T)):
            nv = last_val if t == T - 1 else val[t + 1]
            delta = rew[t] + self.cfg.gamma * nv - val[t]
            last = delta + self.cfg.gamma * self.cfg.gae_lambda * last
            adv[t] = last
        return adv, adv + val

    def update_from_trajectories(self, trajs: list[dict], min_batch: int = 2048) -> dict:
        cfg = self.cfg
        per = {r: {"obs": [], "act": [], "logp": [], "adv": []} for r in self.regions}
        O_all, RET_all = [], []
        with torch.no_grad():
            for tr in trajs:
                o = self.obs_rms.normalize(tr["obs"]).astype(np.float32)      # stats as used by the workers
                v = self.critic(torch.as_tensor(o)).numpy()
                last_val = 0.0 if tr["terminated"] else float(self.critic(
                    torch.as_tensor(self.obs_rms.normalize(tr["last_obs"]).astype(np.float32)).unsqueeze(0)))
                adv, ret = self._gae(tr["rew"].astype(np.float32), v, last_val)
                O_all.append(o); RET_all.append(ret)
                reg = tr["region"]
                for r in self.regions:
                    m = reg == r
                    if m.any():
                        per[r]["obs"].append(o[m]); per[r]["act"].append(tr["act"][m])
                        per[r]["logp"].append(tr["logp"][m]); per[r]["adv"].append(adv[m])
        self.obs_rms.update(np.concatenate([tr["obs"] for tr in trajs]))
        stats = {}
        # ---- critic on every step
        obs = torch.as_tensor(np.concatenate(O_all)); ret = torch.as_tensor(np.concatenate(RET_all))
        n = len(obs); vl = 0.0; k = 0
        for _ in range(cfg.n_epochs):
            perm = torch.randperm(n)
            for i in range(0, n, cfg.batch_size):
                idx = perm[i:i + cfg.batch_size]
                v_loss = 0.5 * ((self.critic(obs[idx]) - ret[idx]) ** 2).mean()
                self.opt_c.zero_grad(); v_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.max_grad_norm); self.opt_c.step()
                vl += v_loss.item(); k += 1
        stats["critic/v_loss"] = vl / max(k, 1); stats["critic/batch"] = n
        # ---- each actor on its own steps
        for r in self.regions:
            name = self.names[r]
            if not per[r]["obs"]:
                stats[f"{name}/skipped_n"] = 0
                continue
            o = torch.as_tensor(np.concatenate(per[r]["obs"]))
            a = torch.as_tensor(np.concatenate(per[r]["act"]).astype(np.float32)).reshape(len(o), -1)
            lp_old = torch.as_tensor(np.concatenate(per[r]["logp"]).astype(np.float32))
            adv = torch.as_tensor(np.concatenate(per[r]["adv"]))
            if len(o) < min_batch:
                stats[f"{name}/skipped_n"] = len(o)
                continue
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            actor, opt = self.actors[r], self.opt_a[r]
            pl = cf = kl = 0.0; k = 0; m = len(o)
            for _ in range(cfg.n_epochs):
                perm = torch.randperm(m)
                for i in range(0, m, cfg.batch_size):
                    idx = perm[i:i + cfg.batch_size]
                    d = actor.dist(o[idx])
                    logp = d.log_prob(a[idx]).sum(-1)
                    ratio = torch.exp(logp - lp_old[idx])
                    s1 = ratio * adv[idx]
                    s2 = torch.clamp(ratio, 1 - cfg.clip_range, 1 + cfg.clip_range) * adv[idx]
                    pi_loss = -torch.min(s1, s2).mean() - cfg.entropy_coef * d.entropy().sum(-1).mean()
                    opt.zero_grad(); pi_loss.backward()
                    nn.utils.clip_grad_norm_(actor.parameters(), cfg.max_grad_norm); opt.step()
                    with torch.no_grad():
                        pl += pi_loss.item(); cf += ((ratio - 1).abs() > cfg.clip_range).float().mean().item()
                        kl += (lp_old[idx] - logp).mean().item(); k += 1
            k = max(k, 1)
            stats.update({f"{name}/pi_loss": pl / k, f"{name}/clipfrac": cf / k, f"{name}/approx_kl": kl / k,
                          f"{name}/std": float(actor.log_std.exp().mean().detach()), f"{name}/batch": m})
            self.n_samples += m
        self.n_updates += 1
        return stats

    def state_dict(self):
        return {"actors": {r: self.actors[r].state_dict() for r in self.regions}, "critic": self.critic.state_dict(),
                "obs_rms": self.obs_rms.state(), "n_updates": self.n_updates, "n_samples": self.n_samples,
                "shared_critic": True}

    def load_state_dict(self, s):
        for r in self.regions:
            self.actors[r].load_state_dict(s["actors"][r])
        self.critic.load_state_dict(s["critic"]); self.obs_rms.load(s["obs_rms"])
        self.n_updates = s["n_updates"]; self.n_samples = s["n_samples"]
