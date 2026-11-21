"""Squashed-Gaussian SAC, twin Q targets, replay and automatic temperature.

Equations: Haarnoja et al.; OpenAI Spinning Up SAC documentation.
This implementation is new; legacy network scaffolds are not training agents.
"""
from __future__ import annotations

import copy
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def mlp(inputs, outputs, width):
    return nn.Sequential(nn.Linear(inputs, width), nn.ReLU(), nn.Linear(width, width),
                         nn.ReLU(), nn.Linear(width, outputs))


class Actor(nn.Module):
    def __init__(self, state_dim, width=64):
        super().__init__()
        self.net = mlp(state_dim, 4, width)

    def sample(self, state, deterministic=False):
        mean, log_std = self.net(state).chunk(2, dim=-1)
        log_std = log_std.clamp(-5.0, 2.0)
        distribution = torch.distributions.Normal(mean, log_std.exp())
        raw = mean if deterministic else distribution.rsample()
        action = torch.tanh(raw)
        correction = 2 * (math.log(2) - raw - F.softplus(-2 * raw))
        log_prob = (distribution.log_prob(raw) - correction).sum(-1, keepdim=True)
        return action, log_prob


class TwinQ(nn.Module):
    def __init__(self, state_dim, width=64):
        super().__init__()
        self.q1, self.q2 = mlp(state_dim + 2, 1, width), mlp(state_dim + 2, 1, width)

    def forward(self, state, action):
        x = torch.cat((state, action), -1)
        return self.q1(x), self.q2(x)


class Replay:
    def __init__(self, state_dim, capacity, seed):
        if capacity < 1:
            raise ValueError("replay capacity must be positive")
        self.state = np.empty((capacity, state_dim), np.float32)
        self.next_state = np.empty_like(self.state)
        self.action = np.empty((capacity, 2), np.float32)
        self.reward = np.empty((capacity, 1), np.float32)
        self.terminated = np.empty((capacity, 1), np.float32)
        self.capacity, self.size, self.index = capacity, 0, 0
        self.rng = np.random.default_rng(seed)

    def add(self, state, action, reward, next_state, terminated):
        s, a, ns = np.asarray(state), np.asarray(action), np.asarray(next_state)
        if s.shape != self.state.shape[1:] or ns.shape != s.shape or a.shape != (2,):
            raise ValueError("invalid transition dimensions")
        if not all(np.isfinite(x).all() for x in (s, a, ns, np.asarray(reward))):
            raise ValueError("transition must be finite")
        if np.abs(a).max() > 1.000001 or not isinstance(terminated, (bool, np.bool_)):
            raise ValueError("invalid action or termination mask")
        i = self.index
        self.state[i], self.action[i], self.reward[i] = s, a, reward
        self.next_state[i], self.terminated[i] = ns, terminated
        self.index = (i + 1) % self.capacity
        self.size = min(self.capacity, self.size + 1)

    def sample(self, count):
        if count < 1 or self.size < count:
            raise ValueError("not enough replay transitions")
        idx = self.rng.choice(self.size, count, replace=False)
        return [torch.from_numpy(x[idx]) for x in
                (self.state, self.action, self.reward, self.next_state, self.terminated)]


class SAC:
    def __init__(self, state_dim=26, width=64, learning_rate=3e-4, gamma=0.99, tau=0.005):
        if state_dim < 3 or width < 4 or not 0 < learning_rate < 1 or not 0 <= gamma <= 1 or not 0 < tau <= 1:
            raise ValueError("invalid SAC configuration")
        self.config = dict(state_dim=state_dim, width=width, learning_rate=learning_rate, gamma=gamma, tau=tau)
        self.actor, self.critic = Actor(state_dim, width), TwinQ(state_dim, width)
        self.target = copy.deepcopy(self.critic).requires_grad_(False)
        self.log_alpha = nn.Parameter(torch.tensor(math.log(0.2)))
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=learning_rate)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=learning_rate)
        self.gamma, self.tau, self.updates = gamma, tau, 0

    def act(self, state, deterministic=False):
        state = np.asarray(state, np.float32)
        if state.shape != (self.config['state_dim'],) or not np.isfinite(state).all():
            raise ValueError("invalid policy observation")
        with torch.no_grad():
            action, _ = self.actor.sample(torch.from_numpy(state).unsqueeze(0), deterministic)
        if not torch.isfinite(action).all():
            raise RuntimeError("nonfinite policy action")
        return action[0].numpy()

    def update(self, replay, batch_size):
        state, action, reward, next_state, terminated = replay.sample(batch_size)
        alpha = self.log_alpha.exp().detach()
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_state)
            q1, q2 = self.target(next_state, next_action)
            # Time-limit truncation does not make the underlying MDP absorbing.
            target = reward + self.gamma * (1 - terminated) * (torch.minimum(q1, q2) - alpha * next_log_prob)
        q1, q2 = self.critic(state, action)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0, error_if_nonfinite=True)
        self.critic_optimizer.step()
        self.critic.requires_grad_(False)
        try:
            sampled, log_prob = self.actor.sample(state)
            q1, q2 = self.critic(state, sampled)
            actor_loss = (alpha * log_prob - torch.minimum(q1, q2)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0, error_if_nonfinite=True)
            self.actor_optimizer.step()
        finally:
            self.critic.requires_grad_(True)
        alpha_loss = -(self.log_alpha * (log_prob.detach() - 2.0)).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        with torch.no_grad():
            self.log_alpha.clamp_(-10.0, 2.0)
            for target_parameter, parameter in zip(self.target.parameters(), self.critic.parameters()):
                target_parameter.lerp_(parameter, self.tau)
        self.updates += 1
        metrics = dict(actor_loss=actor_loss.detach().item(), critic_loss=critic_loss.detach().item(),
                       alpha=self.log_alpha.exp().detach().item(), alpha_loss=alpha_loss.detach().item())
        if not all(math.isfinite(x) for x in metrics.values()):
            raise RuntimeError("nonfinite optimization metrics")
        return metrics
