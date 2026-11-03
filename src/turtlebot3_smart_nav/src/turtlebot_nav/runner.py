"""Bounded train -> checkpoint -> evaluate pipeline with per-episode evidence."""
from __future__ import annotations

import json
import random
import time
import numpy as np
import torch
from .evidence import (digest, load_policy, new_directory, runtime, save_policy,
                       source_digest, write_json)
from .sac import Replay, SAC
from .simulation import KinematicEnv


def seed_all(seed):
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**31:
        raise ValueError('seed must be an integer in [0,2**31)')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def environment(backend, max_steps):
    if backend == 'cpu':
        return KinematicEnv(max_steps=max_steps)
    if backend == 'ros':
        from .ros_env import ROSNavigationEnv
        return ROSNavigationEnv(max_steps=max_steps)
    raise ValueError('backend must be cpu or ros')


def train(out, backend='cpu', steps=10000, seed=7, max_steps=250,
          warmup=500, batch_size=64, width=64, learning_rate=3e-4):
    seed_all(seed)
    if (steps <= 0 or max_steps <= 0 or warmup < batch_size or batch_size < 2
            or steps <= warmup or not 4 <= width <= 256):
        raise ValueError('require steps>warmup>=batch_size>=2, max_steps>0 and width in[4,256]')
    agent = SAC(width=width, learning_rate=learning_rate)
    replay = Replay(26, max(batch_size, min(steps, 100000)), seed)
    rng = np.random.default_rng(seed)
    env = environment(backend, max_steps)
    try:
        output = new_directory(out)
        source = source_digest()
        meta = dict(backend=env.backend, source_sha256=source, runtime=runtime(), seed=seed,
                    requested_steps=steps, max_steps=max_steps, warmup=warmup,
                    batch_size=batch_size, inference_checkpoint_only=True,
                    observation_contract='24range/max3.5m,distance/8m,bearing/pi; clipped distance',
                    action_contract='normalized[-1,1]^2 -> forward[0,.22]m/s, yaw[-2,2]rad/s')
        write_json(output/'configuration.json', dict(meta, agent=agent.config))
        episode, episode_steps, total_reward, completed = 0, 0, 0.0, []
        state = env.reset(seed)
        started = time.monotonic()
        with (output/'optimization.jsonl').open('w', encoding='utf-8') as logs:
            for t in range(steps):
                action = rng.uniform(-1, 1, 2).astype(np.float32) if t < warmup else agent.act(state)
                next_state, reward, terminated, truncated, info = env.step(action)
                replay.add(state, action, reward, next_state, terminated)
                state = next_state
                total_reward += reward
                episode_steps += 1
                if t >= warmup:
                    metrics = agent.update(replay, batch_size)
                    logs.write(json.dumps(dict(step=t+1, **metrics), allow_nan=False)+'\n')
                if terminated or truncated:
                    completed.append(dict(episode=episode, seed=seed+episode, reward=total_reward, **info))
                    episode += 1
                    total_reward, episode_steps = 0.0, 0
                    if t + 1 < steps:
                        state = env.reset(seed+episode)
        if source_digest() != source:
            raise RuntimeError('source changed during training; checkpoint publication refused')
        checksum = save_policy(agent, output/'policy.npz', meta)
        summary = dict(meta, actual_steps=steps, updates=agent.updates, completed_episodes=completed,
                       unfinished_episode_steps=episode_steps, policy_sha256=checksum,
                       seconds=time.monotonic()-started,
                       scope='Training execution evidence; no navigation performance or hardware guarantee.')
        write_json(output/'training.json', summary)
        return summary
    finally:
        env.close()


def evaluate(out, policy, backend='cpu', episodes=20, seed=10000, max_steps=250,
             baseline=False, allow_backend_transfer=False):
    seed_all(seed)
    if episodes < 1 or max_steps < 1:
        raise ValueError('episodes/max_steps must be positive')
    if baseline and policy:
        raise ValueError('choose a checkpoint or an explicit baseline, not both')
    if not baseline and not policy:
        raise ValueError('checkpoint required unless --baseline is explicit')
    agent, metadata = (None, {}) if baseline else load_policy(policy)
    env = environment(backend, max_steps)
    try:
        transfer = bool(not baseline and metadata['backend'] != env.backend)
        if transfer and not allow_backend_transfer:
            raise ValueError('checkpoint backend differs; explicit --allow-backend-transfer required')
        output = new_directory(out)
        source = source_digest()
        records = []
        with (output/'trajectories.jsonl').open('w', encoding='utf-8') as trajectory:
            for episode in range(episodes):
                state = env.reset(seed+episode)
                total_reward = 0.0
                for step in range(max_steps):
                    if baseline:
                        # Declared geometric baseline; no learned-weight substitution.
                        angle = float(state[-1])*np.pi
                        front = float(np.min(state[10:15]))*3.5
                        turn = np.clip(angle*1.2, -1, 1)
                        if front < 0.45:
                            turn = 1.0 if state[15:21].mean() > state[3:9].mean() else -1.0
                        speed = max(0.0, 1-abs(turn)) if front > 0.3 else 0.0
                        action = np.array([2*speed-1, turn], np.float32)
                    else:
                        action = agent.act(state, deterministic=True)
                    state, reward, terminated, truncated, info = env.step(action)
                    total_reward += reward
                    trajectory.write(json.dumps(dict(episode=episode, step=step+1, action=action.tolist(),
                                                      observation=state.tolist(), **info), allow_nan=False)+'\n')
                    if terminated or truncated:
                        break
                if not (terminated or truncated):
                    raise RuntimeError('environment failed to enforce its step limit')
                records.append(dict(episode=episode, seed=seed+episode, reward=total_reward, **info))
        if source_digest() != source:
            raise RuntimeError('source changed during evaluation; result publication refused')
        counts = {key: sum(r['outcome'] == key for r in records)
                  for key in ['success', 'obstacle_stop', 'timeout']}
        if sum(counts.values()) != episodes:
            raise RuntimeError('unclassified evaluation outcome')
        summary = dict(backend=env.backend, source_sha256=source, runtime=runtime(), seed=seed,
                       max_steps=max_steps, episodes=records, counts=counts,
                       policy='geometric-baseline' if baseline else str(policy),
                       policy_sha256=None if baseline else digest(policy), checkpoint_metadata=metadata,
                       backend_transfer=transfer, success_rate=counts['success']/episodes,
                       deterministic_actions=True, trajectories_sha256=digest(output/'trajectories.jsonl'),
                       scope='Finite simulation outcomes; obstacle_stop is a clearance stop, not a measured hardware collision.')
        write_json(output/'evaluation.json', summary)
        return summary
    finally:
        env.close()
