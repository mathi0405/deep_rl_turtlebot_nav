import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from turtlebot_nav.sensors import SensorError, lidar_bins, observation, transition_reward, velocity
from turtlebot_nav.simulation import KinematicEnv
from turtlebot_nav.sac import Actor, Replay, SAC
from turtlebot_nav.evidence import load_policy, save_policy, source_digest
from turtlebot_nav.runner import evaluate, seed_all, train


class SensorTests(unittest.TestCase):
    def test_every_scan_element_is_covered(self):
        for n in [24, 25, 359, 360, 361, 721]:
            for i in [0, n//2, n-1]:
                scan = np.full(n, 3.5)
                scan[i] = 0.05
                self.assertEqual(len(lidar_bins(scan)), 24)
                self.assertAlmostEqual(float(lidar_bins(scan).min()), 0.05)

    def test_invalid_scans_fail_closed(self):
        for scan in [[], [1]*10, [np.nan]*360, [-np.inf]*360, [0]*360, [-1]*360, [[1]*24]]:
            with self.subTest(scan=str(scan)[:20]), self.assertRaises(SensorError):
                lidar_bins(scan)

    def test_positive_infinity_is_no_return(self):
        np.testing.assert_array_equal(lidar_bins([np.inf]*360), np.full(24, 3.5))

    def test_goal_is_not_constant_and_wraps(self):
        a, d = observation([1]*24, [0, 0, np.pi], [1, 0])
        b, _ = observation([1]*24, [0, 0, 0], [0, 2])
        self.assertEqual(d, 1)
        self.assertAlmostEqual(float(a[-1]), -1)
        self.assertAlmostEqual(float(b[-1]), 0.5)
        self.assertFalse(np.array_equal(a[-2:], b[-2:]))

    def test_invalid_pose_and_action(self):
        with self.assertRaises(SensorError):
            observation([1]*24, [np.nan, 0, 0], [1, 0])
        for action in [[2, 0], [np.nan, 0], [0], [0, 0, 0]]:
            with self.assertRaises(ValueError):
                velocity(action)
        self.assertEqual(velocity([-1, 0]), (0, 0))
        self.assertEqual(velocity([1, 1]), (0.22, 2))

    def test_outcomes_and_time_limit_are_distinct(self):
        self.assertEqual(transition_reward(1, 0.1, False, True, False)[1:], (True, False, 'success'))
        self.assertEqual(transition_reward(1, 1, False, False, True)[1:], (False, True, 'timeout'))
        self.assertEqual(transition_reward(1, .1, True, True, True)[1:], (True, False, 'obstacle_stop'))


class SimulationTests(unittest.TestCase):
    def test_seed_and_trace_are_reproducible(self):
        a, b = KinematicEnv(max_steps=8), KinematicEnv(max_steps=8)
        np.testing.assert_array_equal(a.reset(13), b.reset(13))
        for _ in range(8):
            ra, rb = a.step([-1, 0]), b.step([-1, 0])
            np.testing.assert_array_equal(ra[0], rb[0])
            self.assertEqual(ra[1:], rb[1:])
        self.assertEqual(ra[4]['outcome'], 'timeout')
        with self.assertRaises(RuntimeError):
            a.step([-1, 0])

    def test_invalid_action_does_not_mutate_pose(self):
        env = KinematicEnv()
        env.reset(2)
        original = env.pose.copy()
        with self.assertRaises(ValueError):
            env.step([np.inf, 0])
        np.testing.assert_array_equal(original, env.pose)

    def test_goal_completion(self):
        env = KinematicEnv()
        env.reset(2)
        env.goal = env.pose[:2].copy()
        result = env.step([-1, 0])
        self.assertEqual(result[4]['outcome'], 'success')


class LearningTests(unittest.TestCase):
    def setUp(self):
        seed_all(7)

    def replay(self, terminal=False):
        replay = Replay(26, 64, 3)
        for i in range(32):
            replay.add(np.full(26, i/32), [0.2, -.2], 2.0, np.full(26, (i+1)/32), terminal)
        return replay

    def test_replay_ring_and_validation(self):
        replay = Replay(26, 3, 3)
        with self.assertRaises(ValueError):
            replay.sample(1)
        for i in range(10):
            replay.add(np.full(26, i), [0, 0], 0, np.zeros(26), False)
        self.assertEqual(replay.size, 3)
        self.assertEqual(set(replay.sample(3)[0][:, 0].tolist()), {7, 8, 9})
        with self.assertRaises(ValueError):
            replay.add(np.full(26, np.nan), [0, 0], 0, np.zeros(26), False)

    def test_squashed_policy_is_finite_at_saturation(self):
        actor = Actor(26, 16)
        actions, logs = actor.sample(torch.full((8, 26), 1000.0))
        self.assertTrue(torch.isfinite(logs).all())
        self.assertTrue((actions.abs() <= 1).all())
        actions, _ = actor.sample(torch.zeros(1, 26), deterministic=True)
        again, _ = actor.sample(torch.zeros(1, 26), deterministic=True)
        self.assertTrue(torch.equal(actions, again))

    def test_real_optimizer_updates_actor_critic_and_targets(self):
        agent = SAC(width=16)
        actor = [p.clone() for p in agent.actor.parameters()]
        critic = [p.clone() for p in agent.critic.parameters()]
        target = [p.clone() for p in agent.target.parameters()]
        metrics = agent.update(self.replay(), 8)
        self.assertTrue(all(np.isfinite(x) for x in metrics.values()))
        for old, new in [(actor, agent.actor.parameters()), (critic, agent.critic.parameters()),
                         (target, agent.target.parameters())]:
            self.assertTrue(any(not torch.equal(a, b) for a, b in zip(old, new)))
        self.assertEqual(agent.updates, 1)

    def test_terminal_target_has_no_bootstrap(self):
        agent = SAC(width=16)
        for p in agent.critic.parameters():
            p.data.zero_()
        metrics = agent.update(self.replay(terminal=True), 8)
        self.assertAlmostEqual(metrics['critic_loss'], 8.0, places=5)

    def test_checkpoint_roundtrip_and_failures(self):
        agent = SAC(width=16)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)/'policy.npz'
            meta = dict(backend='cpu-kinematic-v1', source_sha256=source_digest())
            save_policy(agent, p, meta)
            loaded, _ = load_policy(p)
            state = np.zeros(26, np.float32)
            np.testing.assert_array_equal(agent.act(state, True), loaded.act(state, True))
            with self.assertRaises(FileNotFoundError):
                load_policy(Path(td)/'missing.pt')
            bad = dict(format_version=1, config=dict(state_dim=999999, width=999999), metadata=meta)
            torch.save(bad, p)
            with self.assertRaises(ValueError):
                load_policy(p)

    def test_object_arrays_are_rejected_without_unpickling(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)/'object.npz'
            np.savez(p, payload=np.array([{'not': 'a tensor'}], dtype=object))
            with self.assertRaises(ValueError):
                load_policy(p)

    def test_oversized_network_config_is_rejected_before_allocation(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)/'oversized.npz'
            np.savez(p, format_version=np.array(1), updates=np.array(0),
                     config_json=np.array(json.dumps(dict(state_dim=26, width=10**9,
                                                         learning_rate=.0003, gamma=.99, tau=.005))),
                     metadata_json=np.array(json.dumps(dict(backend='cpu', source_sha256='x'))))
            with self.assertRaises(ValueError):
                load_policy(p)


class PipelineTests(unittest.TestCase):
    def test_train_save_load_evaluate_and_negative_controls(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            summary = train(out/'train', steps=96, warmup=16, batch_size=8, width=16, max_steps=8)
            self.assertEqual(summary['actual_steps'], 96)
            self.assertEqual(summary['updates'], 80)
            a = evaluate(out/'eval', out/'train/policy.npz', episodes=3, seed=100, max_steps=8)
            b = evaluate(out/'repeat', out/'train/policy.npz', episodes=3, seed=100, max_steps=8)
            self.assertEqual(a['episodes'], b['episodes'])
            self.assertEqual(a['trajectories_sha256'], b['trajectories_sha256'])
            self.assertEqual(sum(a['counts'].values()), 3)
            with self.assertRaises(FileExistsError):
                evaluate(out/'eval', out/'train/policy.npz', episodes=1, max_steps=8)
            with self.assertRaises(FileNotFoundError):
                evaluate(out/'missing', out/'not-there.pt')
            self.assertFalse((out/'missing').exists())
            self.assertEqual(len(json.loads((out/'eval/evaluation.json').read_text())['episodes']), 3)

    def test_cleanup_on_environment_failure(self):
        class Broken(KinematicEnv):
            closed = False
            def reset(self, seed):
                raise RuntimeError('injected sensor fault')
            def close(self):
                self.closed = True
        env = Broken()
        with tempfile.TemporaryDirectory() as td, patch('turtlebot_nav.runner.environment', return_value=env):
            with self.assertRaises(RuntimeError):
                train(Path(td)/'failure', steps=32, warmup=8, batch_size=4)
        self.assertTrue(env.closed)


if __name__ == '__main__':
    unittest.main()
