# Measured CPU navigation results

Executed 3 November 2026. Three separate SAC policies were trained for 30,000 steps
each with seeds 7, 17 and 29. Each performed 29,500 optimizer updates after 500 warmup
steps. Width 64, batch 64, learning rate0.0003 and max 250 steps per episode were
fixed before evaluation. The frozen suite uses episode seeds 100000–100049.
No learning parameter or controller was tuned against these results.

| Policy | Successes | Obstacle stops | Timeouts | Success rate |
|---|---:|---:|---:|---:|
| SAC training seed 7 |50/50|0|0|100%|
| SAC training seed 17 |48/50|1|1|96%|
| SAC training seed 29 |47/50|3|0|94%|
| Declared geometric baseline |20/50|0|30|40%|

These are the same 50initial conditions evaluated with each policy, not 150
independent environment designs. The three-policy mean is 96.7%, with a 94–100%
range. The geometric baseline is a simple goal-heading/front-clearance rule,
not a tuned planner or Nav2 benchmark. The results do not establish general
superiority of SAC over classical navigation.

The environment is a deterministic2-D unicycle simulator with three fixed
circular obstacles, walls and ideal ranges/pose. Starts/goals differ by seed;
maps, sensor noise, wheel slip, dynamic agents and hardware do not vary.
`obstacle_stop` is a clearance violation, not a measured physical contact.
The failed missions of seeds 17 and 29 remain in the machine-readable evidence.
There is no claim of 100% reliability for the project as a whole.

`cpu_evidence.json` records the source fingerprint, exact runtime, checkpoint
and trajectory hashes, configuration and every episode outcome. Full local
training/evaluation artifacts are in`runs/npz-*`and are deliberately not
checked in. The checkpoints use bounded non-pickle NPZ arrays; model data is
not loaded through`torch.load`. CI publishes a small executable-pipeline
artifact, not these 30,000-step performance experiments.

The checkpoint-format change was verified by repeating all three training
runs and evaluations with unchanged learning settings. The earlier private
pickle-format pilot remains historical; published evidence refers to the
final NPZ pipeline only. Source fingerprints remained unchanged during each
published training/evaluation run.

Software checks cover actual parameter updates, termination masks, numerical
stability, sensor rejection, checkpoint rejection/roundtrip, deterministic
evaluation, error cleanup and ROS message/service contracts with mocks.
An installed wheel also executes the full train/save/load/evaluate proof.

ROS1/Gazebo container execution is unverified on this revision. The local
Docker engine was unavailable; the applicant explicitly chose CPU-only
validation for now. Mock ROS tests and these ideal-sensor results do not
establish Gazebo compatibility, sim-to-real transfer or hardware safety.
