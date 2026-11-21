# Navigation pipeline

The supported learner is continuous-action SAC. The old five-algorithm
factory has been retired rather than presenting network definitions as
complete algorithms.

`LaserScan + world-referenced odometry + goal -> validated 26-value state ->
SAC actor -> normalized action -> velocity command -> bounded episode ->
replay -> twin Q/actor/temperature updates -> checkpoint -> deterministic
evaluation -> JSON results and trajectories`.

The observation contains24 range bins normalized by 3.5 m, clipped goal distance
divided by 8 m, and wrapped goal bearing divided bypi. Every scan element belongs
to a bin. Positive infinity means no return; NaN, negative infinity,
nonpositive ranges and undersized scans fail closed. A normalized action maps
to forward speed0–0.22m/s and angular speed−2–2rad/s. Physical robots are not
supported.

## Training

SAC samples a squashed Gaussian with the change-of-variables log-probability
correction. Two Q networks minimize Bellman residuals against slowly updated
target networks. The actor uses the minimum Q and entropy term; temperature
is learned with target entropy−2. Gradient norms and numerical outputs are
checked. Replay stores actual termination separately from time-limit
truncation so a timeout does not incorrectly suppress the bootstrap.

The checkpoint is a non-pickle NPZ archive containing actor arrays and JSON metadata. Archive sizes, NPY headers, tensor names, shapes, dtypes and finiteness are checked before loading with `allow_pickle=False`. No `torch.load` call is used. It is an inference checkpoint, not an exact resume
snapshot: replay, optimizer state and RNG state are deliberately not promised.
Missing checkpoints never substitute random policies. Deterministic
evaluation uses tanh(mean), not a fresh SAC sample.

## CPU backend

This is a small2-D unicycle approximation with ray/circle/wall intersections,
three static circles, ideal pose/ranges, and a0.2s timestep. It is not a
TurtleBot physics model. Its purpose is to exercise the complete learning and
evaluation pipeline and make controlled faults reproducible. Goal positions
and starts are sampled reproducibly with minimum spacing and obstacle margin.
It does not model noise, wheel slip, pedestrians, dynamics or real-world
sensor failures. A finite clear run is not a safety guarantee.

## ROS1/Gazebo backend

The adapter requires `/use_sim_time=true`, the provided world and a
world-referenced TurtleBot odometry source. It uses isolated
`/turtlebot_lab/{cmd_vel,scan,odom}` topics. Sensors must arrive after command
publication, and scan/odometry stamps must align within 0.15 s. Scan angles are
reordered into the CPU contract's−pi..pi convention. The laser must face
base-forward and cover a full circle with at least3.5m range.

Gazebo resets use the same sampled start/goal coordinates as the CPU world.
The initial odometry is checked against the requested world reset. Service
calls and sensor waits have wall-clock limits. Command publication is
followed by a zero command and physics pause on normal completion or error.
The training node must not be connected to a shared physical robot ROS master.
There is no hardware mode, hardware watchdog or deployment approval.

The ROS step duration depends on new sensor publications and is not an exact
0.2s match to the CPU integrator. Physics, scan placement and sensor timing
also differ. Checkpoints cannot cross backends without an explicit transfer
flag, and such an experiment is labelled as a backend transfer.

## Outcomes and evidence

Episodes end as success, obstacle_stop or timeout. `obstacle_stop` is an
operational clearance stop and does not claim a contact sensor measured a
physical collision. Errors such as invalid sensors abort the run; they do not
become successful or silently omitted episodes. Outputs include exact seed,
step limits, runtime versions, source/model digests, rewards, outcomes and
trajectories. Checksums aid reproducibility/integrity, not authenticity.

A passing CI pipeline establishes software contracts and executable plumbing,
not navigation quality. Policy quality is reported separately over the frozen
suite in `configs/qualification.json`, including every failed mission.

References: [SAC paper](https://arxiv.org/abs/1801.01290),
[Spinning Up equations](https://spinningup.openai.com/en/latest/algorithms/sac.html),
[PyTorch reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html).
