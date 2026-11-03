"""Small executable integration check; does not assert learned-policy quality."""
import argparse
from pathlib import Path
from turtlebot_nav.runner import evaluate, train
from turtlebot_nav.evidence import write_json

parser = argparse.ArgumentParser()
parser.add_argument('--out', required=True)
args = parser.parse_args()
root = Path(args.out)
root.mkdir(parents=True, exist_ok=False)
training = train(root/'train', steps=256, warmup=32, batch_size=16, width=16, max_steps=16)
evaluation = evaluate(root/'evaluate', root/'train/policy.npz', episodes=4, seed=123, max_steps=16)
repeat = evaluate(root/'repeat', root/'train/policy.npz', episodes=4, seed=123, max_steps=16)
try:
    evaluate(root/'missing', root/'does-not-exist.pt')
except FileNotFoundError:
    missing_refused = True
else:
    missing_refused = False
checks = dict(actual_optimizer_updates=training['updates'] == 224,
              all_episodes_classified=sum(evaluation['counts'].values()) == 4,
              deterministic_replay=evaluation['trajectories_sha256'] == repeat['trajectories_sha256'],
              missing_checkpoint_refused=missing_refused,
              missing_evaluation_not_created=not (root/'missing').exists())
write_json(root/'proof.json', dict(checks=checks, passed=all(checks.values()),
                                 scope='CPU pipeline execution only; not navigation qualification'))
if not all(checks.values()):
    raise SystemExit(1)
print(checks)
