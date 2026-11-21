from __future__ import annotations

import argparse
import json
import sys
import zipfile
from .runner import evaluate, train


def main(argv=None):
    parser = argparse.ArgumentParser(description='SAC training and finite navigation evaluation')
    modes = parser.add_subparsers(dest='mode', required=True)
    training = modes.add_parser('train')
    evaluation = modes.add_parser('evaluate')
    for p in (training, evaluation):
        p.add_argument('--backend', choices=['cpu', 'ros'], default='cpu')
        p.add_argument('--out', required=True, help='New output directory; existing directories are refused')
        p.add_argument('--seed', type=int, default=7 if p is training else 10000)
        p.add_argument('--max-steps', type=int, default=250)
    training.add_argument('--steps', type=int, default=10000)
    training.add_argument('--warmup', type=int, default=500)
    training.add_argument('--batch-size', type=int, default=64)
    training.add_argument('--width', type=int, default=64)
    training.add_argument('--learning-rate', type=float, default=3e-4)
    evaluation.add_argument('--policy')
    evaluation.add_argument('--episodes', type=int, default=20)
    evaluation.add_argument('--baseline', action='store_true')
    evaluation.add_argument('--allow-backend-transfer', action='store_true')
    args = vars(parser.parse_args(argv))
    mode = args.pop('mode')
    try:
        result = train(**args) if mode == 'train' else evaluate(**args)
    except (ValueError, OSError, RuntimeError, zipfile.BadZipFile, EOFError) as e:
        print('ERROR: '+str(e), file=sys.stderr)
        return 2
    print(json.dumps({key: result[key] for key in (('actual_steps', 'updates', 'policy_sha256')
                                                 if mode == 'train' else ('counts', 'success_rate', 'backend'))}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
