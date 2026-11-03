"""Atomic evidence and bounded non-pickle NPZ policy checkpoints."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import platform
import tempfile
import zipfile
import numpy as np
import torch
from .sac import SAC


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_digest():
    h = hashlib.sha256()
    for p in sorted(Path(__file__).parent.glob('*.py')):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def runtime():
    return dict(python=platform.python_version(), numpy=str(np.__version__), torch=str(torch.__version__))


def new_directory(path):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=False)
    return p


def write_json(path, data):
    path = Path(path)
    with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=path.parent, delete=False) as f:
        temporary = Path(f.name)
        try:
            json.dump(data, f, indent=2, allow_nan=False)
            f.write('\n')
        except BaseException:
            f.close()
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def save_policy(agent, path, metadata):
    path = Path(path)
    payload = dict(format_version=np.array(1, dtype=np.int64),
                   config_json=np.array(json.dumps(agent.config, allow_nan=False)),
                   metadata_json=np.array(json.dumps(metadata, allow_nan=False)),
                   updates=np.array(agent.updates, dtype=np.int64))
    payload.update({'actor.'+name: value.detach().cpu().numpy()
                    for name, value in agent.actor.state_dict().items()})
    # Inference checkpoint, intentionally not an exact training-resume snapshot.
    temporary = path.with_suffix('.tmp')
    try:
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, **payload)
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()
    return digest(path)


def load_policy(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError('checkpoint is required; evaluation never substitutes random weights')
    if path.stat().st_size > 16*1024*1024:
        raise ValueError('checkpoint exceeds16MiB')
    # Validate archive sizes and NPY headers BEFORE NumPy allocates their arrays.
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        if (len(entries) > 32 or len({x.filename for x in entries}) != len(entries)
                or sum(x.file_size for x in entries) > 16*1024*1024):
            raise ValueError('invalid checkpoint archive size/entries')
        for entry in entries:
            if not entry.filename.endswith('.npy') or '/' in entry.filename:
                raise ValueError('checkpoint contains a non-array entry')
            with archive.open(entry) as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
                elif version == (2, 0):
                    shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
                else:
                    raise ValueError('unsupported NPY version')
                count = math.prod(shape)
                if dtype.hasobject or count > 1000000 or count*dtype.itemsize > 8*1024*1024:
                    raise ValueError('unsafe or oversized checkpoint array')
                if stream.tell()+count*dtype.itemsize != entry.file_size:
                    raise ValueError('array header/payload size mismatch')
    with np.load(path, allow_pickle=False) as data:
        for name in ['format_version', 'config_json', 'metadata_json', 'updates']:
            if name not in data or data[name].shape != ():
                raise ValueError('checkpoint scalar metadata missing')
        if data['format_version'].item() != 1:
            raise ValueError('unsupported checkpoint format')
        config = json.loads(data['config_json'].item())
        metadata = json.loads(data['metadata_json'].item())
        if not isinstance(config, dict) or not isinstance(metadata, dict):
            raise ValueError('checkpoint config and metadata are required')
        if set(config) != {'state_dim', 'width', 'learning_rate', 'gamma', 'tau'}:
            raise ValueError('unsupported checkpoint configuration keys')
        if config.get('state_dim') != 26 or not isinstance(config.get('width'), int) or not 4 <= config['width'] <= 256:
            raise ValueError('unsupported checkpoint dimensions')
        if not isinstance(metadata.get('backend'), str) or not isinstance(metadata.get('source_sha256'), str):
            raise ValueError('checkpoint provenance is missing')
        agent = SAC(**config)
        expected = agent.actor.state_dict()
        if set(data.files) != {'actor.'+x for x in expected} | {'format_version', 'config_json', 'metadata_json', 'updates'}:
            raise ValueError('checkpoint tensor names do not match architecture')
        state = {}
        for name, tensor in expected.items():
            array = data['actor.'+name]
            if array.shape != tuple(tensor.shape) or array.dtype != np.float32 or not np.isfinite(array).all():
                raise ValueError('invalid checkpoint tensor: '+name)
            state[name] = torch.from_numpy(array.copy())
        agent.actor.load_state_dict(state, strict=True)
        agent.updates = int(data['updates'].item())
    agent.actor.eval()
    return agent, metadata
