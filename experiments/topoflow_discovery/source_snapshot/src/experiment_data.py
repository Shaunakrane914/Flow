"""Explicit dataset provenance and grouped splits; never infer parent IDs from crops."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def decode_log(values, epsilon, name):
    if not np.isfinite(epsilon) or epsilon < 0:
        raise ValueError(f'{name}: epsilon must be finite and nonnegative')
    with np.errstate(over='ignore', invalid='ignore'):
        decoded = np.power(10., np.asarray(values, dtype=np.float64)) - epsilon
    if not np.isfinite(decoded).all() or (decoded <= 0).any():
        raise ValueError(f'{name}: nonpositive/nonfinite physical values after inverse transform')
    return decoded


def load_manifest(path, protocol='grouped'):
    path = Path(path).resolve()
    if not path.is_file():
        raise ValueError(f'Manifest not found: {path}. Recover data and create a manifest first.')
    manifest = json.loads(path.read_text(encoding='utf-8-sig'))
    if manifest.get('schema_version') != 1 or manifest.get('purpose') not in ('research', 'pilot'):
        raise ValueError('Manifest needs schema_version=1 and purpose=research or pilot')
    encoding = manifest.get('encoding', {})
    required_encoding = ('diameter_epsilon_m', 'volume_epsilon_m3', 'target_epsilon_m2')
    if any(key not in encoding for key in required_encoding):
        raise ValueError('Declare all feature/target additive epsilons, even when zero')
    samples = manifest.get('samples', [])
    if not samples:
        raise ValueError('Manifest has no samples; weights alone are not a training dataset')
    records, graphs, ids, hashes, graph_hashes, parent_splits = [], [], set(), set(), set(), {}
    for sample in samples:
        required = ('id', 'graph', 'parent_id', 'formation', 'split', 'porosity', 'label_source')
        if any(key not in sample for key in required):
            raise ValueError(f'Sample must declare {required}')
        for key in ('id', 'parent_id', 'formation', 'label_source'):
            if not isinstance(sample[key], str) or sample[key].strip().lower() in ('', 'unknown', 'todo'):
                raise ValueError(f'Invalid {key} for sample {sample.get("id")}')
        if sample['id'] in ids:
            raise ValueError(f'Duplicate sample ID: {sample["id"]}')
        ids.add(sample['id'])
        split = sample['split']
        if split not in ('train', 'validation', 'test'):
            raise ValueError(f'Invalid split: {split}')
        parent = sample['parent_id']
        if parent in parent_splits and parent_splits[parent] != split:
            raise ValueError(f'Parent leakage: {parent} occurs in different splits')
        parent_splits[parent] = split
        phi = float(sample['porosity'])
        if not np.isfinite(phi) or not 0 < phi < 1:
            raise ValueError(f'{sample["id"]}: porosity must be measured and strictly between 0 and 1')
        graph_path = (path.parent / sample['graph']).resolve()
        if not graph_path.is_file():
            raise ValueError(f'Missing graph for {sample["id"]}: {graph_path}')
        file_hash = sha256(graph_path)
        if file_hash in hashes:
            raise ValueError(f'Duplicate graph file content: {sample["id"]}')
        hashes.add(file_hash)
        # PyG graphs use pickle. Only load locally prepared/trusted files.
        graph = torch.load(graph_path, map_location='cpu', weights_only=False)
        if not isinstance(graph, Data) or graph.x.ndim != 2 or graph.x.shape[1] != 2:
            raise ValueError(f'{sample["id"]}: expected PyG Data with two log node features')
        if not torch.isfinite(graph.x).all() or graph.x.shape[0] < 2:
            raise ValueError(f'{sample["id"]}: invalid node features')
        edges = graph.edge_index
        if edges.dtype != torch.long or edges.ndim != 2 or edges.shape[0] != 2 or edges.numel() == 0:
            raise ValueError(f'{sample["id"]}: invalid edge_index')
        if edges.min() < 0 or edges.max() >= graph.x.shape[0]:
            raise ValueError(f'{sample["id"]}: edge index out of bounds')
        pairs = set(map(tuple, edges.T.tolist()))
        if any((b, a) not in pairs for a, b in pairs):
            raise ValueError(f'{sample["id"]}: graph must include reverse edges')
        local_encoding = dict(encoding, **sample.get('encoding', {}))
        diameter = decode_log(graph.x[:, 0].numpy(), float(local_encoding['diameter_epsilon_m']), 'diameter')
        volume = decode_log(graph.x[:, 1].numpy(), float(local_encoding['volume_epsilon_m3']), 'volume')
        if graph.y.numel() != 1:
            raise ValueError(f'{sample["id"]}: target must be scalar')
        k = float(decode_log(graph.y.numpy().reshape(-1), float(local_encoding['target_epsilon_m2']), 'target')[0])
        # Canonical log10(SI value), with no additive epsilon, is common to all datasets.
        canonical_x = np.column_stack((np.log10(diameter), np.log10(volume))).astype(np.float32)
        content = hashlib.sha256(canonical_x.tobytes() + np.asarray(sorted(pairs), dtype=np.int64).tobytes()).hexdigest()
        if content in graph_hashes:
            raise ValueError(f'Duplicate graph tensors: {sample["id"]}')
        graph_hashes.add(content)
        radius = diameter / 2
        cv = float(radius.std(ddof=0) / radius.mean())
        degree = np.bincount(edges[1].numpy(), minlength=len(diameter))
        # Input-only descriptors: physical porosity, size/volume statistics and network size/connectivity.
        descriptors = [phi, np.log10(diameter).mean(), np.log10(diameter).std(),
                       np.log10(volume).mean(), np.log10(volume).std(),
                       np.log10(len(diameter)), np.log10(len(pairs)), degree.mean(), degree.std(), cv]
        clean_graph = Data(x=torch.from_numpy(canonical_x), edge_index=edges.clone(),
                           y=torch.tensor([np.log10(k)], dtype=torch.float32))
        record = dict(sample, graph=str(graph_path), graph_sha256=file_hash,
                      canonical_graph_sha256=content, true_k_m2=k, true_log10_k_m2=float(np.log10(k)),
                      pore_radius_cv=cv, num_nodes=len(diameter), num_directed_edges=edges.shape[1],
                      descriptors=list(map(float, descriptors)), encoding=local_encoding)
        records.append(record)
        graphs.append(clean_graph)
    for split in ('train', 'validation', 'test'):
        if not any(r['split'] == split for r in records):
            raise ValueError(f'Empty {split} split')
    if protocol == 'formation-holdout':
        test_formations = {r['formation'] for r in records if r['split'] == 'test'}
        seen = {r['formation'] for r in records if r['split'] != 'test'}
        if test_formations & seen:
            raise ValueError('Formation holdout requires test categories absent from train AND validation')
    return manifest, records, graphs
