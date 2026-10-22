"""Fetch small public rock archives and prepare a clearly labelled PNM pilot.

The original five-category dataset is NOT reconstructed. Independent archive
specimens stay in one split. Crop labels are newly solved PNM approximations,
never copied from published full-volume permeability values.
"""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import zipfile

import httpx
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.research_pnm import extract_pilot_graph

IMPERIAL = 'https://www.imperial.ac.uk/earth-science/research/research-groups/pore-scale-modelling/micro-ct-images-and-networks/'
ESTAILLADES_URL = 'https://imperialcollegelondon.box.com/shared/static/2n4lb95amz7fstk19q826uzu922eb4b1.zip'
LABEL_SOURCE = 'PNM cylindrical throats; inscribed throat diameters; direct center distances; SciPy mass balance'


def file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def download(url, destination):
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + '.part')
    print(f'Downloading {destination.name}', flush=True)
    with httpx.stream('GET', url, follow_redirects=True, timeout=90) as response:
        response.raise_for_status()
        with partial.open('wb') as stream:
            for block in response.iter_bytes():
                stream.write(block)
    partial.replace(destination)


def estaillades_crops(archive, size):
    with zipfile.ZipFile(archive) as package:
        header_name = next(n for n in package.namelist() if n.endswith('.mhd'))
        header = package.read(header_name).decode('utf-8', errors='replace')
        dims = tuple(map(int, re.search(r'DimSize\s*=\s*(\d+)\s+(\d+)\s+(\d+)', header).groups()))
        spacing_um = tuple(map(float, re.search(r'ElementSpacing\s*=\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)', header).groups()))
        if len(set(spacing_um)) != 1:
            raise ValueError('Anisotropic voxels need an explicit physical model')
        raw_name = next(n for n in package.namelist() if n.endswith('.raw.gz'))
        starts = [(0, 0, 0), (0, 256, 256), (256, 0, 256), (256, 256, 0)]
        if size > 128:
            raise ValueError('Pilot crop size must be <=128')
        crops = [np.empty((size, size, size), dtype=np.uint8) for _ in starts]
        # Raw x is fastest; numpy volumes are indexed z,y,x. Do not unpack the 1 GB raw file to disk.
        with package.open(raw_name) as compressed, gzip.GzipFile(fileobj=compressed) as stream:
            for z in range(max(s[0] for s in starts) + size):
                slab = stream.read(dims[0] * dims[1])
                if len(slab) != dims[0] * dims[1]:
                    raise ValueError('Truncated Estaillades raw data')
                plane = np.frombuffer(slab, dtype=np.uint8).reshape(dims[1], dims[0])
                for crop, (z0, y0, x0) in zip(crops, starts):
                    if z0 <= z < z0 + size:
                        crop[z - z0] = plane[y0:y0 + size, x0:x0 + size]
        return header, spacing_um[0] * 1e-6, list(zip(starts, crops))


def figshare_crops(archive, name, size):
    if not shutil.which('tar'):
        raise ValueError('A 7z-capable tar executable is required for the legacy Figshare archives')
    header = subprocess.check_output(['tar', '-xOf', str(archive), f'{name}.nhdr']).decode('utf-8', errors='replace')
    dims = tuple(map(int, re.search(r'sizes:\s*(\d+)\s+(\d+)\s+(\d+)', header).groups()))
    spacing_um = float(re.search(r'space directions:\s*\(([\d.]+),', header).group(1))
    if 'uchar' not in header or 'encoding: raw' not in header:
        raise ValueError('Unexpected raw image encoding')
    raw = subprocess.check_output(['tar', '-xOf', str(archive), f'{name}.raw'])
    if len(raw) != int(np.prod(dims)):
        raise ValueError('Raw image size does not match source header')
    image = np.frombuffer(raw, dtype=np.uint8).reshape(dims[::-1])
    step = min(dims) - size
    starts = [(0, 0, 0), (0, step, step), (step, 0, step), (step, step, 0)]
    crops = [(start, image[z:z + size, y:y + size, x:x + size].copy()) for start in starts for z, y, x in [start]]
    return header, spacing_um * 1e-6, crops


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'data/pilot_public')
    parser.add_argument('--crop-size', type=int, default=128)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output exists; use a new directory to preserve extraction evidence')
    if not 16 <= args.crop_size <= 128:
        parser.error('Crop size must be between 16 and 128')
    args.output.mkdir(parents=True)
    source_dir = ROOT / 'data/public'
    sources, samples, rejected = [], [], []
    definitions = [('C1', 1189257, 'train', 'Carbonate'), ('C2', 1189258, 'train', 'Carbonate'),
                   ('Berea', 1153794, 'validation', 'Sandstone'), ('Estaillades', None, 'test', 'Carbonate')]
    for name, article_id, split, formation in definitions:
        print(f'Preparing {name} ({split})', flush=True)
        try:
            if article_id:
                response = httpx.get(f'https://api.figshare.com/v2/articles/{article_id}', timeout=60)
                response.raise_for_status()
                metadata = response.json()
                file = next(f for f in metadata['files'] if f['name'].endswith('.7z'))
                archive = source_dir / file['name']
                download(file['download_url'], archive)
                if archive.stat().st_size != file['size']:
                    raise ValueError('Downloaded archive size differs from source metadata')
                official_md5 = file.get('computed_md5') or file.get('supplied_md5')
                if official_md5 and hashlib.md5(archive.read_bytes()).hexdigest() != official_md5:
                    raise ValueError('Archive checksum differs from repository metadata')
                header, spacing, crops = figshare_crops(archive, name, args.crop_size)
                source = {'name': name, 'url': f'https://doi.org/10.6084/m9.figshare.{article_id}',
                          'license': metadata.get('license'), 'repository_metadata': metadata}
            else:
                archive = source_dir / 'estaillades/Estaillades_1000c_3p31136um.zip'
                download(ESTAILLADES_URL, archive)
                header, spacing, crops = estaillades_crops(archive, args.crop_size)
                source = {'name': name, 'url': IMPERIAL, 'download_url': ESTAILLADES_URL,
                          'license': 'Public institutional download; redistribution licence not verified',
                          'published_full_volume_dns_k_not_used_as_crop_target': True}
            source.update(archive_sha256=file_hash(archive), voxel_size_m=spacing, split=split,
                          header=header, pore_label_assumption=0,
                          note='Pilot only: source-label interpretation and PNM closure need independent validation')
            sources.append(source)
            for index, (start, raw) in enumerate(crops):
                sample_id = f'{name}_{index}'
                record = {'id': sample_id, 'source_name': name, 'split': split, 'crop_origin_zyx': list(start)}
                try:
                    if not set(np.unique(raw)).issubset({0, 1, 255}):
                        raise ValueError('Unexpected segmentation labels; do not guess a grayscale threshold')
                    image = raw == 0
                    phi = float(image.mean())
                    if not 0 < phi < .6:
                        raise ValueError('Pore-label assumption needs review; observed porosity outside pilot range')
                    print(f'  Extracting {sample_id}: porosity={phi:.4f}', flush=True)
                    graph, diagnostics = extract_pilot_graph(image, spacing)
                    graph_path = args.output / f'{sample_id}.pt'
                    torch.save(graph, graph_path)
                    np.save(args.output / f'{sample_id}.npy', image.astype(np.uint8))
                    samples.append(dict(record, graph=graph_path.name, parent_id=f'public_{name}_original_scan',
                                        formation=formation, porosity=phi, label_source=LABEL_SOURCE,
                                        voxel_size_m=spacing, shape_zyx=list(image.shape),
                                        source_url=source['url'], flow_axis='array axis 0 (source z)',
                                        diagnostics=diagnostics))
                    print(f'  Accepted {sample_id}: {graph.num_nodes} pores', flush=True)
                except Exception as exc:
                    rejected.append(dict(record, reason=f'{type(exc).__name__}: {exc}'))
                    print(f'  Rejected {sample_id}: {exc}', flush=True)
        except Exception as exc:
            rejected.append({'source_name': name, 'reason': f'{type(exc).__name__}: {exc}'})
            print(f'  Source unavailable: {name}: {type(exc).__name__}', flush=True)
        (args.output / 'sources.json').write_text(json.dumps(sources, indent=2), encoding='utf-8')
        (args.output / 'rejections.json').write_text(json.dumps(rejected, indent=2), encoding='utf-8')
    manifest = {'schema_version': 1, 'purpose': 'pilot', 'created_utc': datetime.now(timezone.utc).isoformat(),
                'encoding': {'diameter_epsilon_m': 0, 'volume_epsilon_m3': 0, 'target_epsilon_m2': 0},
                'limitations': ['Four source specimens at most; source interpretation is not independently validated.',
                                'C1 and C2 are additional public carbonates, not recovered MEC/ILS/Savonnieres.',
                                'Crop targets are PNM approximations, not direct simulations or full-volume reference values.',
                                'One held-out parent cannot support scientific confidence intervals or a CV threshold.'],
                'samples': samples}
    path = args.output / 'manifest.json'
    path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'Prepared {len(samples)} pilot graphs; {len(rejected)} rejections. Manifest: {path}')
    return 0 if all(any(s['split'] == split for s in samples) for split in ('train', 'validation', 'test')) else 2


if __name__ == '__main__':
    raise SystemExit(main())
