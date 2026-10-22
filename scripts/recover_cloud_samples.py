"""Recover original demo volumes without exposing credentials or inventing provenance."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import sys

import httpx
import numpy as np
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.supabase_utils import SAMPLE_CATALOG


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'data/recovered_cloud')
    parser.add_argument('--per-category', type=int, default=2)
    args = parser.parse_args()
    if args.per_category < 1:
        parser.error('--per-category must be positive')
    load_dotenv(ROOT / '.env')
    base, key = os.getenv('SUPABASE_URL', ''), os.getenv('SUPABASE_KEY', '')
    if not base or not key:
        parser.exit(2, 'Cloud credentials are unavailable. No samples recovered.\n')
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    with httpx.Client(timeout=60, headers={'apikey': key, 'Authorization': f'Bearer {key}'}) as client:
        for name, category in SAMPLE_CATALOG.items():
            recovered = 0
            for sample in category['samples'].values():
                if recovered >= args.per_category:
                    break
                storage_path = f"{category['folder']}/{sample['file']}"
                record = {'category': name, 'storage_path': storage_path,
                          'source': 'existing project rock-samples bucket',
                          'provenance_status': 'parent scan and physical resolution unverified'}
                try:
                    response = client.get(f'{base.rstrip("/")}/storage/v1/object/rock-samples/{storage_path}')
                    response.raise_for_status()
                    volume = np.load(io.BytesIO(response.content), allow_pickle=False)
                    if volume.ndim != 3 or not np.isfinite(volume).all():
                        raise ValueError('Not a finite 3D array')
                    destination = args.output / category['folder'] / sample['file']
                    destination.parent.mkdir(exist_ok=True)
                    if destination.exists() and destination.read_bytes() != response.content:
                        raise ValueError('Existing file differs; refusing overwrite')
                    destination.write_bytes(response.content)
                    record.update(status='recovered', path=str(destination.resolve()),
                                  sha256=hashlib.sha256(response.content).hexdigest(),
                                  shape=list(volume.shape), dtype=str(volume.dtype),
                                  values=np.unique(volume).tolist(),
                                  nonzero_fraction=float(np.mean(volume != 0)))
                    recovered += 1
                    print(f'Recovered {storage_path}: {volume.shape}', flush=True)
                except Exception as exc:
                    # HTTP exception strings can contain project URLs. Keep logs credential-free.
                    record.update(status='failed', error_type=type(exc).__name__)
                    print(f'Unavailable {storage_path}: {type(exc).__name__}', flush=True)
                records.append(record)
    (args.output / 'inventory.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
    print(f'Recovered {sum(r["status"] == "recovered" for r in records)} volumes. '
          'These are recovery samples, not a validated research dataset.')
    return 0 if any(r['status'] == 'recovered' for r in records) else 2


if __name__ == '__main__':
    raise SystemExit(main())
