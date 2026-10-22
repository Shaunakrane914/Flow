"""Verify accepted natural crops against original archive bytes and physical metadata.

Historical rejection records were manually entered; this audit does not certify
them and does not infer labels for missing/nonpercolating crops.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import zipfile

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def source_crops(parent,size=128):
    name={'parent_berea':'Berea','parent_c1':'C1','parent_c2':'C2'}.get(parent)
    if name:
        archive=ROOT/'data/public'/f'{name}.7z'
        header=subprocess.check_output(['tar','-xOf',str(archive),f'{name}.nhdr']).decode('utf-8')
        shape=tuple(map(int,re.search(r'sizes:\s*(\d+)\s+(\d+)\s+(\d+)',header).groups()))[::-1]
        spacing=float(re.search(r'space directions:\s*\(([\d.]+),',header).group(1))*1e-6
        raw=subprocess.check_output(['tar','-xOf',str(archive),f'{name}.raw'])
        volume=np.frombuffer(raw,dtype=np.uint8).reshape(shape)
        origins=[(z,y,x) for z in [0,shape[0]-size] for y in [0,shape[1]-size] for x in [0,shape[2]-size]]
        crops=[(volume[z:z+size,y:y+size,x:x+size]==0) for z,y,x in origins]
    else:
        archive=ROOT/'data/public/estaillades/Estaillades_1000c_3p31136um.zip'
        origins=[(0,0,0),(0,256,0),(0,0,256),(0,256,256),
                 (256,0,0),(256,256,0),(256,0,256),(256,256,256)]
        crops=[np.empty((size,)*3,dtype=bool) for _ in origins]
        with zipfile.ZipFile(archive) as package:
            header=package.read(next(n for n in package.namelist() if n.endswith('.mhd'))).decode('utf-8')
            shape=tuple(map(int,re.search(r'DimSize\s*=\s*(\d+)\s+(\d+)\s+(\d+)',header).groups()))[::-1]
            spacing=float(re.search(r'ElementSpacing\s*=\s*([\d.]+)',header).group(1))*1e-6
            with package.open(next(n for n in package.namelist() if n.endswith('.raw.gz'))) as compressed:
                with gzip.GzipFile(fileobj=compressed) as stream:
                    for z in range(max(s[0] for s in origins)+size):
                        slab=stream.read(shape[1]*shape[2])
                        if len(slab)!=shape[1]*shape[2]:raise ValueError('Truncated source')
                        plane=np.frombuffer(slab,dtype=np.uint8).reshape(shape[1:])
                        for crop,(z0,y0,x0) in zip(crops,origins):
                            if z0<=z<z0+size:crop[z-z0]=plane[y0:y0+size,x0:x0+size]==0
    return archive,header,shape,spacing,origins,crops


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',default='data/benchmark_dataset/manifest.json')
    parser.add_argument('--output',default='experiments/topoflow_research_v2')
    args=parser.parse_args()
    # This byte-level audit deliberately avoids loading Torch or PoreSpy.
    manifest=json.loads(Path(args.manifest).read_text(encoding='utf-8-sig'))
    output=Path(args.output)/'data_audit';output.mkdir(exist_ok=False,parents=True)
    reports=[];sources=[]
    for parent in sorted({s['parent_id'] for s in manifest['samples'] if s['formation']!='Synthetic'}):
        archive,header,shape,spacing,origins,crops=source_crops(parent)
        (output/f'{parent}_source_header.txt').write_text(header,encoding='utf-8')
        sources.append({'parent_id':parent,'archive':str(archive.relative_to(ROOT)),
                        'archive_sha256':digest(archive),'source_shape_zyx':shape,'voxel_size_m':spacing,
                        'phase_rule':'original 0 is pore; saved binary 1 is pore',
                        'physical_validation':'Phase convention inherited from source preparation; no DNS validation'})
        for record in manifest['samples']:
            if record['parent_id']!=parent:continue
            index=int(record['id'].rsplit('_',1)[1])
            chunk=Path(args.manifest).resolve().parent/'chunks'/f"{record['id']}.npy"
            saved=np.load(chunk)
            if not np.isin(saved,[0,1]).all():raise ValueError('Saved chunk is not binary')
            matches=np.array_equal(saved.astype(bool),crops[index])
            spacing_matches=bool(np.isclose(spacing,record['voxel_size_m'],rtol=1e-9,atol=0))
            if not matches or not spacing_matches:raise ValueError(f"Source mismatch: {record['id']}")
            reports.append({'sample_id':record['id'],'parent_id':parent,'origin_zyx':origins[index],
                            'shape_zyx':saved.shape,'voxel_size_m':spacing,'porosity_from_binary':float(saved.mean()),
                            'porosity_in_manifest':record['porosity'],'chunk_sha256':digest(chunk),
                            'graph_sha256':digest(Path(args.manifest).resolve().parent/record['graph']),
                            'source_crop_exact_match':matches,'source_resolution_matches':spacing_matches})
        print(f'Byte-verified accepted crops: {parent}',flush=True)
    value={'verified_natural_crops':len(reports),'natural_parents':len(sources),'sources':sources,'samples':reports,
           'limitations':['Only saved accepted crops audited; attempted/rejected counts not certified.',
                          'Historical rejection logs contain manually entered values and cannot establish extraction outcomes.',
                          'Original graph labels require fresh solver verification; see resources/extraction_summary.json.',
                          'Non-overlapping crops from one scan remain correlated and are one parent, not independent rocks.',
                          'This audit establishes byte provenance and resolution, not physical model accuracy.']}
    (output/'audit.json').write_text(json.dumps(value,indent=2),encoding='utf-8')


if __name__=='__main__':main()
