"""Prepare all prelocked external crop attempts from the official Imperial archive."""
import gzip
import hashlib
import json
from pathlib import Path
import re
import sys
import zipfile
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.research_pnm import extract_pilot_graph
from src.experiment_data import sha256
from scripts.run_experiments import write_json,write_csv


def read_source(package,name):
    header=package.read(f'ImagesICPSC2007/{name}.mhd').decode('utf-8',errors='replace')
    dims=tuple(map(int,re.search(r'DimSize\s*=\s*(\d+)\s+(\d+)\s+(\d+)',header).groups()))
    spacing=tuple(map(float,re.search(r'ElementSize\s*=\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)',header).groups()))
    if len(set(spacing))!=1:raise ValueError('Non-isotropic source')
    raw=gzip.decompress(package.read(f'ImagesICPSC2007/{name}.raw.gz'))
    if len(raw)!=int(np.prod(dims)):raise ValueError('Unexpected source dimensions')
    volume=np.frombuffer(raw,dtype=np.uint8).reshape(dims[::-1])
    if not set(np.unique(volume)).issubset({0,1,255}):raise ValueError('Unexpected phase encoding')
    return header,volume,spacing[0]*1e-6


def main():
    torch.set_num_threads(1)
    protocol_path=ROOT/'experiments/topoflow_followup/external_protocol.json'
    protocol=json.loads(protocol_path.read_text())
    out=ROOT/'data/locked_external';out.mkdir(parents=True,exist_ok=False)
    evidence=ROOT/'experiments/topoflow_followup/external_data';evidence.mkdir(parents=True,exist_ok=False)
    archive=ROOT/'data/public/ImagesICPSC2007.zip'
    archive_sha1=hashlib.sha1(archive.read_bytes()).hexdigest()
    if archive_sha1!='d07c1d3403a19d4b39492d9671805cfd3b65c2f5':raise ValueError('Archive differs from official Box metadata')
    attempts=[];samples=[];references=[]
    with zipfile.ZipFile(archive) as package:
        for name in protocol['external_parents']:
            header,volume,voxel=read_source(package,name)
            (evidence/f'{name}_source_header.txt').write_text(header,encoding='utf-8')
            reference={axis:float(re.search(r'K'+axis+r'\s*\(mD\)\s*([\d.]+)',header).group(1)) for axis in 'xyz'}
            references.append({'parent':name,'source_shape_zyx':list(volume.shape),'voxel_size_m':voxel,
                               'published_full_volume_directional_k_mD':reference,
                               'published_network_k_mD':float(re.search(r'Network K\s*\(mD\)\s*([\d.]+)',header).group(1)),
                               'not_assigned_to_crops':True,'boundary_condition_match_independently_verified':False})
            starts=[(z,y,x) for z in [0,volume.shape[0]-128]
                    for y in [0,volume.shape[1]-128] for x in [0,volume.shape[2]-128]]
            for index,(z,y,x) in enumerate(starts):
                sid=f'external_{name}_crop_{index:02d}';binary=(volume[z:z+128,y:y+128,x:x+128]==0)
                row={'sample_id':sid,'parent_id':'external_'+name,'origin_zyx':f'{z},{y},{x}',
                     'porosity':float(binary.mean()),'status':'pending','reason':'','graph_sha256':''}
                try:
                    graph,diag=extract_pilot_graph(binary,voxel)
                    torch.save(graph,out/f'{sid}.pt');np.save(out/f'{sid}.npy',binary.astype(np.uint8))
                    row.update(status='accepted',graph_sha256=sha256(out/f'{sid}.pt'))
                    samples.append({'id':sid,'parent_id':'external_'+name,'graph':f'{sid}.pt','formation':'Sandstone',
                                    'split':'test','porosity':float(binary.mean()),'label_source':'PNM cylindrical throats; same research_pnm closure',
                                    'voxel_size_m':voxel,'shape_zyx':[128]*3,'source_origin_zyx':[z,y,x],
                                    'source_url':protocol['source_page'],'source_archive_sha256':sha256(archive),
                                    'diagnostics':diag})
                except ValueError as exc:
                    row.update(status='rejected',reason=str(exc))
                attempts.append(row);write_csv(evidence/'crop_attempts.csv',attempts)
                print(sid,row['status'],row['reason'],flush=True)
            del volume
    # Keep earlier development roles in the loader manifest; final fitting uses all four development parents.
    development=json.loads((ROOT/'data/benchmark_dataset/manifest.json').read_text())
    combined=[]
    for r in development['samples']:
        if r['formation']=='Synthetic':continue
        combined.append(dict(r,graph=str((ROOT/'data/benchmark_dataset'/r['graph']).resolve())))
    combined.extend(samples)
    write_json(out/'manifest.json',dict(schema_version=1,purpose='research',encoding=development['encoding'],samples=combined))
    write_json(evidence/'source_references.json',references)
    write_json(evidence/'preparation.json',{'external_protocol_sha256':sha256(protocol_path),'archive_sha1':archive_sha1,
               'archive_sha256':sha256(archive),'external_attempts':len(attempts),'accepted':len(samples),
               'rejected':sum(r['status']=='rejected' for r in attempts),'source_sha256':sha256(__file__),
               'extractor_sha256':sha256(ROOT/'src/research_pnm.py'),
               'phase_rule':'0 is pore; corroborate whole-volume fraction against header porosity',
               'external_scans_are_not_used_for_tuning':True})
    print('Locked external preparation complete.',flush=True)


if __name__=='__main__':main()
