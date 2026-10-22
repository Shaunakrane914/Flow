"""Rebuild natural development crops from public archives; never invent rejections.

Existing data directories are preserved. Synthetic graphs are not required for
the natural-only V2 or follow-up studies.
"""
from pathlib import Path
import sys
import zipfile
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.prepare_locked_external import read_source
from scripts.audit_research_data import source_crops
from scripts.run_experiments import write_json,write_csv
from src.research_pnm import extract_pilot_graph


def main():
    torch.set_num_threads(1)
    out=ROOT/'data/benchmark_dataset';out.mkdir(parents=True,exist_ok=False)
    (out/'graphs').mkdir();(out/'chunks').mkdir()
    records=[];attempts=[]
    with zipfile.ZipFile(ROOT/'data/public/ImagesICPSC2007.zip') as package:
        for name,parent,split in [('Berea','parent_berea','validation'),('C1','parent_c1','train'),
                                  ('C2','parent_c2','train'),('Estaillades','parent_estaillades','test')]:
            if name=='Estaillades':
                _,header,_,voxel,origins,crops=source_crops(parent)
            else:
                header,volume,voxel=read_source(package,name)
                origins=[(z,y,x) for z in [0,volume.shape[0]-128]
                         for y in [0,volume.shape[1]-128] for x in [0,volume.shape[2]-128]]
                crops=[volume[z:z+128,y:y+128,x:x+128]==0 for z,y,x in origins]
            for index,(origin,binary) in enumerate(zip(origins,crops)):
                sid=f'{parent}_crop_{index:02d}'
                row={'sample_id':sid,'parent_id':parent,'origin_zyx':','.join(map(str,origin)),
                     'status':'pending','reason':''}
                try:
                    graph,diag=extract_pilot_graph(binary,voxel)
                    torch.save(graph,out/'graphs'/f'{sid}.pt');np.save(out/'chunks'/f'{sid}.npy',binary.astype(np.uint8))
                    records.append({'id':sid,'parent_id':parent,'formation':'Sandstone' if name=='Berea' else 'Carbonate',
                                    'graph':f'graphs/{sid}.pt','split':split,'porosity':round(float(binary.mean()),4),
                                    'label_source':'PNM cylindrical throats; inscribed throat diameters; direct center distances; SciPy mass balance',
                                    'voxel_size_m':voxel,'shape_zyx':[128]*3,'source_origin_zyx':list(origin),
                                    'diagnostics':diag})
                    row['status']='accepted'
                except ValueError as exc:row.update(status='rejected',reason=str(exc))
                attempts.append(row);write_csv(out/'crop_attempts.csv',attempts)
                print(sid,row['status'],flush=True)
    write_json(out/'manifest.json',{'schema_version':1,'purpose':'research',
               'encoding':{'diameter_epsilon_m':0,'volume_epsilon_m3':0,'target_epsilon_m2':0},'samples':records})


if __name__=='__main__':main()
