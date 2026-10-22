"""Prepare a bounded pilot with eight development and four locked external graphs."""
from datetime import datetime,timezone
from pathlib import Path
import json
import sys
import numpy as np
import torch
from scipy.sparse import coo_matrix,diags
from scipy.sparse.linalg import splu
from torch_geometric.data import Data

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.experiment_data import load_manifest,sha256
from src.research_pnm import extract_pilot_graph,solve_permeability
from scripts.run_experiments import write_json


def build_graph(image,voxel,rng):
    _,_,net=extract_pilot_graph(image,voxel,return_network=True)
    coords=np.asarray(net['pore.coords'],dtype=float);conns=np.asarray(net['throat.conns'])
    diameter=np.asarray(net['throat.inscribed_diameter'],dtype=float)
    length=np.asarray(net['throat.direct_length'],dtype=float)
    k,diag=solve_permeability(coords,conns,diameter,length,net['pore.xmin'],net['pore.xmax'],
                              image.shape,voxel,return_fields=True)
    fields=diag.pop('fields');g=fields['throat_conductance'];s=fields['dissipation_fraction']
    internal=np.flatnonzero(fields['active_pores_mask'] & ~net['pore.xmin'] & ~net['pore.xmax'])
    a,b=conns.T;n=len(coords)
    weights=coo_matrix((np.r_[g,g],(np.r_[a,b],np.r_[b,a])),shape=(n,n)).tocsr()
    lap=diags(np.asarray(weights.sum(1)).ravel())-weights
    leverage=np.zeros(len(conns))
    if len(internal):
        lu=splu(lap[internal][:,internal].tocsc())
        remap=np.full(n,-1,dtype=int);remap[internal]=np.arange(len(internal))
        for start in range(0,len(conns),128):
            stop=min(start+128,len(conns));inc=np.zeros((len(internal),stop-start))
            for sign,nodes in [(1,a[start:stop]),(-1,b[start:stop])]:
                valid=remap[nodes]>=0;inc[remap[nodes[valid]],np.flatnonzero(valid)]+=sign
            leverage[start:stop]=g[start:stop]*np.sum(inc*lu.solve(inc),axis=0)
    assert abs(s.sum()-1)<1e-6
    assert (s>=-1e-7).all() and (leverage>=-1e-7).all() and (s+leverage<=1+1e-6).all()
    s=np.maximum(s,0);leverage=np.clip(leverage,0,1-s)
    eligible=np.flatnonzero((~net['pore.boundary'])[conns].all(1))
    top=eligible[np.argsort(s[eligible],kind='stable')[-3:]]
    random=rng.choice(np.setdiff1d(eligible,top),3,replace=False)
    selected=np.r_[top,random];mask=np.zeros(len(conns),dtype=bool);mask[selected]=True
    rates=np.array([.1,.25,.5]);change=(1-rates)**4-1
    target=-change[None,:]*s[:,None]/(1+change[None,:]*leverage[:,None])
    # Independently check the finite-change identity against fresh linear solves.
    errors=[]
    for edge in [int(top[-1]),int(random[0])]:
        changed=diameter.copy();changed[edge]*=.5
        value,_=solve_permeability(coords,conns,changed,length,net['pore.xmin'],net['pore.xmax'],image.shape,voxel)
        errors.append(abs((1-value/k)-target[edge,2]))
    assert max(errors)<1e-6
    physical=~net['pore.boundary'];pore_d=np.asarray(net['pore.equivalent_diameter'],dtype=float)
    logscale=np.log10(pore_d[physical]).mean();scale=10**logscale
    volume=np.asarray(net['pore.volume'],dtype=float)
    node=np.column_stack([np.log10(np.maximum(pore_d,.01*voxel)/scale),
                          np.log10(np.maximum(volume,1e-6*voxel**3)/scale**3),
                          coords/(np.asarray(image.shape)*voxel),net['pore.xmin'],net['pore.xmax']])
    difference=np.abs(coords[a]-coords[b]);direction=difference/np.maximum(np.linalg.norm(difference,axis=1,keepdims=True),voxel*1e-4)
    edge=np.column_stack([np.log10(diameter/scale),np.log10(length/scale),
                          np.log10((diameter/scale)**4/(length/scale)),direction])
    graph=Data(x=torch.tensor(node,dtype=torch.float32),
               edge_index=torch.tensor(np.r_[conns,conns[:,::-1]].T.copy(),dtype=torch.long),
               edge_attr=torch.tensor(np.r_[edge,edge],dtype=torch.float32),
               throat_index=torch.tensor(conns.T.copy(),dtype=torch.long),
               throat_attr=torch.tensor(edge,dtype=torch.float32),
               sensitivity=torch.tensor(s,dtype=torch.float32),
               leverage_target=torch.tensor(leverage,dtype=torch.float32),
               response_target=torch.tensor(target,dtype=torch.float32),
               selected_mask=torch.tensor(mask),internal_mask=torch.tensor(np.isin(np.arange(len(conns)),eligible)),
               high_mask=torch.tensor(np.isin(np.arange(len(conns)),top)))
    assert torch.isfinite(graph.x).all() and torch.isfinite(graph.edge_attr).all()
    return graph,{'base_k_m2':k,'relative_mass_imbalance':diag['relative_mass_imbalance'],
                  'dissipation_sum':float(s.sum()),'finite_response_identity_max_error':float(max(errors)),
                  'nodes':n,'undirected_throats':len(conns),'internal_throats':len(eligible),
                  'selected_high_throats':top.tolist(),'selected_random_throats':random.tolist()}


def main():
    torch.set_num_threads(1)
    _,records,_=load_manifest(ROOT/'data/locked_external/manifest.json')
    selected=[]
    for parent in sorted({r['parent_id'] for r in records}):
        candidates=sorted([r for r in records if r['parent_id']==parent],key=lambda r:r['id'])
        selected.extend(candidates[:1 if parent.startswith('external_') else 2])
    out=ROOT/'data/throat_pilot';out.mkdir(parents=True,exist_ok=False)
    evidence=ROOT/'experiments/topoflow_followup/throat_aware_pilot';evidence.mkdir(parents=True,exist_ok=False)
    write_json(evidence/'preparation_protocol.json',{'locked_utc':datetime.now(timezone.utc).isoformat(),
       'sample_ids':[r['id'] for r in selected],'development_graphs':8,'external_graphs':4,
       'selection':'two lexicographically first accepted crops per development scan; one per external scan',
       'target':'PNM sensitivity and exact single-throat fractional response under that closure',
       'physical_reference':'No voxel-resolved target is used in this intervention pilot',
       'input_features':'node geometry, coordinates, boundary flags, throat geometry and conductance proxy only; no pressures, flows or labels',
       'source_sha256':sha256(__file__)})
    results=[];rng=np.random.default_rng(20261022)
    for record in selected:
        directory=ROOT/'data/locked_external' if record['parent_id'].startswith('external_') else ROOT/'data/benchmark_dataset/chunks'
        image=np.load(directory/f"{record['id']}.npy").astype(bool)
        graph,diagnostics=build_graph(image,record['voxel_size_m'],rng)
        path=out/f"{record['id']}.pt";torch.save(graph,path)
        results.append({'sample_id':record['id'],'parent_id':record['parent_id'],'role':'external' if record['parent_id'].startswith('external_') else 'development',
                        'graph':str(path),'graph_sha256':sha256(path),'diagnostics':diagnostics})
        write_json(evidence/'graph_inventory.json',results)
        print('Prepared throat-aware pilot',record['id'],flush=True)
    write_json(out/'manifest.json',results)


if __name__=='__main__':main()
