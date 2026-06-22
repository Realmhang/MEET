import yaml
import numpy as np
import pickle as pk
from sklearn.neighbors import BallTree
import networkx as nx
import biotite.structure as struc
import biotite.structure.io as strucio
import biotite.structure.io.pdb as pdbio
import biotite.structure.io.pdbx as cifio
from biotite.structure import superimpose
import argparse
import os


def parse_yaml(yaml_path, cap_name):
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    data = data[cap_name]
    return {
        k:{_k: [a.strip() for a in _v.split(',')] for _k, _v in v.items()} for k,v in data.items()
    }
    #return {k: [a.strip() for a in v.split(',')] for k, v in data[mode].items()}

def read_structure(file):
    if file.endswith('.pdb'):
        return pdbio.PDBFile.read(file).get_structure(include_bonds=True)[0]
    elif file.endswith('.cif'):
        return cifio.get_structure(cifio.CIFFile.read(file),include_bonds=True)[0]
    else:
        raise ValueError('Input must be PDB or CIF')

def get_next_res(struct, res, direction=1):
    res_ids = np.unique(struct.res_id)
    idx = np.where(res_ids == res.res_id[0])[0][0]
    next_idx = idx + direction
    if 0 <= next_idx < len(res_ids):
        next_id = res_ids[next_idx]
        return struct[struct.res_id == next_id]
    return None

def get_atoms_coords(res, atom_names):
    coords = []
    for name in atom_names:
        atom = res[res.atom_name == name]
        if len(atom) != 1:
            raise ValueError(f'{len(atom)} atoms with name of {name} are found in residue {res.res_name[0]}')
        coords.append(atom.coord[0])
    return coords

def capping(cap_file, chain_L,ter_res,ref_coords, atom_map):
    NME,NME_1,ACE,ACE1 = ter_res
    # Read structures
    cap_struct = read_structure(cap_file)
    cap_struct.res_name = ['CAP' for _ in cap_struct.res_name]
    #remove all H atoms from cap
    cap_struct = cap_struct[cap_struct.element != 'H']

    # Cap atoms from yaml
    cap_Cter = get_atoms_coords(cap_struct, atom_map['C_ter'])
    cap_Nter = get_atoms_coords(cap_struct, atom_map['N_ter'])
    cap_coords = np.stack(cap_Cter+cap_Nter)

    # Superimpose cap onto chain L
    fitted, trans = superimpose(ref_coords,cap_coords)
    rmsd = struc.rmsd(ref_coords,fitted)
    cap_struct = trans.apply(cap_struct)
    averaged = (fitted+ref_coords)/2
    for atom_names, coord in zip(atom_map['C_ter']+atom_map['N_ter'],averaged):
        cap_struct[cap_struct.atom_name == atom_names].coord = np.expand_dims(coord, axis=0)
    i = 0
    for res_id,atom_names in zip([NME.res_id[0],NME_1.res_id[0],ACE.res_id[0],ACE1.res_id[0]],[['CA', 'N'],['C', 'O', 'CA'],['CA', 'C', 'O'],['N', 'CA']]):
        for atom_name in atom_names:
            chain_L.coord[(chain_L.res_id == res_id)&(chain_L.atom_name == atom_name)] = averaged[i]
            i += 1
    del i
    # Remove NME and ACE from chain L
    keep = ~np.isin(chain_L.res_id, [NME.res_id[0], ACE.res_id[0]])
    chain_L_mod = chain_L[keep]

    # Add cap residue with NME res_id
    cap_struct.res_id = np.full(cap_struct.array_length(), NME.res_id[0])
    cap_struct.chain_id = np.full(cap_struct.array_length(), 'L')
    # Remove last 3 atoms in C_ter and last 2 in N_ter from cap
    cap_keep = np.ones(cap_struct.array_length(), dtype=bool)
    for atom in atom_map['C_ter'][-3:] + atom_map['N_ter'][-2:]:
        idx = np.where(cap_struct.atom_name == atom)[0]
        if len(idx) > 0:
            cap_keep[idx[0]] = False
    cap_struct_mod = cap_struct[cap_keep]

    # Merge cap into chain L
    merged = struc.concatenate([chain_L_mod, cap_struct_mod])
    bond_list = merged.bonds
    # Add new bonds (connect C_ter[1] to NME-1 C, N_ter[1] to ACE+1 N)
    NME_1_C_idx = np.where((merged.res_id == NME_1.res_id[0]) & (merged.atom_name == 'C'))[0][0]
    C_ter_1_idx = np.where((merged.res_name == cap_struct[0].res_name) & (merged.atom_name == atom_map['C_ter'][1]))[0][0]
    bond_list.add_bond(NME_1_C_idx,C_ter_1_idx,bond_type=1)
    ACE1_N_idx = np.where((merged.res_id == ACE1.res_id[0]) & (merged.atom_name == 'N'))[0][0]
    N_ter_1_idx = np.where((merged.res_name == cap_struct[0].res_name) & (merged.atom_name == atom_map['N_ter'][1]))[0][0]
    bond_list.add_bond(ACE1_N_idx,N_ter_1_idx,bond_type=1)

    return rmsd,merged

def cal_descriptor(coords):
    # # centerlized the coord, and get a descriptor as distance to the center
    # center = np.mean(coords, axis=0)
    # desc = np.linalg.norm(coords - center, axis=1)

    # pairwise distance as a descriptor
    c_ter = coords[[0,3,4]]
    n_ter = coords[[5,7,9]]
    pair_dist = np.linalg.norm(c_ter[:,np.newaxis,:] - n_ter[np.newaxis,:,:], axis=2)
    desc = pair_dist.flatten()
    return desc

def buildindex(folder,cap_numbering,cap_name,output_dir):
    cap_numbering = parse_yaml(cap_numbering, cap_name)
    cap_file_names = []
    descriptors = []
    for f in os.listdir(folder):
        if f.endswith('.cif'):
            struct = read_structure(os.path.join(folder,f))
            for mode,order in cap_numbering.items():
                coords = np.array(get_atoms_coords(struct, order['C_ter']+order['N_ter']))
                desc = cal_descriptor(coords)
                cap_file_names.append(f[:-4]+f'#{mode}')
                descriptors.append(desc)
    descriptors = np.stack(descriptors)
    balltree = BallTree(descriptors, metric='euclidean')
    with open(os.path.join(output_dir,os.path.basename(folder)+'_balltree.pkl'),'wb') as f:
        pk.dump((cap_file_names,balltree),f)

def best_capping(input_file,cap_folder,cap_name,cap_numbering,balltree_file,output_dir,rmsd_cutoff=0.5,vae=False,approx_k=5):
    cap_numbering = parse_yaml(cap_numbering, cap_name)
    with open(balltree_file,'rb') as f:
        cap_file_names,balltree = pk.load(f)
    
    input_struct = read_structure(input_file)
    chain_L = input_struct[input_struct.chain_id == 'L']
    chain_R = input_struct[input_struct.chain_id == 'R']

    # Find NME, NME-1, ACE, ACE+1
    if not vae:
        NME = chain_L[chain_L.res_name == 'NME']
    else:
        NME = chain_L[chain_L.res_name == 'LI1']
    NME_1 = get_next_res(chain_L, NME, -1)
    if not vae:
        ACE = chain_L[chain_L.res_name == 'ACE']
    else:
        ACE = chain_L[chain_L.res_name == 'LI0']
    ACE1 = get_next_res(chain_L, ACE, 1)

    # Get coordinates for superposition
    # NME: CA N; NME-1: C O CA; ACE: CA C O; ACE+1: N CA
    ref_atoms = []
    if not vae:
        ref_atoms += get_atoms_coords(NME, ['CA', 'N'])
    else:
        ref_atoms += get_atoms_coords(NME, ['C', 'N'])
    ref_atoms += get_atoms_coords(NME_1, ['C', 'O', 'CA'])
    if not vae:
        ref_atoms += get_atoms_coords(ACE, ['CA', 'C', 'O'])
    else:
        ref_atoms += get_atoms_coords(ACE, ['C', 'C1', 'O'])
    ref_atoms += get_atoms_coords(ACE1, ['N', 'CA'])
    ref_coords = np.stack(ref_atoms)

    query = cal_descriptor(ref_coords)
    if approx_k !=0: #return k-best balltree index
        indices = balltree.query(np.expand_dims(query, axis=0), k=approx_k, return_distance=False)
        
    else: #return all balltree indices in rmsd_cutoff*math.sqrt(9)*2
        indices = balltree.query_radius(np.expand_dims(query, axis=0), r=rmsd_cutoff*6, return_distance=False)

    indices = list(indices[0])
    if len(indices) == 0:
        #print(f'No capping found with rmsd < {rmsd_cutoff} for {os.path.basename(input_file)}')
        return None
    rmsd_results,chain_L_results = [],[]
    for i in indices:
        cap_file,mode = cap_file_names[i].split('#')
        cap_file = os.path.join(cap_folder,cap_file+'.cif')
        rmsd,modified_chain_L = capping(cap_file, chain_L,[NME,NME_1,ACE,ACE1],ref_coords, cap_numbering[mode])
        rmsd_results.append(rmsd)
        chain_L_results.append(modified_chain_L)
    # Find the best capping
    sorted_indices = np.argsort(rmsd_results)
    for idx in sorted_indices:
        best_rmsd = rmsd_results[idx]
        best_chain_L = chain_L_results[idx]
        if best_rmsd <= rmsd_cutoff:
            clash_idx = clash_check(best_chain_L,chain_R)
            if len(clash_idx) == 0:
                print(f'Best capping found with rmsd {best_rmsd:.4f} for {os.path.basename(input_file)} with {cap_file_names[idx]}')
                strucio.save_structure(os.path.join(output_dir,os.path.basename(input_file)[:-4]+'_caplig.cif'),best_chain_L,include_bonds=True)
                return best_rmsd,cap_file_names[idx],best_chain_L,chain_R
        else:
            #print(f'No capping found with rmsd < {rmsd_cutoff} for {os.path.basename(input_file)}')
            return None

def clash_check(chainL,chainR,thres=0.8):
    vdw_radius_dict = {'C':1.7,'N':1.55,'O':1.52,'S':1.8}
    lig_bonds = chainL.bonds._bonds[:,:2].tolist()
    lig_graph = nx.Graph()
    lig_graph.add_edges_from(lig_bonds)
    chain_L_coord = chainL.coord
    cplx_coord = np.concatenate([chainL.coord,chainR.coord],axis=0)
    btree = BallTree(cplx_coord, metric='euclidean',leaf_size=10)
    contact_idx,contact_dist = btree.query_radius(chain_L_coord, r=3.6*thres, return_distance=True)
    clash_idx = []
    for i in range(len(chain_L_coord)):
        indices = contact_idx[i].tolist()
        dists = contact_dist[i].tolist()
        for j,d in zip(indices,dists):
            if j < len(chain_L_coord):
                if j == i or nx.shortest_path_length(lig_graph, i, j) <= 3:
                    continue
                if d <= thres*(vdw_radius_dict[chainL[j].element] + vdw_radius_dict[chainL[i].element]):
                    clash_idx.append((i,j))
            else:
                if d <= thres*(vdw_radius_dict[chainR[j-len(chain_L_coord)].element] + vdw_radius_dict[chainL[i].element]):
                    clash_idx.append((i,j-len(chain_L_coord)))
    return clash_idx


