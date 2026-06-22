#!/usr/bin/python
# -*- coding:utf-8 -*-
import warnings
from typing import List

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdchem, rdDetermineBonds
from biotite.structure import BondType as BT
from biotite.structure.io.pdbx import CIFCategory

from . import const
from .hierarchy import Atom, Block, Complex, BondType


def is_aa(block: Block) -> bool:
    '''return whether this block is an amino acid'''
    return is_standard_aa(block.name)
    if block.name in const.AA_GEOMETRY: # known amino acids
        return True
    # non-canonical amino acids: with N, CA, C, O (WARN: not so reliable)
    profile = { atom_name: False for atom_name in const.backbone_atoms }
    for atom in block:
        if atom.name in profile: profile[atom.name] = True
    for atom_name in profile:
        if not profile[atom_name]: return False
    return True


def is_standard_aa(abrv: str) -> bool:
    '''3-code abbreviation'''
    return abrv in const.chi_angles_atoms


def is_standard_base(abrv: str) -> bool:
    bases = { abrv: True for _, abrv in const.bases }
    return abrv in bases


def is_standard_block(abrv: str) -> bool:
    '''abbreviation in pdb (3-code for aa, DA/T/C/G for DNA, A/G/C/U for RNA)'''
    return is_standard_aa(abrv) or is_standard_base(abrv)


def bond_type_from_rdkit(bond: rdchem.Bond) -> BondType:
    '''Convert RDKit bond type to custom BondType.'''
    if bond == rdchem.BondType.SINGLE or bond.GetBondType() == rdchem.BondType.SINGLE:
        return BondType.SINGLE
    elif bond == rdchem.BondType.DOUBLE or bond.GetBondType() == rdchem.BondType.DOUBLE:
        return BondType.DOUBLE
    elif bond == rdchem.BondType.TRIPLE or bond.GetBondType() == rdchem.BondType.TRIPLE:
        return BondType.TRIPLE
    elif bond == rdchem.BondType.AROMATIC or bond.GetBondType() == rdchem.BondType.AROMATIC:
        return BondType.AROMATIC
    else:
        return BondType.NONE
    

def bond_type_from_biotite(bond):
    if bond == BT.ANY or bond == BT.COORDINATION:
        warnings.warn(f'Unknown bond {bond} is automatically set to single bond')
        bond = BT.SINGLE # for CONECT records (mostly in cyclic peptides) and coordination bonds
    if isinstance(bond, BT):
        biotite_bond = bond
    else: 
        biotite_bond = BT(bond)
    return BondType(biotite_bond.without_aromaticity().value)


def bond_type_to_rdkit(bond: BondType):
    '''Convert custom BondType to RDKit bond type'''
    if bond == BondType.SINGLE:
        return rdchem.BondType.SINGLE
    elif bond == BondType.DOUBLE:
        return rdchem.BondType.DOUBLE
    elif bond == BondType.TRIPLE:
        return rdchem.BondType.TRIPLE
    elif bond == BondType.AROMATIC:
        return rdchem.BondType.AROMATIC
    else:
        return None
    

def extract_atom_coords(block: Block, names: List[str]) -> List:
    '''extract atom coords given the names'''
    name2coords = {}
    for atom in block:
        name2coords[atom.name] = atom.get_coord()
    coords = []
    for name in names: coords.append(name2coords.get(name, None))
    return coords


def recur_index(obj, index: tuple):
    for _id in index:
        obj = obj[_id]
    return obj


def index_to_numerical_index(obj, index: tuple):
    numerical_index = []
    for i in index:
        numerical_index.append(obj.id2idx[i])
        obj = obj[i]
    return tuple(numerical_index)


def overwrite_block(cplx: Complex, index: tuple, block: Block):
    if not isinstance(index[0], int): index = index_to_numerical_index(cplx, index)
    mol = cplx[index[0]]
    mol.blocks[index[1]] = block
    # delete the original bonds related to this block
    new_bonds = []
    for bond in cplx.bonds:
        if bond.index1[0] == index[0] and bond.index1[1] == index[1]:
            continue
        elif bond.index2[0] == index[0] and bond.index2[1] == index[1]:
            continue
        else: new_bonds.append(bond)
    cplx.bonds = new_bonds


def format_standard_aa_block(block: Block) -> Block:
    assert is_standard_aa(block)
    # TODO: rename atoms according to the amino acid type
    # i.e. for atom in block: atom.name = formatted name (N, CA, C, O, CB, ...)
    return None


def renumber_res_id(res_ids: List[tuple]):
    # assume res_ids are ordered within each chain
    offset_map = {}
    new_res_ids = []
    for chain, (res_nb, insert_code) in res_ids:
        if chain not in offset_map: offset_map[chain] = 0
        if insert_code == '':
            new_res_ids.append((chain, (res_nb + offset_map[chain], '')))
        else:
            offset_map[chain] += 1
            new_res_ids.append((chain, (res_nb + offset_map[chain], '')))
    return new_res_ids


def _is_peptide_bond(cplx, start_id, end_id, bond_type):
    start_block, end_block = recur_index(cplx, start_id[:-1]), recur_index(cplx, end_id[:-1])
    start_atom, end_atom = recur_index(cplx, start_id), recur_index(cplx, end_id)

    both_aa = is_standard_aa(start_block.name) and is_standard_aa(end_block.name)
    single_bond = bond_type == BondType.SINGLE

    if not (both_aa and single_bond): return False

    # sequence distance
    seq_dist = end_block.id[0] - start_block.id[0]
    if seq_dist == 0: # same position id. different insertion code
        start_iccode = start_block.id[1]
        end_iccode = end_block.id[1]
        if len(start_iccode) > 1 or len(end_iccode) > 1: return False # weird residue id, must not be peptide
        if start_iccode == '': start_iccode = chr(ord('A') - 1)
        if end_iccode == '': end_iccode = chr(ord('A') - 1)
        seq_dist = ord(end_iccode) - ord(start_iccode)
    if abs(seq_dist) != 1: return False # not consecutive
    if seq_dist == -1: start_atom, end_atom = end_atom, start_atom  # swap

    return start_atom.name == 'C' and end_atom.name == 'N'


def _wrap_coord(val, width):
    val = str(round(val, 3))
    if len(val) > width:
        warnings.warn(f'coordinate overflow: {val}')
        val = val[:width]
        val = val.rstrip('.') # in case the dot is accidentally the last one
    return val.rjust(width)


def create_rw_mol_with_coord(lig_array) -> Chem.Mol:
    # extract ligand information
    atoms = lig_array.element.tolist()
    coordinates = list(map(tuple, lig_array.coord.tolist())) # coordinates -> list[tuple]
    mol = Chem.RWMol()
    conf = Chem.Conformer(len(atoms)) 
    rdkit_atom_mapping = {}
    for i, (atom, coord) in enumerate(zip(atoms, coordinates)):
        rd_atom = Chem.Atom(atom)
        rd_atom.SetNoImplicit(True)
        idx = mol.AddAtom(rd_atom)
        conf.SetAtomPosition(idx, coord)  
        rdkit_atom_mapping[i] = idx 

    # add conformer
    mol.AddConformer(conf,assignId=True)
    rdDetermineBonds.DetermineConnectivity(mol)
    return mol

def assign_bond(refmol: Chem.Mol, mol: Chem.Mol) -> Chem.Mol:
    #from https://code.byted.org/voyager-dev/ProtenixPostProcess/blob/dev/postprocess/utils.py
    refmol2 = Chem.Mol(refmol)
    mol2 = Chem.Mol(mol)    
    coords = np.array(mol2.GetConformer().GetPositions()).astype(np.float64)
    refmol3 = Chem.Mol(refmol)
    conf = mol2.GetConformer()

    try: # Assume that ref_mol and mol have exactly the same atom orders
        assert all([mol2.GetAtomWithIdx(i).GetSymbol() == refmol2.GetAtomWithIdx(i).GetSymbol() for i in range(refmol2.GetNumAtoms())]), 'Switch to substruct matching'
        conf.SetPositions(coords)
    except:
        # -Backup mapping strategy - to find matched substruct
        # Set bonds to SINGLE
        for b in mol2.GetBonds():
            if b.GetBondType() != Chem.BondType.SINGLE:
                b.SetBondType(Chem.BondType.SINGLE)
                b.SetIsAromatic(False)
        for b in refmol2.GetBonds():
            b.SetBondType(Chem.BondType.SINGLE)
            b.SetIsAromatic(False)
        # Set atom charges to zero;
        for a in refmol2.GetAtoms():
            a.SetFormalCharge(0)
        for a in mol2.GetAtoms():
            a.SetFormalCharge(0)
        matching = mol2.GetSubstructMatches(refmol2, uniquify=False)
        conf.SetPositions(coords[matching[0],:])

    refmol3.AddConformer(conf, assignId=True)
    Chem.SanitizeMol(refmol3)
    return refmol3

def mapping_bond_order_from_ref_smiles(mol, smiles) -> Chem.Mol:
    # get reference molecule from smiles
    params = Chem.SmilesParserParams()
    params.removeHs = True
    ref_mol = Chem.RemoveAllHs(Chem.MolFromSmiles(smiles, params))
    Chem.Kekulize(ref_mol, clearAromaticFlags=True)
    assert ref_mol.GetNumAtoms() == mol.GetNumAtoms(), f"when assigning bond order, atom number {ref_mol.GetNumAtoms()} and is {mol.GetNumAtoms()} not equal."
    prepared_ligand = assign_bond(ref_mol, mol)
    return prepared_ligand


def assign_new_chain_id(all_chain_ids):
    id_dict = { c: True for c in all_chain_ids }
    c = 'A'
    while c in id_dict:
        c = chr(ord(c) + 1)
    if c >= 'Z':
        warnings.warn(f'All characters from A to Z are occupied during chain ID assignment. Trying lower-case characters')
        c = 'a'
        while c in id_dict:
            c = chr(ord(c) + 1)
        if c >= 'z':
            warnings.warn(f'All characters from A to Z and a to z are occupied during chain ID assignment. Have to use space')
            c = ' '
    return c

BONDCAT = CIFCategory.deserialize(
'''
#
loop_
_chem_comp_bond.pdbx_ordinal 
_chem_comp_bond.comp_id 
_chem_comp_bond.atom_id_1 
_chem_comp_bond.atom_id_2 
_chem_comp_bond.value_order 
_chem_comp_bond.pdbx_aromatic_flag 
_chem_comp_bond.pdbx_stereo_config 
1   ALA N   CA  SING N ?
2   ALA CA  C   SING N ?
3   ALA CA  CB  SING N ?
4   ALA C   O   DOUB N ?
5   CYS N   CA  SING N ?
6   CYS CA  C   SING N ?
7   CYS CA  CB  SING N ?
8   CYS C   O   DOUB N ?
9   CYS CB  SG  SING N ?
10  ASP N   CA  SING N ?
11  ASP CA  C   SING N ?
12  ASP CA  CB  SING N ?
13  ASP C   O   DOUB N ?
14  ASP CB  CG  SING N ?
15  ASP CG  OD1 DOUB N ?
16  ASP CG  OD2 SING N ?
17  GLU N   CA  SING N ?
18  GLU CA  C   SING N ?
19  GLU CA  CB  SING N ?
20  GLU C   O   DOUB N ?
21  GLU CB  CG  SING N ?
22  GLU CG  CD  SING N ?
23  GLU CD  OE1 DOUB N ?
24  GLU CD  OE2 SING N ?
25  PHE N   CA  SING N ?
26  PHE CA  C   SING N ?
27  PHE CA  CB  SING N ?
28  PHE C   O   DOUB N ?
29  PHE CB  CG  SING N ?
30  PHE CG  CD1 DOUB Y ?
31  PHE CG  CD2 SING Y ?
32  PHE CD1 CE1 SING Y ?
33  PHE CD2 CE2 DOUB Y ?
34  PHE CE1 CZ  DOUB Y ?
35  PHE CE2 CZ  SING Y ?
36  GLY N   CA  SING N ?
37  GLY CA  C   SING N ?
38  GLY C   O   DOUB N ?
39  HIS N   CA  SING N ?
40  HIS CA  C   SING N ?
41  HIS CA  CB  SING N ?
42  HIS C   O   DOUB N ?
43  HIS CB  CG  SING N ?
44  HIS CG  ND1 SING Y ?
45  HIS CG  CD2 DOUB Y ?
46  HIS ND1 CE1 DOUB Y ?
47  HIS CD2 NE2 SING Y ?
48  HIS CE1 NE2 SING Y ?
49  ILE N   CA  SING N ?
50  ILE CA  C   SING N ?
51  ILE CA  CB  SING N ?
52  ILE C   O   DOUB N ?
53  ILE CB  CG1 SING N ?
54  ILE CB  CG2 SING N ?
55  ILE CG1 CD1 SING N ?
56  LYS N   CA  SING N ?
57  LYS CA  C   SING N ?
58  LYS CA  CB  SING N ?
59  LYS C   O   DOUB N ?
60  LYS CB  CG  SING N ?
61  LYS CG  CD  SING N ?
62  LYS CD  CE  SING N ?
63  LYS CE  NZ  SING N ?
64  LEU N   CA  SING N ?
65  LEU CA  C   SING N ?
66  LEU CA  CB  SING N ?
67  LEU C   O   DOUB N ?
68  LEU CB  CG  SING N ?
69  LEU CG  CD1 SING N ?
70  LEU CG  CD2 SING N ?
71  MET N   CA  SING N ?
72  MET CA  C   SING N ?
73  MET CA  CB  SING N ?
74  MET C   O   DOUB N ?
75  MET CB  CG  SING N ?
76  MET CG  SD  SING N ?
77  MET SD  CE  SING N ?
78  ASN N   CA  SING N ?
79  ASN CA  C   SING N ?
80  ASN CA  CB  SING N ?
81  ASN C   O   DOUB N ?
82  ASN CB  CG  SING N ?
83  ASN CG  OD1 DOUB N ?
84  ASN CG  ND2 SING N ?
85  PRO N   CA  SING N ?
86  PRO N   CD  SING N ?
87  PRO CA  C   SING N ?
88  PRO CA  CB  SING N ?
89  PRO C   O   DOUB N ?
90  PRO CB  CG  SING N ?
91  PRO CG  CD  SING N ?
92  GLN N   CA  SING N ?
93  GLN CA  C   SING N ?
94  GLN CA  CB  SING N ?
95  GLN C   O   DOUB N ?
96  GLN CB  CG  SING N ?
97  GLN CG  CD  SING N ?
98  GLN CD  OE1 DOUB N ?
99  GLN CD  NE2 SING N ?
100 ARG N   CA  SING N ?
101 ARG CA  C   SING N ?
102 ARG CA  CB  SING N ?
103 ARG C   O   DOUB N ?
104 ARG CB  CG  SING N ?
105 ARG CG  CD  SING N ?
106 ARG CD  NE  SING N ?
107 ARG NE  CZ  SING N ?
108 ARG NH1 CZ  SING N ?
109 ARG NH2 CZ  DOUB N ?
110 SER N   CA  SING N ?
111 SER CA  C   SING N ?
112 SER CA  CB  SING N ?
113 SER C   O   DOUB N ?
114 SER CB  OG  SING N ?
115 THR N   CA  SING N ?
116 THR CA  C   SING N ?
117 THR CA  CB  SING N ?
118 THR C   O   DOUB N ?
119 THR CB  OG1 SING N ?
120 THR CB  CG2 SING N ?
121 VAL N   CA  SING N ?
122 VAL CA  C   SING N ?
123 VAL CA  CB  SING N ?
124 VAL C   O   DOUB N ?
125 VAL CB  CG1 SING N ?
126 VAL CB  CG2 SING N ?
127 TRP N   CA  SING N ?
128 TRP CA  C   SING N ?
129 TRP CA  CB  SING N ?
130 TRP C   O   DOUB N ?
131 TRP CB  CG  SING N ?
132 TRP CG  CD1 DOUB Y ?
133 TRP CG  CD2 SING Y ?
134 TRP CD1 NE1 SING Y ?
135 TRP CD2 CE2 DOUB Y ?
136 TRP CD2 CE3 SING Y ?
137 TRP CE2 NE1 SING Y ?
138 TRP CE2 CZ2 SING Y ?
139 TRP CE3 CZ3 DOUB Y ?
140 TRP CH2 CZ2 DOUB Y ?
141 TRP CH2 CZ3 SING Y ?
142 TYR N   CA  SING N ?
143 TYR CA  C   SING N ?
144 TYR CA  CB  SING N ?
145 TYR C   O   DOUB N ?
146 TYR CB  CG  SING N ?
147 TYR CG  CD1 DOUB Y ?
148 TYR CG  CD2 SING Y ?
149 TYR CD1 CE1 SING Y ?
150 TYR CD2 CE2 DOUB Y ?
151 TYR CE1 CZ  DOUB Y ?
152 TYR CE2 CZ  SING Y ?
153 TYR OH  CZ  SING N ?
154 ACE CA  C   SING N ?
155 ACE C   O   DOUB N ?
156 NME N   CA  SING N ?
'''
)


