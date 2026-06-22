#!/usr/bin/python
# -*- coding:utf-8 -*-
from typing import Optional, List

import numpy as np
import biotite.structure as struc
from biotite.structure.io.pdbx import CIFFile, set_structure

from ..utils import _is_peptide_bond, recur_index, _wrap_coord, is_standard_block
from ..vocab import VOCAB
from ..hierarchy import Complex, Molecule, Block, Atom, BondType
from .. import const


def complex_to_mmcif(
        cplx: Complex,
        mmcif_path: str,
        selected_chains: Optional[List[str]]=None,
        title: Optional[str]=None,
        explict_bonds: Optional[List[tuple]]=None
    ):
    '''
        Args:
            cplx: Complex, the complex to written into pdb file
            pdb_path: str, output path
            selected_chains: list of chain ids to write
            title: the title of the pdb file
            explict_bonds: list of bonds to write as CONECT (each bond is represented as (id1, id2, bond_type)).
                The bond_type will be ignored as pdb do not record such information. The id1 and id2 should be
                provided as numerical ids, e.g. (0, 10, 1) means the atom at cplx[0][10][1].
    '''

    assert mmcif_path.endswith('.cif')

    atom_list = []
    resn_to_hetblocks, het_blocks = {}, {}

    mol: Molecule = None
    block: Block = None
    atom: Atom = None
    atom_number = 0 # biotite start atom number from 0, which is different from complex_to_pdb
    id2atom_number = {}
    for i, mol in enumerate(cplx): # chain
        if selected_chains is not None and mol.id not in selected_chains: continue
        for j, block in enumerate(mol):
            block_name = block.name
            if not is_standard_block(block_name): # fragments
                block_name = VOCAB.abrv_to_symbol(block_name)
            insert_code = block.id[1]
            if 'original_name' in block.properties or block.id[-1].isdigit(): # fragments has an appended insertion code with digits
                block_name = block.properties.get('original_name', None)
                if block_name is None or len(block_name) > 3:
                    if block.id[0] in resn_to_hetblocks: block_name = resn_to_hetblocks[block.id[0]]
                    else:
                        block_name = 'LI' + str(len(het_blocks)) # the original name is too long, might be smiles
                        resn_to_hetblocks[block.id[0]] = block_name
                # this block is only a fragment of the ligand, so we get rid of the insertion code
                # and put it back to a whole residue
                insert_code = ''.join([s for s in insert_code if not s.isdigit()])
            if insert_code.isdigit(): insert_code = chr(ord('A') + int(insert_code))
            # sometimes fragment will lead to insert code like A0, A1 if the residue already has one insert code.
            for k, atom in enumerate(block):
                coord = [_wrap_coord(x, 8) for x in atom.coordinate]
                atom_list.append(struc.Atom(
                    coord=np.array(coord, dtype=np.float32),
                    chain_id=mol.id,
                    res_id=block.id[0],
                    ins_code=insert_code,
                    res_name=block_name,
                    hetor=(not is_standard_block(block_name)),
                    atom_name=atom.name,
                    element=atom.element,
                    occupancy=atom.get_property('occupancy', 1.0),
                    b_factor=atom.get_property('bfactor', 0.0)
                ))
                if not is_standard_block(block_name): # hetatoms
                    if block_name not in het_blocks: het_blocks[block_name] = []
                    het_blocks[block_name].append(atom_list[-1]) # pointer
                id2atom_number[(i, j, k)] = atom_number
                atom_number += 1

    # order atom name in hetblocks
    for block_name in het_blocks:
        atoms = het_blocks[block_name]
        name_cnts = {}
        for atom in atoms:
            name = atom.atom_name
            if name in name_cnts:
                atom.atom_name = f'{name}{name_cnts[name]}'
                name_cnts[name] += 1
            else: name_cnts[name] = 1

    atom_array = struc.array(atom_list)

    # Define bond type for mmCIF
    bond_order = {
        BondType.SINGLE: struc.BondType.SINGLE,
        BondType.DOUBLE: struc.BondType.DOUBLE,
        BondType.TRIPLE: struc.BondType.TRIPLE,
    }

    recorded_bonds = {}
    atom_array.bonds = struc.BondList(atom_array.array_length())

    def add_bond(start_id, end_id, bond_type):
        # if _is_peptide_bond(cplx, start_id, end_id, bond_type): return # do not record normal peptide bond
        start_atom_number = id2atom_number[start_id]
        end_atom_number = id2atom_number[end_id]
        if ((start_atom_number, end_atom_number) in recorded_bonds) or \
           ((end_atom_number, start_atom_number)) in recorded_bonds:
            return
        # add bond
        atom_array.bonds.add_bond(start_atom_number, end_atom_number, bond_order[bond_type])
        recorded_bonds[(start_atom_number, end_atom_number)] = True

    # write non-aa bonds
    for bond in cplx.bonds:
        add_bond(bond.index1, bond.index2, bond.bond_type)

    # write explicit bonds (drop normal peptide bond)
    if explict_bonds is not None:
        for start_id, end_id, bond_type in explict_bonds:
            add_bond(start_id, end_id, bond_type)

    file = CIFFile()
    set_structure(file, atom_array, include_bonds=True)
    file.write(mmcif_path)