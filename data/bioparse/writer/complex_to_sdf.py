#!/usr/bin/python
# -*- coding:utf-8 -*-
from typing import Optional, List

from rdkit import Chem
from rdkit.Geometry import Point3D

from ..hierarchy import Complex, BondType


def complex_to_sdf(
        cplx: Complex,
        out_path: str,
        selected_chains: Optional[List[str]]=None
    ):
    '''
        Transforming a complex into a SDF file
        Args:
            cplx: Complex, the complex to written into pdb file
            selected_chains: Optional[List[str]]=None,
            out_path: str, outputing path of the SDF file
    '''
    if selected_chains is None: selected_chains = [mol.id for mol in cplx]

    fout = Chem.SDWriter(out_path)

    for i, mol in enumerate(cplx): # chain
        if mol.id not in selected_chains: continue
        name = str(mol.id)
        mw = Chem.RWMol()
        cplx_idx2rdkit_idx, coordinates = {}, []
        for j, block in enumerate(mol):
            for k, atom in enumerate(block):
                mw.AddAtom(Chem.Atom(atom.element))
                cplx_idx2rdkit_idx[(i, j, k)] = len(cplx_idx2rdkit_idx)
                coordinates.append(atom.get_coord())
        for bond in cplx.bonds:
            if bond.index1 not in cplx_idx2rdkit_idx: continue
            if bond.index2 not in cplx_idx2rdkit_idx: continue
            atom1 = cplx_idx2rdkit_idx[bond.index1]
            atom2 = cplx_idx2rdkit_idx[bond.index2]
            bond_type = bond.bond_type
            if bond_type == BondType.SINGLE:
                rdkit_bond_type = Chem.rdchem.BondType.SINGLE
            elif bond_type == BondType.DOUBLE:
                rdkit_bond_type = Chem.rdchem.BondType.DOUBLE
            elif bond_type == BondType.TRIPLE:
                rdkit_bond_type = Chem.rdchem.BondType.TRIPLE
            elif bond_type == BondType.AROMATIC:
                rdkit_bond_type = Chem.rdchem.BondType.AROMATIC
            else:
                rdkit_bond_type = Chem.rdchem.BondType.SINGLE
            mw.AddBond(atom1, atom2, rdkit_bond_type)
        mol = mw.GetMol()
        num_atoms = mol.GetNumAtoms()
        conformer = Chem.Conformer(num_atoms)
        for i, (x, y, z) in enumerate(coordinates):
            conformer.SetAtomPosition(i, Point3D(x, y, z))

        # Attach the conformer to the molecule
        mol.AddConformer(conformer)

        mol.SetProp('_Name', name)

        fout.write(mol)
    
    fout.close()