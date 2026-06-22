#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import json
import tempfile
from typing import List

import ray

from data.bioparse import VOCAB, const
from data.bioparse.utils import is_aa, recur_index
from data.bioparse.writer.complex_to_sdf import complex_to_sdf
from data.bioparse.writer.complex_to_pdb import complex_to_pdb
import utils.register as R
from .base import BaseMetric, MetricResult, EvalTask
from .core.rmsd import compute_rmsd
from .core.posebuster import denovo_validity
from .core.dihedrals import dihedral_distribution, jsd_angle_profile
from .core.rosetta import pr_relax, score_interface


######### RMSD #########

@R.register('RMSDComplex')
class RMSDComplex(BaseMetric):

    def __init__(self, need_align: bool=True, all_atom: bool=True, *args, **kwargs):
        self.need_align = need_align
        self.all_atom = all_atom

    @property
    def name(self):
        name = self.__class__.__name__
        name += f'_aligned({self.need_align})'
        name += f'_allatom({self.all_atom})'
        return name

    @property
    def _include_pocket(self):
        return True

    @property
    def _include_ligand(self):
        return True
    
    @ray.remote(num_cpus=1)
    def run(self, input: EvalTask):
        
        ref = input.get_ref_cplx()
        gen = input.get_gen_cplx()

        chains = {}
        if self._include_pocket: chains.update({c: True for c in input.target_chains_ids})
        if self._include_ligand: chains.update({c: True for c in input.ligand_chains_ids})

        ref_coords, gen_coords = [], []

        for res_id in input.model_block_idx:
            if res_id[0] not in chains: continue
            res_id = (res_id[0], tuple(res_id[1]))  # (chain, res_id)
            ref_block = recur_index(ref, res_id)
            if not is_aa(ref_block): continue
            try:
                gen_block = recur_index(gen, res_id)
            except Exception: continue  # generate non-natural fragments by mistake
            if self.all_atom and (ref_block.name != gen_block.name): continue
            atom_order = VOCAB.abrv_to_atoms(ref_block.name) if self.all_atom else const.backbone_atoms
            ref_atom2coord = {atom.name: atom.get_coord() for atom in ref_block}
            gen_atom2coord = {atom.name: atom.get_coord() for atom in gen_block}
            for atom_name in atom_order:
                if atom_name not in ref_atom2coord: continue
                ref_coords.append(ref_atom2coord[atom_name])
                gen_coords.append(gen_atom2coord[atom_name])
        if len(gen_coords) == 0: return MetricResult.FAILED, None, {}
        try:
            rmsd = compute_rmsd(gen_coords, ref_coords, self.need_align)
        except Exception as e:
            return MetricResult.ERROR, None, str(e)
        return MetricResult.PASSED, rmsd, {}


@R.register('RMSDLigand')
class RMSDLigand(RMSDComplex):

    @property
    def _include_pocket(self):
        return False


@R.register('RMSDPocket')
class RMSDPocket(RMSDComplex):

    @property
    def _include_ligand(self):
        return False


######### Posebuster #########
@R.register('PoseBuster')
class PoseBuster(BaseMetric):
    
    @ray.remote(num_cpus=1)
    def run(self, input: EvalTask):
        
        gen = input.get_gen_cplx()

        pocket_file = tempfile.NamedTemporaryFile(suffix='.pdb', delete=True)
        mol_file = tempfile.NamedTemporaryFile(suffix='.sdf', delete=True)

        complex_to_pdb(gen, pocket_file.name, selected_chains=input.target_chains_ids)
        complex_to_sdf(gen, mol_file.name, selected_chains=input.ligand_chains_ids)

        is_valid_flags, data = denovo_validity(pocket_file.name, mol_file.name)
        is_valid = sum(is_valid_flags) == len(is_valid_flags)
        pocket_file.close()
        mol_file.close()
        
        return MetricResult.PASSED, float(is_valid), {} # data


########## Dihedral Angles ##########
@R.register('DihedralAngles')
class DihedralAngles(BaseMetric):
    
    @property
    def _include_pocket(self):
        return True

    @property
    def _include_ligand(self):
        return True

    @property
    def calculate_on_group(self):
        return True

    @ray.remote(num_cpus=1)
    def run(self, input: EvalTask):
        # this is a group task, so run is only responsible for preparing data
        gen = input.get_gen_cplx()
        pdb_file = tempfile.NamedTemporaryFile(suffix='.pdb', delete=False)
        chains = []
        if self._include_pocket: chains += input.target_chains_ids
        if self._include_ligand: chains += input.ligand_chains_ids
        complex_to_pdb(gen, pdb_file.name, selected_chains=chains)
        return MetricResult.PASSED, pdb_file.name, {}

    def collect(self, inputs: List[EvalTask], vals: List[tuple], num_cpus: int=1):
        pdbs = [v[1] for v in vals]
        all_selected_residues = []
        for input in inputs:
            selected = []
            for res_id in input.model_block_idx:
                if (self._include_pocket and (res_id[0] in input.target_chains_ids)) or \
                   (self._include_ligand and (res_id[0] in input.ligand_chains_ids)):
                    selected.append((res_id[0], tuple(res_id[1])))
            all_selected_residues.append(selected)
        dist = dihedral_distribution(pdbs, all_selected_residues=all_selected_residues, num_cpus=num_cpus)
        for path in pdbs: os.remove(path)
        profile = jsd_angle_profile(dist)
        # scores, details
        return { 'backbone_JSD': profile['backbone_overall'], 'sidechain_JSD': profile['sidechain_overall'] }, profile


@R.register('DihedralAnglesPocket')
class DihedralAnglesPocket(DihedralAngles):

    @property
    def _include_ligand(self):
        return False
    

@R.register('DihedralAnglesLigand')
class DihedralAnglesLigand(DihedralAngles):

    @property
    def _include_pocket(self):
        return False


######### Rosetta #########

@R.register('ShapeComplementarity')
class ShapeComplementarity(BaseMetric):
    
    def _score_res_file(self, input: EvalTask):
        root_dir = os.path.dirname(input.gen_cif_path)
        filename = os.path.splitext(os.path.basename(input.gen_cif_path))[0]
        res_file = os.path.join(root_dir, filename + '_rosetta_scores.json')
        return res_file

    def _relax_file(self, input: EvalTask):
        root_dir = os.path.dirname(input.gen_cif_path)
        filename = os.path.splitext(os.path.basename(input.gen_cif_path))[0]
        res_file = os.path.join(root_dir, filename + '_rosetta_relax.pdb')
        return res_file

    def _calculate(self, input: EvalTask):
        score_res_file = self._score_res_file(input)
        # try directly load computed scores
        if os.path.exists(score_res_file):
            try:
                scores = json.load(open(score_res_file, 'r'))
                return scores, None
            except Exception: pass
        
        # relax pdbs
        relax_file = self._relax_file(input)
        if not os.path.exists(relax_file):
            gen = input.get_gen_cplx()
            pdb_file = tempfile.NamedTemporaryFile(suffix='.pdb', delete=True)
            complex_to_pdb(gen, pdb_file.name, selected_chains=input.target_chains_ids + input.ligand_chains_ids)
            try: pr_relax(pdb_file.name, relax_file)
            except Exception as e: return None, e
        
        # calculate scores
        try: scores, _, _ = score_interface(relax_file, input.target_chains_ids, input.ligand_chains_ids)
        except Exception as e: return None, e

        with open(score_res_file, 'w') as fout: json.dump(scores, fout, indent=2)

        return scores, None

    @ray.remote(num_cpus=1)
    def run(self, input: EvalTask):
        scores, details = self._calculate(input)
        if scores is None: return MetricResult.ERROR, None, details
        return MetricResult.PASSED, scores['interface_sc'], {} # data


@R.register('dG')
class dG(ShapeComplementarity):
    @ray.remote(num_cpus=1)
    def run(self, input: EvalTask):
        scores, details = self._calculate(input)
        if scores is None: return MetricResult.ERROR, None, details
        return MetricResult.PASSED, scores['interface_dG'], {} # data
    


@R.register('dG_dSASA')
class dG_dSASA(ShapeComplementarity):
    @ray.remote(num_cpus=1)
    def run(self, input: EvalTask):
        scores, details = self._calculate(input)
        if scores is None: return MetricResult.ERROR, None, details
        return MetricResult.PASSED, scores['interface_dG_SASA_ratio'], {} # data
