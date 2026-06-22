#!/usr/bin/python
# -*- coding:utf-8 -*-
import ray

from data.bioparse import VOCAB
from data.bioparse.utils import is_aa
import utils.register as R
from .base import BaseMetric, MetricResult, EvalTask
from .core.aar import aar
from .core.gravy import gravy


@R.register('AAR')
class AAR(BaseMetric):

    @ray.remote(num_cpus=1)
    def run(self, input: EvalTask):
        
        ref = input.get_ref_cplx()
        gen = input.get_gen_cplx()

        ref_seqs, gen_seqs = '', ''
        chain_aars = {}
        for c in input.ligand_chains_ids:
            ref_pep, gen_pep = ref[c], gen[c]
            ref_seq, gen_seq = [], []
            for a, b in zip(ref_pep, gen_pep):
                if not is_aa(a): continue
                ref_seq.append(VOCAB.abrv_to_symbol(a.name))
                gen_seq.append(VOCAB.abrv_to_symbol(b.name)[0]) # in case fragments are generated which has multiple-letter symbol, which is rare
            ref_seq = ''.join(ref_seq)
            gen_seq = ''.join(gen_seq)
            chain_aars[c] = aar(gen_seq, ref_seq)
            ref_seqs += ref_seq
            gen_seqs += gen_seq
        overall_aar = aar(gen_seqs, ref_seqs)
        return MetricResult.PASSED, overall_aar, chain_aars


@R.register('GRAVY')
class GRAVY(BaseMetric):

    @ray.remote(num_cpus=1)
    def run(self, input: EvalTask):
        hydro_score = gravy(input.gen_seq)
        return MetricResult.PASSED, hydro_score, {}