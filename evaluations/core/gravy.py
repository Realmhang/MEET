#!/usr/bin/python
# -*- coding:utf-8 -*-
from Bio.SeqUtils.ProtParam import ProteinAnalysis


def gravy(seq):
    # hydrophobicity (GRAVY)
    x = ProteinAnalysis(seq)
    gravy = x.gravy()
    return gravy