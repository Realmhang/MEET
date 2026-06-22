#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import re

FILE_DIR = os.path.split(__file__)[0]
TMEXEC = os.path.join(FILE_DIR, 'TMscore')


def tm_score(mod_pdb: str, ref_pdb: str): # non-symmetric
    p = os.popen(f"{TMEXEC} {mod_pdb} {ref_pdb} -atom 'CA  '") # specify atom_opt
    text = p.read()
    p.close()
    res = re.search(r'TM-score\s*= ([0-1]\.[0-9]+)', text)
    if res is None: return None # failed

    score = float(res.group(1))
    return score


def compile_tmscore():
    if os.path.exists(TMEXEC): return # already compiled
    cpp_file = TMEXEC + '.cpp'
    print(f'Compile TMscore from {cpp_file}')
    os.system(f'g++ -static -O3 -ffast-math -lm -o {TMEXEC} {cpp_file}')