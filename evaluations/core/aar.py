#!/usr/bin/python
# -*- coding:utf-8 -*-
def aar(candidate, reference):
    hit = 0
    for a, b in zip(candidate, reference):
        if a == b:
            hit += 1
    return hit / len(reference)