#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 14:29:47 2018

@author: gonsoomoon
"""
def make_mul_index(mul, lens):
    idx_list = list()
    serial = 0 ; idx =0
    while idx < lens:
        idx_list.append(idx)
        serial += 1
        idx = serial * mul
        
    return idx_list

