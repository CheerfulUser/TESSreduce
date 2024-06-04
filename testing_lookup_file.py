#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 10:35:32 2024

@author: zgl12
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import tessreduce as tr

obs = tr.sn_lookup('sn2019vxm', df = True)

# obs_df = pd.DataFrame(obs, columns = ['RA', 'DEC','Sector','Covers'])


tess = tr.tessreduce(obs_list=obs,plot=False,reduce=False)