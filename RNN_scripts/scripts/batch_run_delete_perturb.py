# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 15:16:03 2024

@author: Hiroto
"""

import subprocess
k_all=[5]


for k in k_all:
    cmd = ["python", "delete_perturb_batch.py","--k_values",str(k)]
    print(f"k:{k}")    
    subprocess.run(cmd)