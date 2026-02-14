# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 00:00:20 2024

@author: RHIRAsimulation
"""

import subprocess
t_all=[[0,1,2],[3,4,5],[6,7,8],[9,10]]
k_all=[0,1,2,3,4,5]

counter=0
for k in k_all:
    for t in range(len(t_all)):
        cmd = ["python", "state_perturbation_batch.py","--k_values",str(k_all[k]),"--t_values"]
        for tsub in t_all[t]:
            cmd.append(str(tsub))
            
        if counter>=16:
            print(f"k:{k}, t:{t}")    
            subprocess.run(cmd)
        counter+=1

