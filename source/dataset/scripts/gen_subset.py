#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys 
import numpy as np 

if __name__ == "__main__":
	if len(sys.argv) < 4: 
		print("args: <entrada> <indices> <saida>") 
		sys.exit(1) 
	entrada, indices, saida = sys.argv[1:4] 

	with open(entrada, 'r', encoding="utf-8") as arq: 
		ofertas = arq.read().split("\n") 
	
	idx_subconjunto = np.loadtxt(indices, dtype=int) 

	with open(saida, 'w') as arq: 
		for idx in idx_subconjunto:
			arq.write(ofertas[idx] + "\n") 