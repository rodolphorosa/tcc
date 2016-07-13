#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np 
import sys 
import os.path 
import json 

from sklearn.cross_validation import StratifiedShuffleSplit 

# Mapeia os indices das categorias para valores inteiros 
def mapear_classes (ofertas): 
	caminho_das_clases = "classes.json" 
	categorias = {} 

	# Se o mapa nao existir, cria o mapa e o salva em um arquivo json 
	if not os.path.exists(caminho_das_clases): 	
		_id = 0 

		for oferta in ofertas: 
			categoria, oferta = oferta.split("\t") 
			if not categoria in categorias.keys(): 
				categorias[categoria] = _id 
				_id += 1 

		with open(caminho_das_clases, 'w') as jf: 
			json.dump(categorias, jf) 
	# Caso contrario, carrega o mapa a partir de um arquivo  
	else:
		with open(caminho_das_clases) as jf: 
			categorias = json.load(jf) 

	return categorias 

def gerar_vetor_classes (ofertas):
	y = np.zeros(len(ofertas)) 
	classes = mapear_classes(ofertas) 

	linha = 0 
	for oferta in ofertas: 
		tokens = oferta.split() 
		try:
			classe = classes[tokens[0]] 
			y[linha] = classe 
		except: 
			pass 		
		linha += 1 

	return y 

def dividir_conjunto_de_dados (y, tamanho_teste=.1): 
	sss = StratifiedShuffleSplit(y, 1, test_size=tamanho_teste, random_state=0) 
	for train, test in sss: 
		indices_treino, indices_teste = train, test 
	return (indices_treino, indices_teste) 

if __name__ == "__main__":

	if len(sys.argv) < 5: 
		print("args: <dados_de_entrada> <treino> <teste> <porcentagem_teste>") 
		sys.exit(1) 
	entrada, treino, teste, porcentagem = sys.argv[1:5] 
	dataset = open(entrada, 'r', encoding="utf-8") 

	if not os.path.exists(treino) or not os.path.exists(teste): 
		ofertas = dataset.readlines() 
		y = gerar_vetor_classes(ofertas) 
		idx_treino, idx_teste = dividir_conjunto_de_dados(y, float(porcentagem)) 
		np.savetxt(treino, idx_treino, fmt="%i") 
		np.savetxt(teste, idx_teste, fmt="%i") 
	else: 
		print("Split already performed") 

	dataset.close() 