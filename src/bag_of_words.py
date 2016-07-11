#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np 
import json 
import sys 
import os.path 

from pprint import pprint 
from random import shuffle 
from multiprocessing import cpu_count 
from sklearn.linear_model import SGDClassifier 
from sklearn.utils.class_weight import compute_class_weight 
from sklearn.externals import joblib 
from sklearn.metrics import accuracy_score 
from scipy.sparse import csr_matrix, lil_matrix 

def gerar_vocabulario (ofertas):
	vocabulario = {} 
	for oferta in ofertas: 
		for palavra in oferta: 
			vocabulario[palavra] = 1 

	caminho_do_vocabulario = "../dataset/vocabulario.txt" 
	with open(caminho_do_vocabulario, 'w') as jf: 
		json.dump(vocabulario, jf) 

	return vocabulario 

def gerar_vetor_oferta (oferta, vocabulario): 
	vetor = np.zeros(len(vocabulario)) 
	lista_vocabulario = list(vocabulario) 

	for palavra in oferta: 
		try:
			indice = lista_vocabulario.index(palavra) 
			vetor[indice] += 1 
		except:
			pass 
	return vetor 

def gerar_matrix_ofertas (ofertas, vocabulario): 
	v = list(vocabulario) 

	X = lil_matrix((len(ofertas), len(vocabulario))) 
	linha = 0
	for oferta in ofertas:
		for palavra in oferta:
			try:
				coluna = v.index(palavra) 
				X[linha, coluna] = 1
			except:
				pass
	return X

def treinar(): 
	categorias 	= {} 
	ofertas 	= [] 

	rows = dados_de_entrada.readlines() 
	y = np.zeros(len(rows)) 

	_id = 0 

	print("Creating class vector...") 

	for row in range(len(rows)): 
		categoria, oferta = rows[row].split("\t") 
		ofertas.append(oferta.split()) 
		try: 
			y[row] = categorias[categoria] 
		except: 
			categorias[categoria] = _id 
			y[row] = categorias[categoria] 
			_id += 1 

	print("Computing class weight...") 
	classes = np.unique(y) 
	cw = compute_class_weight("balanced", classes, y) 

	pesos = {}
	for i in range(len(classes)): 
		pesos[classes[i]] = cw[i] 

	# classificador = SGDClassifier(loss="log") 
	classificador = SGDClassifier(loss="huber", average=True, n_jobs=cpu_count(), shuffle=True, class_weight=pesos) 

	print("Creating vocabulary...") 
	vocabulario = gerar_vocabulario(ofertas) 

	print("Starting classifier train...") 
	bloco = len(ofertas)//1000 
	for i in range(1000): 
		print ("Process at iteration ", i) 
		k1, k2 = bloco * i, bloco * (i+1) 
		if k2-k1 > 1: 
			bloco_X, bloco_y = ofertas[k1:k2], y[k1:k2] 
			X = gerar_matrix_ofertas(bloco_X, vocabulario) 
			classificador = classificador.partial_fit(X, bloco_y, classes) 

	return classificador 

def testar(): 
	categorias 	= {} 
	ofertas = [] 

	rows = dados_de_entrada.readlines() 
	y = np.zeros(len(rows)) 

	_id = 0 

	for row in range(len(rows)): 
		categoria, oferta = rows[row].split("\t") 
		ofertas.append(oferta.split()) 
		try: 
			y[row] = categorias[categoria] 
		except: 
			categorias[categoria] = _id 
			y[row] = categorias[categoria] 
			_id += 1 

	with open("../dataset/vocabulario.txt") as jf: 
		vocabulario = json.load(jf) 

	X = gerar_matrix_ofertas(ofertas, vocabulario) 

	y_test = classificador.predict(X) 
	return accuracy_score(y, y_test) 

if __name__ == "__main__":

	if len(sys.argv) > 3: 
		print("args: <dados_de_entrada> <caminho_do_classificador>") 
		sys.exit(1) 

	entrada, caminho_do_classificador = sys.argv[1:3] 
	
	dados_de_entrada 	= open(entrada, 'r', encoding="utf-8") 
	arquivo_categorias 	= open("../dataset/categorias_pai.txt", 'r', encoding="utf-8") 

	if not os.path.exists(caminho_do_classificador): 
		classificador = treinar() 
		joblib.dump(classificador, caminho_do_classificador) 
	else: 
		classificador = joblib.load(caminho_do_classificador) 
		print("Acuraria media: ", testar()) 

	dados_de_entrada.close() 
	arquivo_categorias.close() 