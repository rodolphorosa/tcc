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
from sklearn.cross_validation import train_test_split 
from sklearn.externals import joblib 
from sklearn.metrics import accuracy_score 
from scipy.sparse import csr_matrix, lil_matrix 

# Mapeia os indices das categorias para valores inteiros 
def mapear_classes (ofertas):
	categorias = {} 
	_id = 0 

	for oferta in ofertas: 
		categoria, oferta = oferta.split("\t") 
		if not categoria in categorias.keys(): 
			categorias[categoria] = _id 
			_id += 1 

	# Gera um arquivo json contendo as classes presentes no dataset 
	caminho_das_clases = "../dataset/classes.json" 
	with open(caminho_das_clases, 'w') as jf: 
		json.dump(categorias, jf) 

	return categorias 

# Gera um vocabulario a partir do conjunto de dados 
def gerar_vocabulario (ofertas):
	vocabulario = {} 
	for oferta in ofertas: 
		for palavra in oferta.split(): 
			vocabulario[palavra] = 1 

	idx = 0
	indices = {} 
	for palavra in vocabulario.keys():
		indices[palavra] = idx
		idx += 1

	# Gera um arquivo json contendo o vocabulario gerado 
	caminho_do_vocabulario = "../dataset/vocabulario.json" 
	with open(caminho_do_vocabulario, 'w') as jf: 
		json.dump(vocabulario, jf) 

	# Gera um arquivo json contendo os indices das palavras do dicionario 
	caminho_dos_indices = "../dataset/indices.json" 
	with open(caminho_dos_indices, 'w') as jf: 
		json.dump(indices, jf) 

	return (vocabulario, indices)  

# Constroi a matriz de ocorrencias (bag of words) do conjunto de dados 
# Gera o vetor de classes e um indice invertido que, dada uma classe, 
# retorna o id da categoria pai correspondente 
def criar_dataset (ofertas, indices): 
	indice_invertido = {} 

	# v = list(vocabulario) 
	matriz = lil_matrix((len(ofertas), len(indices))) 
	y = np.zeros(len(ofertas)) 

	classes = mapear_classes(ofertas) 

	for classe in classes: 
		valor = classes[classe] 
		try: 
			indice_invertido[valor].append(classe) 
		except: 
			indice_invertido[valor] = classe 

	linha = 0 
	for oferta in ofertas: 
		# A cada dez mil ofertas lidas, exibe o passo na saida 
		if linha % 10000 == 0: print ("Process at sentence %d" %linha) 
		tokens = oferta.split() 
		classe = classes[tokens[0]] 
		matriz[linha, 0] = classe 
		y[linha] = classe 
		for palavra in tokens[1:]: 
			try: 
				coluna = indices[palavra] 
				matriz[linha, coluna] = 1 
			except: 
				pass 
		linha += 1 
	
	return (y, matriz, indice_invertido) 

# Retorna um classificador a partir de um conjunto de dados de entrada 
# Divide de forma aleatoria o conjunto em dois (treino e teste) 
# a partir de uma porcentagem dada como parametro de entrada 
# Testa o classificador e retorna a acuracia media 
def treinar_porcentagem(porcentagem=0.9): 
	ofertas = dados_de_entrada.readlines() 

	bloco_de_treino = int(len(ofertas) * porcentagem) 

	print("Generating vocabulary... ") 
	vocabulario, indices = gerar_vocabulario(ofertas) 
	
	print("Generating bow matrix... ") 
	y, X, indice_invertido = criar_dataset(ofertas, indices) 

	X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=.1, random_state=42) 

	classificador = SGDClassifier(loss="modified_huber") 

	print("Executing classifier training... ") 
	classificador = classificador.fit(X_treino, y_treino) 

	print("Executing prediction... ") 
	y_predict = classificador.predict(X_teste) 

	print("Calculation accurary... ") 
	acuracia = accuracy_score(y_teste, y_predict) 

	print("Mean accuracy: ", acuracia) 

	return classificador 

if __name__ == "__main__": 

	if len(sys.argv) > 3: 
		print("args: <dados_de_entrada> <caminho_do_classificador>") 
		sys.exit(1) 

	entrada, caminho_do_classificador = sys.argv[1:3] 
	
	dados_de_entrada = open(entrada, 'r', encoding="utf-8") 

	if not os.path.exists(caminho_do_classificador): 
		classificador = treinar_porcentagem() 
		joblib.dump(classificador, caminho_do_classificador) 
	else: 
		classificador = joblib.load(caminho_do_classificador) 
		print("Mean accuracy: ", testar()) 

	dados_de_entrada.close() 