#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import json
import sys
import os.path

from random import shuffle
from multiprocessing import cpu_count
from sklearn.linear_model import SGDClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.externals import joblib

def gerar_vocabulario (ofertas):
	vocabulario = {}
	for oferta in ofertas:
		for palavra in oferta:
			vocabulario[palavra] = 1

	caminho_do_vocabulario = "../dataset/vocabulario.txt" 
	arquivo = open(caminho_do_vocabulario, 'w') 
	json.dump(vocabulario, arquivo)  

	return vocabulario

def gerar_vetor_oferta (oferta, vocabulario):
	vetor = np.zeros(len(vocabulario)) 
	lista_vocabulario = list(vocabulario) 

	for palavra in oferta: 
		indice = lista_vocabulario.index(palavra) 
		vetor[indice] += 1 

	return vetor 

def gerar_matriz_documento (ofertas, vocabulario):
	matriz = np.zeros((len(ofertas), len(vocabulario))) 
	for i in range(len(ofertas)): 
		matriz[i] = gerar_vetor_oferta(ofertas[i], vocabulario) 
	return matriz 

def treinar():

	ofertas = [] 
	categorias = [] 

	rows = conjunto_de_dados.readlines() 
	shuffle(rows) 

	for row in rows:
		colunas = row.split("\t") 
		oferta 	= colunas[1].split() 
		ofertas.append(oferta) 
		categorias.append(colunas[0]) 

	classes = np.unique(categorias) 
	cw = compute_class_weight("auto", classes, categorias) 

	dicionario_de_classes = {} 
	for i in range(len(classes)): 
		dicionario_de_classes[classes[i]] = cw[i] 

	classificador = SGDClassifier(loss="huber", average=True, n_jobs=cpu_count(), shuffle=False, class_weight=dicionario_de_classes) 
	# classificador = SGDClassifier(loss="huber", average=True, n_jobs=cpu_count(), shuffle=False) 

	vocabulario = gerar_vocabulario(ofertas) 
	print("Vocabulary size ", len(vocabulario)) 
	print("Generating bow matrix...") 
	X = gerar_matriz_documento(ofertas, vocabulario) 
	y = categorias 

	print("Training classifier with matrix of dimension ", X.shape) 

	# classificador = classificador.fit(X, y) 

	bloco = len(ofertas)//1000 
	for i in range(1000): 
		print ("Process at iteration ", i) 
		k1, k2 = bloco*i, bloco*(i+1) 
		if k2-k1 > 1: 
			bloco_X, bloco_y = X[k1:k2], y[k1:k2] 
			classificador = classificador.partial_fit(X, y, classes) 

	return classificador 

def testar():
	ofertas = [] 
	categorias = [] 

	rows = conjunto_de_dados.readlines() 
	shuffle(rows) 

	for row in rows:
		colunas = row.split("\t") 
		oferta 	= colunas[1].split() 
		ofertas.append(oferta) 
		categorias.append(colunas[0]) 

	with open("../dataset/vocabulario.txt") as arquivo_json:
		vocabulario = json.load(arquivo_json) 

	X = gerar_matriz_documento(ofertas, vocabulario) 
	y = categorias 

	return classificador.score(X, y) 

if __name__ == "__main__":

	if len(sys.argv) > 3: 
		print("args: <conjunto_de_dados.txt> <caminho_do_classificador>") 
		sys.exit(1) 

	input_, output = sys.argv[1:3] 

	conjunto_de_dados = open(input_, 'r', encoding="utf-8") 

	if not os.path.exists(output): 
		classificador = treinar() 
		joblib.dump(classificador, output) 
	else: 
		classificador = joblib.load(output) 
		print("Acuraria media: ", testar()) 