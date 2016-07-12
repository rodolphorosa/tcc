#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np 
import json 
import sys 
import os.path 
import matplotlib.pyplot as plt 

from pprint import pprint 
from multiprocessing import cpu_count 
from sklearn.linear_model import SGDClassifier 
from sklearn.utils.class_weight import compute_class_weight 
from sklearn.cross_validation import train_test_split 
from sklearn.externals import joblib 
from sklearn.metrics import accuracy_score, confusion_matrix 
from scipy.sparse import csr_matrix, lil_matrix 
from gensim.models import Word2Vec 

# Filtra uma oferta eliminando palavras que nao estejam no vocabulario gerado pelo word2vec 
def filtrar_oferta (oferta, modelo):
	filtrada = [] 
	for palavra in oferta.split(): 
		if palavra in modelo.vocab: 
			filtrada.append(palavra) 
	return filtrada 

# Gera um arquivo de configuracao para execucoes futuras 
def carregar_configuracao (ofertas, dimensao):
	configuracao = {} 
	maior_oferta = max([len(oferta) for oferta in ofertas]) 

	arquivo_de_configuracao = "../dataset/configuracao.json" 
	if not os.path.exists(arquivo_de_configuracao):
		configuracao['dimensao'] = dimensao 
		configuracao['maior_oferta'] = maior_oferta 
		with open(arquivo_de_configuracao, 'w') as jf: 
			json.dump(configuracao, jf) 
	else:
		with open(arquivo_de_configuracao) as jf: 
			configuracao = json.load(jf) 
	return configuracao 

# Mapeia os indices das classes para numeros inteiros e salva em um arquivo  
def mapear_classes (ofertas): 
	caminho_das_clases = "../dataset/iv/classes.json" 
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

def criar_dataset (modelo, ofertas, dimensao): 
	indice_invertido = {} 
	y = np.zeros(len(ofertas)) 

	classes = mapear_classes(ofertas) 

	for classe in classes: 
		valor = classes[classe] 
		try: 
			indice_invertido[valor].append(classe) 
		except: 
			indice_invertido[valor] = classe 

	# Gera o vetor de classes 
	linha = 0
	for oferta in ofertas: 
		categoria = oferta.split()[0] 
		try:
			classe = classes[categoria] 
			y[linha] = classe 
		except: 
			pass 
		linha+=1 

	# Remove as palavras que nao fazem parte do vocabulario gerado pelo word2vec 
	ofertas = [filtrar_oferta(oferta, modelo) for oferta in ofertas] 

	# Carrega um json de configuracao 
	configuracao = carregar_configuracao(ofertas, dimensao) 

	# A dimensao da matriz e' o numero de ofertas do dataset 
	# pelo produto entre o numero de features do word embeddin e o numero de palavras da maior oferta 
	dimensao_matriz = (linha, configuracao['dimensao'] * configuracao['maior_oferta']) 

	matriz = lil_matrix(dimensao_matriz) 

	print("Generating word embedding matrix... ") 
	linha = 0 
	for oferta in ofertas: 
		if linha % 10000 == 0: print ("Process at sentence #%d" %linha) 
		if oferta == []: 
			embedding = np.zeros((1, dimensao_matriz[1])) 
		else: 
			embedding = modelo[oferta] 
			# Ajusta o tamanho vetor ao tamanho do vetor da maior oferta 
			embedding.resize((1, dimensao_matriz[1])) 
			matriz[linha] = embedding 
		linha+=1 

	return (y, matriz, indice_invertido) 

# Retorna a matriz de confusao dados um gabarito e os resultados de teste 
def obter_matriz_confusao (gabarito, predito): 
	matriz_de_confusao = confusion_matrix(gabarito, predito) 
	return matriz_de_confusao 

# Plota a matriz de confusao 
def plotar_matriz_de_confusao (cm, title="Matriz de confusão", cmap=plt.cm.jet): 
	plt.imshow(cm, interpolation='nearest', cmap=cmap) 
	plt.title(title) 
	plt.colorbar() 
	plt.tight_layout() 
	plt.ylabel('Gabarito') 
	plt.xlabel('Predito') 
	plt.show() 

def treinar_classificador ( ): 
	ofertas = conjunto_de_treino.readlines() 

	print("Running classifier train with %d sentences" %len(ofertas)) 
	# Gera a matriz do conjunto de treino 
	# O arquivo de treino ja foi gerado previamente, tendo sido dado como entrada do treino do word2vec 
	y_treino, X_treino, indice_invertido = criar_dataset(modelo, ofertas, dimensao) 

	classificador = SGDClassifier(loss="modified_huber") 

	print("Executing classifier training... ") 
	classificador = classificador.fit(X_treino, y_treino) 

	# Esvazia os conjuntos de treino e teste para evitar consumo de memoria 
	X_treino = [] 
	y_treino = [] 

	# Gera a matriz do conjunto de teste 
	# Da mesma forma, o arquivo de teste ja foi gerado previamente, mas nao fui utilizado no treino do word2vec 
	ofertas = conjunto_de_teste.readlines() 
	y_teste, X_teste, indice_invertido = criar_dataset(modelo, ofertas, dimensao) 

	print("Executing prediction with %d sentences" %len(ofertas)) 
	y_pred = classificador.predict(X_teste) 

	print("Calculating accurary... ") 
	acuracia = accuracy_score(y_teste, y_pred) 

	print("Mean accuracy: ", acuracia) 

	matriz_de_confusao = obter_matriz_confusao(y_teste, y_pred) 
	plotar_matriz_de_confusao(matriz_de_confusao) 

	return classificador 

def testar_classificador ( ):
	ofertas = conjunto_de_teste.readlines() 

	print("Running classifier test with %d sentences" %len(ofertas)) 

	y_teste, X_teste, indice_invertido = criar_dataset(modelo, ofertas, dimensao) 

	print("Executing prediction with %d sentences" %len(ofertas)) 
	y_pred = classificador.predict(X_teste) 

	print("Calculating accurary... ") 
	acuracia = accuracy_score(y_teste, y_pred) 

	print("Mean accuracy: ", acuracia) 

	matriz_de_confusao = obter_matriz_confusao(y_teste, y_pred) 
	plotar_matriz_de_confusao(matriz_de_confusao) 

	return classificador 

if __name__ == "__main__": 

	if len(sys.argv) < 5: 
		print("args: <conjunto_de_treino> <conjunto_de_teste> <caminho_do_modelo> <caminho_do_classificador> <dimensao>") 
		sys.exit(1) 

	treino, teste, caminho_do_modelo, caminho_do_classificador = sys.argv[1:5] 

	try: 
		dimensao = int(sys.argv[5]) 
	except: 
		dimensao = 20 

	conjunto_de_treino = open(treino, 'r', encoding="utf-8") 
	conjunto_de_teste = open(teste, 'r', encoding="utf-8") 
	modelo = Word2Vec.load(caminho_do_modelo) 

	if not os.path.exists(caminho_do_classificador): 
		classificador = treinar_classificador() 
		joblib.dump(classificador, caminho_do_classificador) 
	else: 
		classificador = joblib.load(caminho_do_classificador) 
		testar_classificador() 

	conjunto_de_treino.close() 
	conjunto_de_teste.close() 