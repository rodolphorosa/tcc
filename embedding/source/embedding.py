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
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix 
from scipy.sparse import csr_matrix, lil_matrix 
from gensim.models import Word2Vec 

"""
@brief Mapeia categorias a partir de um conjunto de dados 

Dado com conjunto de dados de entrada, realiza o mapeamento 
dos índices das categorias para valores inteiros variando entre 0 e n, 
onde 'n' é o número de categorias distintas do dataset. 

Após o mapeamento, o dicionário gerado é salvo em um arquivo json. 

Caso o mapeamento já tenha sido feito, o mapa é carregado a partir de um arquivo json. 

@param ofertas Conjunto de dados 
@return categorias Dicionário com as classes mapeadas 
"""
def mapear_classes (ofertas): 
	caminho_das_clases = path + "classes.json" 
	categorias = {} 

	if not os.path.exists(caminho_das_clases): 	
		_id = 0 

		for oferta in ofertas: 
			categoria, oferta = oferta.split("\t") 
			if not categoria in categorias.keys(): 
				categorias[categoria] = _id 
				_id += 1 

		with open(caminho_das_clases, 'w') as jf: 
			json.dump(categorias, jf) 
	else:
		with open(caminho_das_clases) as jf: 
			categorias = json.load(jf) 

	return categorias 

"""
@brief Filtra oferta com base no modelo 

Dada uma oferta, remove as palavras que não estão no vocabulário 
do modelo word2vec dado como entrada. 

@param oferta Oferta de entrada 
@param model Modelo Word2Vec 

@return Descricao da oferta filtrada 
"""
def filtrar_oferta (oferta, modelo):
	f = [] 
	for palavra in oferta.split(): 
		if palavra in modelo.vocab: 
			f.append(palavra) 
	return " ".join(f) 

"""
@brief Calcula o teto das ofertas

Dado um conjunto de ofertas, returna o número de palavras da maior oferta. 

@param ofertas Conjunto de ofertas 
@return teto Número de palavras da maior oferta 
"""
def calcular_teto (ofertas, modelo):
	teto = max([len(oferta.split(), modelo) for oferta in ofertas]) 
	return teto 

"""
@brief Constrói a matriz bag of words 

Dado um conjunto de dados e um vocabulário, 
gera uma matriz esparsa bag of words binária de dimensão NxV onde 
N é o tamanho do dataset e V é o tamanho do vocabulário.  

@param ofertas Conjunto de dados 
@param classes Conjunto de classes (previamente) mapeadas 
@param vocabulario Vocabulário 

@return X Matriz bag of words binária 
@return y Vetor de classes 
@return indice_invertido Índice invertido indicando o id da categoria dada uma classe mapeada 
"""
def criar_dataset (ofertas, classes, usar_teto=False): 
	indice_invertido = {} 
	
	y = np.zeros(len(ofertas)) 

	for classe in classes: 
		valor = classes[classe] 
		try: 
			indice_invertido[valor].append(classe) 
		except: 
			indice_invertido[valor] = classe 

	linha = 0 
	for oferta in ofertas: 
		tokens = oferta.split() 
		try: 
			classe = classes[tokens[0]] 
			y[linha] = classe 
		except: 
			pass 
		linha += 1 

	ofertas = [filtrar_oferta(oferta, modelo) for oferta in ofertas] 

	dim = n_palavras * features 

	X = lil_matrix((len(ofertas), dim)) 

	linha = 0 
	for oferta in ofertas: 
		if oferta.split() == []: 
			pass
		else: 
			embedding = modelo[oferta.split()[0:n_palavras]] 
			embedding.resize(dim) 
			X[linha] = embedding 
		linha+=1 
	
	return (X, y, indice_invertido) 

"""
@brief Gera matriz de confusão 

A partir de um gabarito e dos resultados do teste, calcula a matriz de confusão. 

@param gabarito Gabarito do conjunto de teste 
@param predito Resultados do conjunto de teste 

@return matriz_de_confusão Matriz de confusão calculada 
"""
def obter_matriz_confusao (gabarito, predito): 
	matriz_de_confusao = confusion_matrix(gabarito, predito) 
	return matriz_de_confusao 

"""
@brief Plota a matriz de confusão 

A partir de uma matriz de confusão, plota o gráfico correspondente e o salva em um arquivo. 

@param cm Matriz de confusão 
@param title Título do gráfico 
@param cmap Paleta de cores do gráfico 
"""
def plotar_matriz_confusao (cm, title="Matriz de confusão", cmap=plt.cm.jet): 
	plt.imshow(cm, interpolation='nearest', cmap=cmap) 
	plt.title(title) 
	plt.colorbar() 
	plt.tight_layout() 
	plt.ylabel('Gabarito') 
	plt.xlabel('Predito') 
	plt.gcf().subplots_adjust(bottom=0.15) 
	plt.savefig(graph + "matriz_de_confusao.png") 

"""
@brief Escreve estatísticas da execução 

@param treino Arquivo de treino 
@param teste Arquivo de teste
@param tamanho_treino Tamanho do conjunto de treino 
@param tamaho_teste Tamanho do conjunto de teste 
@param acuracia Acurácia média do treino 
"""
def escrever_estatisticas (acuracia, f1_micro, f1_macro): 
	with open(stats, 'a') as estatisticas: 
		estatisticas.write("%d %.2f %.2f %.2f\n" %(features, (acuracia*100), (f1_micro*100), (f1_macro*100))) 

"""
@brief Executa o treino do classificador 

A partir de um conjunto de dados, executa o treinamento 
de um classificador SGD (Stochatic Gradient Descent). 

Após o treinamento, é feita uma predição sobre um conjunto de teste
e é exibido o gráfico da matriz de confusão e o cálculo da acurácia média. 

@return classificador Classificador SGD treinado
"""
def treinar_classificador (): 

	classificador = SGDClassifier(loss="modified_huber") 

	with open(treino, 'r', encoding="utf-8") as a: 
		ofertas_treino = a.readlines() 

	with open(teste, 'r', encoding="utf-8") as a: 
		ofertas_teste = a.readlines() 

	bloco_treino = len(ofertas_treino) 
	bloco_teste = len(ofertas_teste) 

	print("Running classifier with %d sentences" %(bloco_treino+bloco_teste)) 

	print("Generating class map... ") 
	classes = mapear_classes(ofertas_treino + ofertas_teste) 

	cs = np.unique(list(classes.values())) 

	batch = bloco_treino//1000 

	if batch == 0: 	
		print("Generating bow matrix for training set... ") 
		X_treino, y_treino, indice_invertido = criar_dataset(ofertas_treino, classes) 

		print("Training classifier with %d sentences" %bloco_treino) 
		classificador = classificador.fit(X_treino, y_treino) 
	else: 
		print("Training classifier with %d sentences" %bloco_treino) 
		for i in range(1000): 
			print ("Process at sentence #%d" %(i*batch)) 
			k1, k2 = batch * i, batch * (i+1) 
			if k2-k1 > 1:
				X_treino, y_treino, indice_invertido = criar_dataset(ofertas_treino[k1:k2], classes) 
				classificador = classificador.partial_fit(X_treino, y_treino, cs) 

	print("Finished training") 

	batch = bloco_teste//1000 

	if batch == 0: 
		print("Generating bow matrix for testing set... ") 
		X_teste, y_teste, indice_invertido = criar_dataset(ofertas_teste, classes) 
		
		print("Testing classifier with %d sentences" %bloco_teste) 
		y_pred = classificador.predict(X_teste) 
	else: 
		print("Testing classifier with %d sentences" %bloco_teste) 
		y_pred = [] 
		y_teste = [] 
		for i in range(1000): 
			print ("Process at sentence #%d" %(i*batch)) 
			k1, k2 = batch * i, batch * (i+1) 
			if k2-k1 > 1:
				X_teste, y, indice_invertido = criar_dataset(ofertas_teste[k1:k2], classes) 
				y_pred += list(classificador.predict(X_teste)) 
				y_teste += list(y) 

	print("Finished testing") 

	print("Computing statistics... ") 
	acuracia = accuracy_score(y_teste, y_pred) 
	f1_micro = f1_score(y_teste, y_pred, average="micro") 
	f1_macro = f1_score(y_teste, y_pred, average="macro") 

	print("Mean accuracy: %.2f" %(acuracia*100)) 

	matriz_de_confusao = obter_matriz_confusao(y_teste, y_pred) 
	plotar_matriz_confusao(matriz_de_confusao) 

	escrever_estatisticas(acuracia, f1_micro, f1_macro) 

	return classificador 

""" 
@brief Executa teste a partir de um conjunto de dados 

A partir de um conjunto de dados de entrada, executa um teste 
e retorna a acurácia média. 

@return accuracy_score Acurácia média do treino 

""" 
def testar_classificador(): 

	with open(teste, 'r', encoding="utf-8") as a: 
		ofertas = a.readlines() 

	print("Testing classifier with %d sentences" %len(ofertas)) 

	print("Generating class map... ") 
	classes = mapear_classes(ofertas) 

	batch = len(ofertas)//1000 

	if batch == 0: 
		print("Generating bow matrix... ") 
		X, y, indice_invertido = criar_dataset(ofertas, classes) 

		print("Executing prediction... ") 
		y_pred = classificador.predict(X) 
	else: 
		y_pred = [] 
		y = [] 
		for i in range(1000): 
			print ("Process at sentence #%d" %(i*batch)) 
			k1, k2 = batch * i, batch * (i+1) 
			if k2-k1 > 1:
				X_teste, y_teste, indice_invertido = criar_dataset(ofertas[k1:k2], classes) 
				y_pred += list(classificador.predict(X_teste)) 
				y += list(y_teste) 

	matriz_de_confusao = obter_matriz_confusao(y, y_pred) 
	plotar_matriz_confusao(matriz_de_confusao) 

	print("Computing statistics... ") 
	acuracia = accuracy_score(y, y_pred) 
	f1_micro = f1_score(y, y_pred, average="micro") 
	f1_macro = f1_score(y, y_pred, average="macro") 

	escrever_estatisticas(acuracia, f1_micro, f1_macro) 

	return (acuracia, f1_micro, f1_macro) 

if __name__ == "__main__": 

	path = "../../dataset/files/" 

	if len(sys.argv) < 7: 
		print("args: <treino> <teste> <modelo> <classificador> <features> <n_palavras>") 
		sys.exit(1) 
	treino, teste, path_modelo, path_classificador = sys.argv[1:5] 
	features = int(sys.argv[5]) 
	n_palavras = int(sys.argv[6]) 

	modelo = Word2Vec.load(path_modelo) 

	stats = "../stats/statistics_" + str(n_palavras) + "_palavras.txt" 
	graph = "../graphics/" 

	if not os.path.exists(path_classificador): 
		classificador = treinar_classificador()
		joblib.dump(classificador, path_classificador) 
	else:
		classificador = joblib.load(path_classificador) 
		acuracia, micro, macro = testar_classificador() 
		print("Mean accuracy: %.2f" %(acuracia*100)) 
		print("F1 score micro: %.2f" %(micro*100)) 
		print("F1 score macro: %.2f" %(macro*100)) 