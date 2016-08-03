#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@brief Treinamento do classificador SGD com algoritmo Bag of Words 
@author Rodolpho Rosa da Silva 

"""
import numpy as np 
import json 
import sys 
import os.path 
import matplotlib.pyplot as plt 

from multiprocessing import cpu_count 
from sklearn.linear_model import SGDClassifier 
from sklearn.utils.class_weight import compute_class_weight 
from sklearn.cross_validation import train_test_split, StratifiedShuffleSplit 
from sklearn.externals import joblib 
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix 
from scipy.sparse import csr_matrix, lil_matrix 
from process_features import gerar_features 

"""
@brief Mapeia categorias a partir de um conjunto de dados 

Dado com conjunto de dados de entrada, realiza o mapeamento 
dos índices das categorias para valores inteiros variando entre 0 e n-1, 
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
@brief Geração de vocabulário 

A partir de um conjunto de dados, é gerado um vocabulário 
em formato de dicionário, onde a chave é a palavra 
e o valor é o índice da palavra no vocabulário. 

Ao final, o dicionário é salvo em um arquivo json. 

Caso o vocabulário já tenha sido gerado, ele é carregado 
a partir de um arquivo json. 

@param ofertas Conjunto de dados (ofertas) 
@return vocabulário Vocabulário do conjunto de dados 
""" 
def gerar_vocabulario (ofertas): 
	caminho_do_vocabulario = path + "vocabulario.json" 
	
	if not os.path.exists(caminho_do_vocabulario): 
		dicionario = {} 
		idx = 0
		for oferta in ofertas: 
			if idx % 10000 == 0: print("Process at sentence %d\n" %idx) 

			desc, cat_loja, preco, nome_loja = oferta.strip().split('\t')[2:] 
			
			features = gerar_features(descricao=desc, bigramas=True, primeiro_token=True, categoria_loja=cat_loja, preco=preco, nome_loja=nome_loja) 

			palavras = features.strip().split() 
			
			for palavra in palavras: 
				dicionario[palavra] = 1 

			idx+=1

		idx = 0 
		vocabulario = {} 
		for palavra in dicionario.keys(): 
			vocabulario[palavra] = idx 
			idx += 1 
		
		with open(caminho_do_vocabulario, 'w') as jf: 
			json.dump(vocabulario, jf) 
	else: 
		with open(caminho_do_vocabulario) as jf: 
			vocabulario = json.load(jf) 

	return vocabulario 

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
def criar_dataset (ofertas, classes, vocabulario): 
	indice_invertido = {} 

	X = lil_matrix((len(ofertas), len(vocabulario)), dtype="int") 
	y = np.zeros(len(ofertas)) 

	for classe in classes: 
		valor = classes[classe] 
		try: 
			indice_invertido[valor].append(classe) 
		except: 
			indice_invertido[valor] = classe 

	linha = 0 
	for oferta in ofertas: 
		# if linha % 10000 == 0: print ("Process at sentence #%d" %linha)
		campos = oferta.strip().split('\t') 
		try:
			classe = classes[campos[0]] 
			y[linha] = classe 
		except: 
			pass 		
		
		features = gerar_features(descricao=campos[2], primeiro_token=True) 
		
		features = gerar_features(descricao=campos[2], bigramas=True, primeiro_token=True, categoria_loja=campos[3], preco=campos[4], nome_loja=campos[5]) 
		
		tokens = features.strip().split() 
		
		for palavra in tokens: 
			try: 
				coluna = vocabulario[palavra] 
				X[linha, coluna] = 1 
			except: 
				pass 
		linha += 1 
	
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
def escrever_estatisticas (acuracia, macro): 
	with open(stats, 'a') as estatisticas: 
		estatisticas.write("%.2f\t%.2f\n" %(acuracia*100, macro*100)) 

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

	with open(treino, 'r', encoding='utf-8') as a: 
		ofertas_treino = a.readlines() 

	with open(teste, 'r', encoding='utf-8') as a: 
		ofertas_teste = a.readlines() 

	bloco_treino = len(ofertas_treino) 
	bloco_teste = len(ofertas_teste) 

	print("Running classifier with %d sentences" %(bloco_treino+bloco_teste)) 
	
	print("Generating vocabulary... ") 
	vocabulario = gerar_vocabulario(ofertas_treino + ofertas_teste) 

	print("Generating class map... ") 
	classes = mapear_classes(ofertas_treino + ofertas_teste) 

	cs = np.unique(list(classes.values())) 

	batch = bloco_treino//1000 

	if batch == 0: 	
		print("Generating bow matrix for training set... ") 
		X_treino, y_treino, indice_invertido = criar_dataset(ofertas_treino, classes, vocabulario) 

		print("Training classifier with %d sentences" %bloco_treino) 
		classificador = classificador.fit(X_treino, y_treino) 
	else: 
		print("Training classifier with %d sentences" %bloco_treino) 
		for i in range(1000): 
			print ("Train at sentence #%d" %(i*batch)) 
			k1, k2 = batch * i, batch * (i+1) 
			if k2-k1 > 1:
				X_treino, y_treino, indice_invertido = criar_dataset(ofertas_treino[k1:k2], classes, vocabulario) 
				classificador = classificador.partial_fit(X_treino, y_treino, cs) 

	print("Finished training") 

	X_treino, y_treino = [], [] 

	print("Generating bow matrix for testing set... ") 
	X_teste, y_teste, indice_invertido = criar_dataset(ofertas_teste, classes, vocabulario) 
	
	print("Testing classifier with %d sentences" %bloco_teste) 
	y_predict = classificador.predict(X_teste) 

	print("Finished testing") 

	print("Calculating accurary... ") 
	acuracia = accuracy_score(y_teste, y_predict) 
	f1_macro = f1_score(y_teste, y_predict, average="macro") 

	matriz_de_confusao = obter_matriz_confusao(y_teste, y_predict) 
	# plotar_matriz_confusao(matriz_de_confusao) 

	escrever_estatisticas(acuracia, f1_macro) 

	return classificador 

""" 
@brief Executa teste a partir de um conjunto de dados 

A partir de um conjunto de dados de entrada, executa um teste 
e retorna a acurácia média. 

@return accuracy_score Acurácia média do treino 

""" 
def testar_classificador(): 
	with open(path + "vocabulario.json") as jf: 
		vocabulario = json.load(jf) 

	with open(teste, 'r', encoding="utf-8") as a: 
		ofertas = a.readlines() 

	print("Testing classifier with %d sentences" %len(ofertas)) 

	print("Generating class map... ") 
	classes = mapear_classes(ofertas) 

	print("Generating bow matrix... ") 
	X, y, indice_invertido = criar_dataset(ofertas, classes, vocabulario) 

	print("Executing prediction... ") 
	y_pred = classificador.predict(X) 

	matriz_de_confusao = obter_matriz_confusao(y, y_pred) 
	plotar_matriz_confusao(matriz_de_confusao) 

	acuracia = accuracy_score(y, y_pred) 
	f1_macro = f1_score(y_teste, y_pred, average="macro") 

	escrever_estatisticas(acuracia, f1_macro) 

	return acuracia 

if __name__ == "__main__": 

	path = "../../dataset/files/" 

	if len(sys.argv) < 4: 
		print("args: <treino> <teste> <classificador>") 
		sys.exit(1) 
	treino, teste, caminho_do_classificador = sys.argv[1:4] 

	stats = "../stats/stats_bow.txt" 
	graph = "../graphics/" 

	if not os.path.exists(caminho_do_classificador): 
		classificador = treinar_classificador() 
		joblib.dump(classificador, caminho_do_classificador) 
	else: 
		classificador = joblib.load(caminho_do_classificador) 
		print("Mean accuracy: %.2f" %(testar_classificador()*100)) 