#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@brief Processamento das features das ofertas do dataset. 
@author Rodolpho Rosa da Silva 
"""

import os
import json
import re

path = "../../dataset/files/" 

numero_pattern = re.compile("^\d+$") 
codigo_pattern = re.compile("^cod$") 

"""
@brief Normaliza o texto de uma oferta 

Dada uma string, normaliza os tokens numéricos e código.

@param s Texto de entrada 
return Texto normalizado 
"""
def normalizar(s):
	ns = [] 
	for palavra in s.strip().split(): 
		if numero_pattern.match(palavra):
			palavra = "numero" 
		if codigo_pattern.match(palavra): 
			palavra = "codigo" 
		ns.append(palavra) 
	return " ".join(ns) 

"""
@brief Obter bigramas de uma oferta 

Dada uma oferta, gera o conjunto de bigramas a partir de duas palavras consecutivas. 

@param s Texto de entrada
@return feature Bigramas gerados 
"""
def gerar_bigramas(s):
	bigramas = [] 
	palavras = s.strip().split() 
	
	for i in range(len(palavras)): 
		try: 
			bigrama = "%s+%s" %(palavras[i], palavras[i+1]) 
			bigramas.append(bigrama) 
		except: 
			pass 
	
	feature = " ".join(bigramas) 
	
	return feature 

"""
@brief Processa a feature 'primeiro token' 

Dada uma string, fez o processamento da feature representativa da primeira palavra do texto da oferta. 

@param s Texto de entrada 
@return feature Feature processada 
"""
def processar_primeiro_token(s):
	toks = s.strip().split() 
	feature = "^%s$" %(toks[0]) 
	return feature 

"""
@brief Processa a feauture da cetegoria na loja

Dada uma string, faz o processamento da feature representativa da categoria na loja. 

@param s Texto de entrada 
@return feature Feature processada 
"""
def processar_categoria_loja(s):
	toks = s.strip().split() 
	toks = [tok.lower() for tok in toks] 
	feature = "*%s*" %("+".join(toks)) 
	return feature 

"""
@brief Processa a feature 'nome da loja' 

Dada uma string, faz o processamento da feature representativa do nome da loja. 

@param s Texto de entrada 
@param feature Feature processada 
"""
def processar_nome_loja(s): 
	toks = s.strip().split() 
	toks = [tok.lower() for tok in toks] 
	feature = "**%s**" %("+".join(toks)) 
	return feature 

"""
@brief Gera o dicionário de preços médios por categoria pai

A partir de um conjunto de dados contendo os preços médios por catagoria pai, 
são gerados dois arquivos json com os nomes das categorias pai preprocessados 
e outro contendo os preços médios para cada categoria. 

@return categorias Dicionário contendo as categorias pai 
@return dicio Dicionário contendo os preços médios por categoria pai 
"""
def gerar_dicionario_precos():
	arquivo_categorias = path + "active_parent_categs.txt" 
	arquivo_precos = path + "preco_medio_cat_pai.txt" 
	dicio_precos = path + "dicio_precos.json" 
	dicio_categorias = path + "dicio_categorias.json" 

	categorias = {} 
	if not os.path.exists(dicio_categorias): 
		with open(arquivo_categorias, 'r', encoding='utf-8') as a: 
			for linha in a: 
				toks = linha.split() 
				categoria, nome = toks[0], "%s_por_preco" %("+".join(toks[1:]).lower()) 
				categorias[categoria]  = nome 
		with open(dicio_categorias, 'w') as js: 
			json.dump(categorias, js) 
	else: 
		with open(dicio_categorias) as js: 
			categorias = json.load(js) 

	dicio = {} 
	if not os.path.exists(dicio_precos): 
		with open(arquivo_precos) as a: 
			for linha in a: 
				toks = linha.split() 
				categoria, preco = toks[0], float(toks[-1]) 
				dicio[categoria] = preco 
		with open(dicio_precos, 'w') as js: 
			json.dump(dicio, js) 
	else: 
		with open(dicio_precos) as js:
			dicio = json.load(js) 

	return (categorias, dicio) 

"""
@brief Gera a feature de categoria por preço médio 

Dado um preço de uma oferta, identifica a qual categoria esta oferta pertence. 
Esta cálculo é feito por meio da distância absoluta entre o preço da oferta e o preço médio de cada categoria pai.

@param categorias Dicionário de categorias 
@param precos Dicionáro de preços
@param preco Preço da oferta

@return feature Possível categoria da oferta 
"""
def processar_categoria_preco(categorias, precos, preco):
	distancias = {}
	
	for categoria, p in precos.items():
		distancias[categoria] = abs(float(preco)-p) 
	min_dist = min(distancias.values()) 
	
	for categoria, d in distancias.items(): 
		if d == min_dist: 
			min_dist_categoria = categoria 
	
	feature = categorias[min_dist_categoria] 
	return feature 

"""
@brief Realiza o processamento das features 

A partir dos valores de entrada, realiza o processamento das features. 

@param descricao Descrição da oferta. Default False 
@param bigramas Determina se a feature 'bigramas' será utilizada. Default False 
@param primeiro_token Verifica se a feature 'primeiro token' será utilizada. Default False 
@param categoria_loja Categoria da loja. Default False 
@param preco Preço da oferta. Default False 
@param nome_loja Nome da loja. Default False  
return Features processadas 
"""
def gerar_features(descricao=None, bigramas=False, primeiro_token=False, categoria_loja=None, preco=None, nome_loja=None): 

	f1 = "" 
	f2 = "" 
	f3 = "" 
	f4 = "" 
	f5 = "" 
	f6 = "" 

	if descricao: 
		f1 = normalizar(descricao) 

	if bigramas and descricao: 
		f2 = gerar_bigramas(normalizar(descricao)) 

	if primeiro_token and descricao:
		f3 = processar_primeiro_token(descricao) 
	
	if categoria_loja: 
		f4 = processar_categoria_loja(categoria_loja) 

	categorias, precos = gerar_dicionario_precos() 
	
	if preco: 
		f5 = processar_categoria_preco(categorias, precos, preco) 

	if nome_loja: 
		f6 = processar_nome_loja(nome_loja) 

	return "%s %s %s %s %s %s" %(f1, f2, f3, f4, f5, f6) 


def teste():
	oferta = "2468	2475	camisetas melhor mae mundo 426 1t cod 759969		29.8	EMI Estampas"

	desc, catLoja, preco, nomeLoja = oferta.strip().split('\t')[2:] 

	print(desc) 
	print(catLoja) 
	print(preco) 
	print(nomeLoja) 

	features = gerar_features(descricao=desc, bigramas=True, primeiro_token=True, categoria_loja=catLoja, preco=preco, nome_loja=nomeLoja) 
	print(features) 
