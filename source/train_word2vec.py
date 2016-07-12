#!/usr/bin/env python
# -*- coding: utf-8 -*-

## Script para treinamento do modelo 

import logging
import os.path
import sys
from multiprocessing import cpu_count
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
 
 
if __name__ == '__main__':
    program = os.path.basename(sys.argv[0]) 
    logger = logging.getLogger(program) 
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s') 
    logging.root.setLevel(level=logging.INFO) 

    if len(sys.argv) < 4: 
    	print ("Args: <conjunto_de_treino> <modelo_de_saida> <dimensao_do_vetor>") 
    	sys.exit(1) 
    
    conjunto_de_treino, modelo_de_saida, dimensao = sys.argv[1:4] 

    logger.info("Running %s" % ' '.join(sys.argv)) 

    modelo = Word2Vec(LineSentence(conjunto_de_treino), size=int(dimensao), window=5, min_count=5, workers=cpu_count()) 
    modelo.init_sims(replace=True) 
    modelo.save(modelo_de_saida) 