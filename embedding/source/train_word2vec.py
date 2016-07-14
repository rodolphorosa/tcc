#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
    	print ("args: <treino> <modelo> <features>") 
    	sys.exit(1) 
    
    treino, model = sys.argv[1:3] 
    features = int(sys.argv[3]) 

    logger.info("Running %s" % ' '.join(sys.argv)) 

    modelo = Word2Vec(LineSentence(treino), size=features, window=5, min_count=5, workers=cpu_count()) 
    modelo.init_sims(replace=True) 
    modelo.save(model) 