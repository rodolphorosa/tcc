#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import os.path
import sys
import multiprocessing
import time
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
 
 
if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program) 
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s') 
    logging.root.setLevel(level=logging.INFO) 
 
    if len(sys.argv) < 3: 
        print ("Args: <script.py> <training_set.txt> <modelname.model>") 
        sys.exit(1) 
    
    inp, outp = sys.argv[1:3] 

    logger.info("Running %s" % ' '.join(sys.argv)) 
 
    model = Word2Vec(LineSentence(inp), size=20, window=5, min_count=10, workers=multiprocessing.cpu_count()) 
    model.init_sims(replace=True)  
    model.save(outp)
