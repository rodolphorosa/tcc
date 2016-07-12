#!/usr/bin/env python
# -*- coding: utf-8 -*-

##  Remove o campo de id da categoria pai e deixa apenas o texto da descricao da oferta 

import logging
import os.path
import sys
import re

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
 
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))
 
    if len(sys.argv) < 3:
        print ("Args: <entrada> <saida>")
        sys.exit(1)
    entrada, saida = sys.argv[1:3] 

    arquivo_entrada = open(entrada, 'r', encoding='utf-8') 
    arquivo_saida = open(saida, 'w') 

    for oferta in arquivo_entrada.readlines():
        descricao_da_oferta = oferta.split('\t')[1] 
        arquivo_saida.write(descricao_da_oferta) 
    
    arquivo_entrada.close() 
    arquivo_saida.close() 