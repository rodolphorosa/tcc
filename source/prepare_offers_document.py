#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os.path
import sys
import re
import time

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
 
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))
 
    # check and process input arguments
    if len(sys.argv) < 3:
        print (globals()['__doc__'] % locals())
        sys.exit(1)
    entrada, saida = sys.argv[1:3]

    arquivo_entrada = open(entrada, 'r', encoding='utf-8') 
    arquivo_saida = open(saida, 'w') 

    cabecalho = arquivo_entrada.readline() 

    numero_pattern = re.compile("^\d+$") 
    codigo_pattern = re.compile("^cod$") 

    for linha in arquivo_entrada: 
        tokens = linha.lower().split('\t') 
        oferta = tokens[2].split() 
        for i in range(len(oferta)):
            if numero_pattern.match(oferta[i]):
                oferta[i] = "numero"
            elif codigo_pattern.match(oferta[i]):
                oferta[i] = "codigo"
        arquivo_saida.write(" ".join(oferta) + "\n") 

    arquivo_entrada.close() 
    arquivo_saida.close() 