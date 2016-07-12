# Filtra as ofertas com base no vocabulario 
# e gera dois arquivos com as ofertas filtradas e os ids das categorias pais

import sys
from gensim.models import Word2Vec
from embedder import filter_sentence

if __name__ == "__main__":

	if (len(sys.argv) < 3):
		print ("Entre com <nome_modelo> <nome_dataset> <nome_saida>")
		sys.exit(1)

	modelname = sys.argv[1] 
	inputfile = sys.argv[2] 
	outputfile = sys.argv[3] 

	model = Word2Vec.load(modelname) 
	
	input_ = open(inputfile, 'r', encoding='utf-8') 
	output = open(outputfile, 'w') 

	input_.readline() 

	sentences = []
	ids = []

	for row in input_.readlines():
		cols = row.split('\t')
		description = filter_sentence(model, cols[2]) 
		sentences.append(description) 
		ids.append(cols[0]) 

	n_sentences = len(sentences) 
	max_words = max(len(sentence) for sentence in sentences) 
	vector_dimension = model[list(model.vocab.keys())[0]].shape[0] 

	header = " ".join([str(n_sentences), str(max_words), str(vector_dimension)]) 

	output.write(header + "\n") 

	for i in range(len(sentences)):
		output.write(ids[i] + "\t" + " ".join(sentences[i]) + "\n") 

	input_.close() 
	output.close() 