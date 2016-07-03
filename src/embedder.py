from numpy import array, zeros
from gensim.models import Word2Vec
from pprint import pprint

def get_dimension(model, document):
	vector_size = model[model.vocab.keys()[0]].shape[0]
	max_length 	= max(len(sentence) for sentence in document)
	n_lines 	= len(document)
	n_columns 	= max_length * vector_size
	return (n_lines, n_columns)

def filter_sentence(model, sentence):
	words = sentence.split()
	filtered = []
	for word in words:
		if word in model.vocab:
			filtered.append(word)
	return filtered

def get_vector(model, sentence, dimension):
	if sentence == []:
		return zeros((1, dimension))
	vector = model[sentence]
	vector.resize(dimension)
	return vector

def get_matrix(model, document, dimension):
	doc_matrix 	= zeros(dimension)
	for sentence in document:
		try:
			embedding = get_vector(model, sentence, dimension[1])
			doc_matrix[document.index(sentence)] = embedding
		except:
			pass
	return array(doc_matrix)