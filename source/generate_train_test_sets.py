import sys 
from sklearn.cross_validation import train_test_split 

if __name__ == "__main__": 

	if len(sys.argv) > 5: 
		print("args: <conjunto_de_dados> <conjunto_de_treino> <conjunto_de_teste> <porcentagem_de_teste>") 
		sys.exit(1) 

	entrada, treino, teste, porcentagem = sys.argv[1:5] 

	conjunto_de_dados = open(entrada, 'r', encoding="utf-8") 

	conjunto_de_treino, conjunto_de_teste = train_test_split(conjunto_de_dados.readlines(), test_size=float(porcentagem), random_state=42) 

	with open(treino, 'w') as arquivo:
		for dado in conjunto_de_treino:
			arquivo.write(dado) 

	with open(teste, 'w') as arquivo:
		for dado in conjunto_de_teste:
			arquivo.write(dado) 

	conjunto_de_dados.close() 