#Alexis Langlois
'''
Ce script produit le fichier @weights_target_file qui doit contenir les poids associes aux n-grammes:
	Colonne1: Sous-sequence n-gramme
	Colonne2: Poids
Le poids d'un n-gramme est defini par l'ecart le plus important de son rang suivant les differentes classes divise par le nombre total de n-grammes (e.g. (rang maximal - rang minimal) / 1024)
'''

import numpy as np

weights_target_file = 'data/weights/3gram_weights'
counts_directory = 'data/counts/'
n_gram_file = 'ngrams/3grams'
normalization = sum(1 for line in open(n_gram_file))

with open(weights_target_file, 'w') as weight_file:
	with open(n_gram_file, 'r') as ngrams:
		for gram in ngrams:
			ranks = []
			for c in range(0,25):
				with open(counts_directory+'chr'+str(c)+'_counts', 'r') as counts:
					for count in counts:
						if count.split(' ')[1] == gram.strip():
							rank = count.split(' ')[0]
							rank = float(rank)
							ranks.append(rank)
			weight = (max(ranks) - min(ranks)) / normalization
			weight_file.write(gram.strip() + ' ' + str(weight) + '\n')
			print gram.strip() + ' ' + str(weight)