#Alexis Langlois
'''
Ce script produit le fichier @features_file_path contenant les vecteurs de features des reads de @dataset, soit:
	Nombre de matchs de chacun des n-grammes du fichier @ngrams_file_path avec saut de longueur @gap dans chaque read.
	Les reads doivent etre de la forme read###tag.
'''

dataset = 'data/reads'
ngrams_file_path = 'ngrams/3grams'
feature_file_path = 'features/3grams_gap10_features'
gap = 10

import re

with open(feature_file_path, 'w') as new_file:
	with open(dataset, 'r') as reads:
		i = 0
		for read in reads:
			i += 1
			with open(ngrams_file_path, 'r') as grams:
				for gram in grams:
					subsequence = gram.strip()
					cpt = 0
					for pos in range(1, len(subsequence)):
						first_part = subsequence[:pos]
						second_part = subsequence[pos:]
						regex_to_match = first_part + r"(.){" + str(gap) + "}" + second_part
						cpt += len(re.findall(regex_to_match, read.split('###')[0]))
						print re.findall(regex_to_match, read.split('###')[0]) 
						print len(re.findall(regex_to_match, read.split('###')[0]))
					new_file.write(str(cpt) + ' ')
			new_file.write('\n')