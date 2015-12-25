#Alexis Langlois
'''
Produit un fichier @reads_file_path contenant des vecteurs de features de la forme:
	Colonne1: n-gramme
	Colonne2: Compte du n-gramme * poids correspondant
'''

reads_file_path = 'data/reads'
weights_file_path = 'data/weights/5gram_weights'
target_file_path = 'features/5grams_weight_features'

#Chargement des poids et des n-grammes
ngrams = []
with open(weights_file_path, 'r') as grams:
	for w in grams:
		ngrams.append(w)

#Creation des features
with open(target_file_path,'w') as new_file:
	with open(reads_file_path, 'r') as reads:
		i = 0
		for read in reads:
			newLine = ''
			i+=1
			for seq in ngrams:
				gram = seq.split(' ')[0]
				if gram in read:
					weight = seq.split(' ')[1].strip()
					count = read.count(gram)
					total_count = float(count) * float(weight)
					total_count = "{0:.2f}".format(total_count)
					newLine += str(total_count) + ' '
				else:
					newLine += '0 '
			new_file.write(newLine + '\n')
			
			#Avancement
			if i % 10000 == 0:
				print i