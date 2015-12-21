#Alexis Langlois
'''
Cree un fichier produisant des vecteurs de features sous la forme suivante:
	-Une colonne par n-gram (contenu dans le fichier 'ngrams_file_path')
	-Chaque colonne obtient comme valeur le compte de ce n-gram dans un read
'''

reads_file_path = 'data/reads'
ngrams_file_path = 'ngrams/3grams'
target_file_path = 'features/3grams_count_features'

with open(target_file_path,'w') as new_file:
	with open(reads_file_path, 'r') as reads:
		i = 0
		for read in reads:
			newLine = ''
			print i
			i+=1
			with open(ngrams_file_path) as seqs:
				for seq in seqs:
					if seq.strip('\n') in read:
						newLine += str(read.count(seq.strip('\n'))) + ' '
					else:
						newLine += '0 '
			new_file.write(newLine + '\n')