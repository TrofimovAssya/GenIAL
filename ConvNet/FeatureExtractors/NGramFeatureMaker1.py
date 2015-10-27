'''
Cree un fichier contenant une representation par features des reads traitable par le MLP sous la forme suivante:
	
	-Une colonne par n-gram (contenu dans le fichier specifie - voir NGramExtractor)
	-Chaque colonne obtient comme valeur le compte de ce n-gram
	
'''

reads_file_path = 'reads'
ngrams_file_path = 'counts'
target_file_path = 'features'

with open(target_file_path,'w') as new_file:
	with open(reads_file_path) as reads:
		i = 0
		for read in reads:
			newLine = ''
			print i
			i+=1
			with open(ngrams_file_path) as seqs:
				for seq in seqs:
					splitted_seq = seq.split(' ')
					if splitted_seq[0] in read:
						newLine += str(read.count(splitted_seq[0])) + ' '
					else:
						newLine += '0 '
			new_file.write(newLine + '\n')