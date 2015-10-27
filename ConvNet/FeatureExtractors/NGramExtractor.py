'''
Cree un fichier contenant:

	Colonne1: Sous sequences n gram contenues dans une serie de reads (de la plus frequente a la moins frequente)
	Colonne2: Comptes des sous sequences
'''

n = 4
reads_file_path = 'reads'
target_file_path = 'counts'

import numpy as np

def cut(string):
	length = len(string)
	alist = []
	for i in xrange(length):
		for j in xrange(i,length):
			if len(string[i:j + 1]) == n:
				alist.append(string[i:j + 1]) 
	return alist
	
def count(data):
	seqs = []
	counts = []
	for read1 in data:
		print len(seqs)
		sub_strings = cut(read1)
		for seq in sub_strings:
			if seq not in seqs:
				seqs.append(seq)
				counts.append(1)
			else:
				counts[seqs.index(seq)] += 1				
	best_indices = np.argsort(counts)[-len(counts):][::-1]
	best_seqs = []
	for indice in best_indices:
		best_seqs.append(seqs[indice] + ' ' + str(counts[indice]))
	print np.sort(counts)[::-1]
	return best_seqs

with open(reads_file_path) as f:
    content = f.readlines()
nucleotides = []
for i in range(len(content)):
    nucleotides.append(content[i].strip('\n'))
counts = count(nucleotides)
with open(target_file_path,'w') as new_file:
	for sequence in counts:
		new_file.write("%s\n" % sequence)