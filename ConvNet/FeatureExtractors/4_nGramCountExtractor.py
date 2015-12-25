#Alexis Langlois
'''
Produit les comptes des n-grams contenus dans les fichiers de reads positifs de la forme:
	Colonne1: Rang de la sous-sequence n-gram
	Colonne2: Valeur de la sous-sequence n-gram
	Colonne3: Comptes
Le nom des fichiers de comptes prend la forme: 'chrX_counts', ou X est le numero du chromosome.
Les fichiers des reads positifs sont crees avec le script 3_oneVSallExtractor.py.
L'inconnu N est simplement ignore dans le processus d'extraction.
'''

n = 3
positive_reads_directory = 'data/'
target_directory = 'data/counts/'

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
	i = 0
	for read1 in data:
		read = read1.replace('N', '')
		i = i + 1
		sub_strings = cut(read)
		for seq in sub_strings:
			if seq not in seqs:
				seqs.append(seq)
				counts.append(1)
			else:
				counts[seqs.index(seq)] += 1				
	best_indices = np.argsort(counts)[-len(counts):][::-1]
	best_seqs = []
	rank = 1
	for indice in best_indices:
		best_seqs.append(str(rank) + ' ' + seqs[indice] + ' ' + str(counts[indice]))
		rank += 1
	print np.sort(counts)[::-1]
	return best_seqs

for i in range(0, 25):
	with open(positive_reads_directory+'chr'+str(i)+'_pos', 'r') as f:
		content = f.readlines()
	nucleotides = []
	for j in range(len(content)):
		nucleotides.append(content[j].strip('\n'))
	counts = count(nucleotides)
	with open(target_directory+'chr'+str(i)+'_counts', 'w') as new_file:
		for sequence in counts:
			new_file.write("%s\n" % sequence)
	print 'chromosome ' + str(i) + ' ok'