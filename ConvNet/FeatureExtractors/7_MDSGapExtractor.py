#Alexis Langlois
'''
Ce script produit le fichier @target_file_path qui contient la liste des sous-sequences distinctives minimales pour un chromosome.
Les sous-sequences doivent apparaitre dans un ratio @beta de la classe positive (@pos_file_path) et dans un ratio @alpha des classes negatives (@neg_file_path)
Les fichiers d'exemples doivent etre de la forme: read###tag
'''

import re

pos_file_path = 'data/chr1_pos'
neg_file_path = 'data/chr1_neg'
target_file_path = 'MDSgram'
gap = 10
beta = 0.40
alpha = 0.25
ARN = ['A', 'C', 'G', 'T']
SMDS = []
c = ''

#Verification pour super-sequence (seq1 is included in seq2)
def SuperSequence(ds, nc):
	for seq in ds:
		if seq in nc:
			return True
	return False

#Tri par frequence des sous-sequences
def SortByFrequencies(ds):
	freqs = []
	for seq in ds:
		cpt = 0
		with open(pos_file_path, 'r') as reads:
			for read in reads:
				cpt = cpt + read.count(seq)
		with open(neg_file_path, 'r') as reads:
			for read in reads:
				cpt = cpt + read.count(seq)
		freqs.append(cpt)
	return [x for (y,x) in sorted(zip(freqs,ds))]

#Fonction de denombrement des sous-sequence avec gap
def SupportCount(subsequence, gap, reads):
	cpt = 0
	read_cpt = 0
	with open(reads, 'r') as reads:
		for read in reads:
			read_cpt = read_cpt + 1
			for pos in range(1, len(subsequence)):
				first_part = subsequence[:pos]
				second_part = subsequence[pos:]
				regex_to_match = first_part + r"(.){" + str(gap) + "}" + second_part
				if re.search(regex_to_match, read.split('###')[0]):
					cpt = cpt + 1
					break
	return float(float(cpt) / float(read_cpt))

#Fonction principale: les conditions alpha et beta sont verifiees
def CandidateGen(c, gap, beta, alpha):
	ds = []
	for i in ARN:
		nc = c + i
		if SuperSequence(ds, nc) == False:
			supp_pos = SupportCount(nc, gap, pos_file_path)
			if supp_pos >= beta:
				print supp_pos
				supp_neg = SupportCount(nc, gap, neg_file_path)
				print supp_neg
				if supp_neg <= alpha:
					ds.append(nc)
				else:
					CandidateGen(nc, gap, beta, alpha)
		print ds
		print nc
	SMDS.extend(ds)	

#Uniquement les sous-sequences minimales sont conservees
def Minimisation(ds):
	pt = []
	asc_ds = SortByFrequencies(ds)
	for s in asc_ds:
		isIncluded = False
		for p in pt:
			if p in s:
				asc_ds.remove(s)
				isIncluded = True
		if not isIncluded:
			pt.append(s)
	return asc_ds

#Main
CandidateGen(c, gap, beta, alpha)
mds = Minimisation(SMDS)
print mds

#Ecriture
with open(target_file_path, 'a') as new_file:
	for gram in mds:
		new_file.write(gram + ' ' + gap + '\n')