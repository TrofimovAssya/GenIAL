#Alexis Langlois
'''
Ce script produit le fichier @features_file_path contenant:
	Presence (1) ou Absence (0) d'une sous-chaine distinctive minimale (MDS) dans un read.
	Les sous-chaines recherchees sont contenues dans le fichier @mds_file_path produit par le script 7_MDSGapExtractor.py de la forme:
		AGCT 10
			ou sous chaine = AGCT, gap = 10
'''

reads_file_path = 'data/reads_mc'
features_file_path = 'feature/mdsg_features'
mds_file_path = 'MDSgrams'

import re

with open(features_file_path, 'w') as new_file:
	with open(reads_file_path, 'r') as reads:
		i = 0
		for read in reads:
			i += 1
			print i
			with open(mds_file_path, 'r') as subsequences:
				for subsequence_gap in subsequences:
					subsequence = subsequence_gap.split(' ')[0]
					gap = subsequence_gap.split(' ')[1].strip('\n')
					isIncluded = False
					for pos in range(1, len(subsequence)):
						first_part = subsequence[:pos]
						second_part = subsequence[pos:]
						regex_to_match = first_part + r"(.){" + str(gap) + "}" + second_part
						if re.search(regex_to_match, read.strip()):
							isIncluded = True
							break
					if isIncluded:
						new_file.write('1 ')
					else:
						new_file.write('0 ')
			new_file.write('\n')