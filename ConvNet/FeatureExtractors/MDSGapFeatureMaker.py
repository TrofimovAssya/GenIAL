'''
Cree un fichier contenant:
	Presence (1) ou Absence (0) d'une sous-chaine avec gap dans un read.
	Les sous-chaines avec gap recherchees sont specifiees a l'aide de MDSGapExtractor.py et doivent etre sous la forme:
		
		AGCT 5
		
		ou {sous chaine = AGCT, gap = 5}
'''

import re

with open('feature/mdsg_features', 'w') as new_file:
	with open('data/reads_mc', 'r') as reads:
		i = 0
		for read in reads:
			i += 1
			print i
			with open('MDSgrams', 'r') as subsequences:
				for subsequence_gap in subsequences:
					subsequence = subsequence_gap.split(' ')[0]
					gap = subsequence_gap.split(' ')[1].strip('\n')
					isIncluded = False
					for pos in range(1, len(subsequence)):
						first_part = subsequence[:pos]
						second_part = subsequence[pos:]
						regex_to_match = first_part + r"(.){" + str(gap) + "}" + second_part
						if re.search(regex_to_match, read.split('###')[0]):
							isIncluded = True
							break
					if isIncluded:
						new_file.write('1 ')
					else:
						new_file.write('0 ')
			new_file.write('\n')