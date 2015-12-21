#Alexis Langlois
'''
Regroupe les sequences positives et les sequences negatives pour chaque chromosome contenu dans le fichier @dataset_path.
Le nom des fichiers produits prend la forme suivante: chrX_pos et chrX_neg.
Le dataset doit etre de la forme read###tag.
'''

dataset_path = 'data/reads'
onevsall_directory = 'data/'

for i in range(0,25):
	with open(onevsall_directory+'chr'+str(i)+'_pos', 'a') as positives, open(onevsall_directory+'chr'+str(i)+'_neg', 'a') as negatives:
		with open(dataset_path, 'r') as reads:
			for read in reads:
				seq = read.split('###chr')[0]
				label = read.split('###chr')[1].strip()
				if (label == str(i+1)):
					positives.write(seq + '\n')
				else:
					negatives.write(seq + '\n')
	print 'chromosome ' + str(i+1) + ' ok'