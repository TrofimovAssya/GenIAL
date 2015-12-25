import sys
import cPickle as pickle

totalcount = {}
jfDB = {}
chromapriori = {}

f = open(sys.argv[1],"r")

for i in f:
    i = i.strip().split("\t")
    totalcount[i[0]] = int(i[1])

f.close()

f = open(sys.argv[2],"r")
for i in f:
    i = i.strip().split("\t")
    jfDB[i[0]] = i[1]

f.close()

f = open(sys.argv[3],"r")
for i in f:
    i = i.strip().split(" ")
    chromapriori[i[1]] = i[0]

f.close()

pickle.dump(totalcount,open("kmersum.p","w"))
pickle.dump(jfDB,open("jfDB.p","w"))
pickle.dump(chromapriori,open("class_apriori.p","w"))


