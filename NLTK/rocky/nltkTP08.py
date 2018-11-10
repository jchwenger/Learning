#Tutorial playlist here: https://www.youtube.com/watch?v=DG3IA0GVX60&index=8&list=PLcTXcpndN-Sl9eYrKM6jtcOTgC52EJnqH

from nltk.corpus import wordnet 

# In order to import everything, use the * operator
# import *

catArr = wordnet.synsets("cat")
dogArr = wordnet.synsets("dog")

print(dogArr)
print(catArr)

doi = dogArr[0]
coi = catArr[0]

print(doi)
print(coi)

print(doi.wup_similarity(coi))
print(doi.path_similarity(coi))
# A similarity of 1 describes sameness
print(doi.path_similarity(doi))

# Leacock Chodorow similarity
print(doi.lch_similarity(coi))
print(doi.lch_similarity(doi))

# Same results when inverted, 
# 3.63... seems to be the maximum?

print(coi.lch_similarity(doi))
print(coi.lch_similarity(coi))