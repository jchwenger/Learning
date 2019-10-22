# Tutorial playlist here: https://www.youtube.com/watch?v=s4HnwYn4aj0&list=PLcTXcpndN-Sl9eYrKM6jtcOTgC52EJnqH&index=6

from nltk.corpus import wordnet 

sArr = wordnet.synsets('win')
print(sArr)
woi = sArr[2]
print(woi)
print(woi.pos())
print(woi.lemmas())
print(woi.lemmas()[0].name())

synArr = []
antArr = []

for syn in sArr:
	for lem in syn.lemmas():
		synArr.append(lem.name())

print(synArr)
print(len(synArr))
print(set(synArr))
print(len(set(synArr)))

print(woi.lemmas()[0].antonyms())

for syn in sArr:
	for lem in syn.lemmas():
		for ant in lem.antonyms():
			antArr.append(ant.name())

print(antArr)
print(len(antArr))
print(set(antArr))
print(len(set(antArr)))