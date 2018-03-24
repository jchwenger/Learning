# Tutorial playlist here: https://www.youtube.com/watch?v=onWfWqRO-Gc&list=PLcTXcpndN-Sl9eYrKM6jtcOTgC52EJnqH&index=5

from nltk.corpus import wordnet

word1 = "weapon"
synArray = wordnet.synsets(word1)
print(synArray)
woi = synArray[0]
print(woi)
print(woi.definition())
print(woi.name())
print(woi.pos())

print(woi.hypernyms())
print(woi.hyponyms())

woi2 = woi.hyponyms()[12]
print(woi2)
print(woi2.hypernyms())
print(woi2.definition())

