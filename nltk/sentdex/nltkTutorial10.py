# Tutorial bz SentDex creator here : https://www.youtube.com/watch?v=FLZvOKSCkxY&list=PLQVvvaa0QuDf2JswnfiGkliBInZnIC4HL

import nltk

# Lesson 10 - WordNet

# from nltk.corpus import wordnet

syns = wordnet.synsets("program")

# synset
print(syns[2])

# the lemmas within it
print(syns[2].lemmas())

# just the word for one of the lemmas
print(syns[2].lemmas()[0].name())

# definition
print(syns[2].definition())

# examples
print(syns[2].examples())

synonyms = []
antonyms = []

for syn in wordnet.synsets("good"):
	for l in syn.lemmas():
		# print("l synonyms: ", l)
		synonyms.append(l.name())
		if l.antonyms():
			# print("l antonym: ", l.antonyms())	
			antonyms.append(l.antonyms()[0].name())

print(set(synonyms))
print(set(antonyms))

# meaning similarity 

w1 = wordnet.synset("ship.n.01")
w2 = wordnet.synset("boat.n.01")

w1 = wordnet.synset("ship.n.01")
w2 = wordnet.synset("car.n.01")

w1 = wordnet.synset("faith.n.01")
w2 = wordnet.synset("belief.n.01")

w1 = wordnet.synset("satan.n.01")
w2 = wordnet.synset("devil.n.01")

w1 = wordnet.synset("banana.n.01")
w2 = wordnet.synset("love.n.01")

print(w1.wup_similarity(w2))