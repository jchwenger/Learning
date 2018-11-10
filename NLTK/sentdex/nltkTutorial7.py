# Tutorial bz SentDex creator here : https://www.youtube.com/watch?v=FLZvOKSCkxY&list=PLQVvvaa0QuDf2JswnfiGkliBInZnIC4HL

import nltk

# Lesson 7 - Named Entity Recognition


from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer 

train_text = state_union.raw("2005-GWBush.txt")
sample_text = state_union.raw("2006-GWBush.txt")

custom_sent_tokenizer = PunktSentenceTokenizer(sample_text)

tokenized = custom_sent_tokenizer.tokenize(sample_text)



def process_content():

	for i in tokenized[:2]:
		words = nltk.word_tokenize(i)
		tagged = nltk.pos_tag(words)

		namedEnt = nltk.ne_chunk(tagged)
		# Only to tag NEs without the tag included (saying if it's a person, a place, etc.)
		# namedEnt = nltk.ne_chunk(tagged, binary = True) 


		namedEnt.draw()


process_content()




# What are the Named Entity labels, found here:
# https://stackoverflow.com/questions/45210930/what-are-the-entity-types-for-nltk

# chunker = nltk.data.load(nltk.chunk._MULTICLASS_NE_CHUNKER) # cf. nltk/chunk/__init__.py
# for l in sorted(chunker._tagger._classifier.labels()):
# 	print(l)

# B-FACILITY
# B-GPE
# B-GSP
# B-LOCATION
# B-ORGANIZATION
# B-PERSON
# I-FACILITY
# I-GPE
# I-GSP
# I-LOCATION
# I-ORGANIZATION
# I-PERSON

# Other list found here: http://www.nltk.org/book/ch07.html 

# ORGANIZATION 	Georgia-Pacific Corp., WHO
# PERSON 	Eddy Bonte, President Obama
# LOCATION 	Murray River, Mount Everest
# DATE 	June, 2008-06-29
# TIME 	two fifty a m, 1:30 p.m.
# MONEY 	175 million Canadian Dollars, GBP 10.40
# PERCENT 	twenty pct, 18.75 %
# FACILITY 	Washington Monument, Stonehenge
# GPE 	South East Asia, Midlothian