# Tutorial bz SentDex creator here : https://www.youtube.com/watch?v=FLZvOKSCkxY&list=PLQVvvaa0QuDf2JswnfiGkliBInZnIC4HL

import nltk

# Lesson 6 - Chinking


from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer 

train_text = state_union.raw("2005-GWBush.txt")
sample_text = state_union.raw("2006-GWBush.txt")

custom_sent_tokenizer = PunktSentenceTokenizer(sample_text)

tokenized = custom_sent_tokenizer.tokenize(sample_text)



def process_content():
	try: 
		for i in tokenized:
			words = nltk.word_tokenize(i)
			tagged = nltk.pos_tag(words)


			chunkGram = r"""Chunk: {<.*>+}
									}<VB.?|IN|DT>+{"""

			chunkParser = nltk.RegexpParser(chunkGram)

			chunked = chunkParser.parse(tagged)

			# print(chunked)
			# chunked.draw()


	except Exception as e:
		print(str(e))

process_content()
