# Tutorial bz SentDex creator here : https://www.youtube.com/watch?v=FLZvOKSCkxY&list=PLQVvvaa0QuDf2JswnfiGkliBInZnIC4HL

import nltk


# Lesson 1

nltk.download()
tokenizing - word tokenizers, sentence tokenizers

from nltk.tokenize import sent_tokenize, word_tokenize


example_text = "Hullo hullo, Mrs. Father Mucker! What's up in Trumplandia? Here the war is raging, the cunts are all like dogs with 666 tails." 

print(sent_tokenize(example_text))
print(word_tokenize(example_text))

for i in word_tokenize(example_text):
	print(i)