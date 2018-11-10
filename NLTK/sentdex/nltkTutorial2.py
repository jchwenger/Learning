# Tutorial bz SentDex creator here : https://www.youtube.com/watch?v=FLZvOKSCkxY&list=PLQVvvaa0QuDf2JswnfiGkliBInZnIC4HL

import nltk


# Lesson 2 - stop words

from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize

example_text = "Hullo hullo, Mrs. Father Mucker! What's up in Trumplandia? Here the war is raging, the cunts are all like dogs with 666 tails." 
example_text += " This is an example showing off stop fucking word filtration."

stop_words = set(stopwords.words("english"))

# print(stop_words)
# print(len(stop_words))
# sw = list(stop_words)
# print(sw[0])

words = word_tokenize(example_text)

# filtered_sentence = []

# for w in words:
# 	if w not in stop_words:
# 		filtered_sentence.append(w)

filtered_sentence = [w for w in words if not w in stop_words]

print(filtered_sentence)