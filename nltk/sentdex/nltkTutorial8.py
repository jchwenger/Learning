# Tutorial bz SentDex creator here : https://www.youtube.com/watch?v=FLZvOKSCkxY&list=PLQVvvaa0QuDf2JswnfiGkliBInZnIC4HL

import nltk


# Lesson 8 - Lemmatizing


from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

print(lemmatizer.lemmatize("cats"))
print(lemmatizer.lemmatize("cacti"))
print(lemmatizer.lemmatize("rocks"))
print(lemmatizer.lemmatize("fell"))
print(lemmatizer.lemmatize("me"))

print(lemmatizer.lemmatize("better", pos="a"))
print(lemmatizer.lemmatize("best", pos="a"))
print(lemmatizer.lemmatize("runs"))
print(lemmatizer.lemmatize("ran", 'v'))