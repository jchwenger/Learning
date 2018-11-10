# Tutorial playlist here: https://www.youtube.com/watch?v=wDPlrk8ZTMU&index=2&list=PLcTXcpndN-Sl9eYrKM6jtcOTgC52EJnqH


import nltk

from nltk.tokenize import word_tokenize

sent1 = "Let\'s hack this mother fucking Linux droid! We shan't wait."

arr = word_tokenize(sent1)

print(arr)

from nltk.tokenize import TreebankWordTokenizer

tok2 = TreebankWordTokenizer()

from nltk.tokenize import WordPunctTokenizer 

tok3 = WordPunctTokenizer()

print(tok2.tokenize(sent1))
print(tok3.tokenize(sent1))
