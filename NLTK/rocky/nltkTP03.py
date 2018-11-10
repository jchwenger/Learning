# Tutorial playlist here: https://www.youtube.com/watch?v=OrJqt8zHTX4&list=PLcTXcpndN-Sl9eYrKM6jtcOTgC52EJnqH&index=4

from nltk.tokenize import regexp_tokenize 

parag1 = "I won't do this, you shan't do that."

from nltk.tokenize import word_tokenize 

# print(word_tokenize(parag1))

print(regexp_tokenize(parag1, "[\w']+"))

from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer("[\w]+")

print(tokenizer.tokenize(parag1))