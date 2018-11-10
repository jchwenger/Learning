# Tutorial playlist: https://www.youtube.com/watch?v=uWAcgI96nEU&list=PLcTXcpndN-Sl9eYrKM6jtcOTgC52EJnqH&index=12

from nltk.stem import WordNetLemmatizer 

lzr = WordNetLemmatizer()

print(lzr.lemmatize('dancing'))
print(lzr.lemmatize('working'))

# are these nouns or verbs?

print(lzr.lemmatize('dancing', pos='v'))
print(lzr.lemmatize('working', pos='v'))
print(lzr.lemmatize('working', pos='a'))

print(lzr.lemmatize('kings'))

print(lzr.lemmatize('sings'))
print(lzr.lemmatize('sings', pos='v'))

print(lzr.lemmatize('abruptly', pos='r'))

from nltk.stem import PorterStemmer

stm = PorterStemmer()

print(stm.stem('dancing'))
print(lzr.lemmatize('dancing', pos='v'))

print(stm.stem('believes'))
print(lzr.lemmatize('believes'))
print(lzr.lemmatize('believes', pos='v'))

print(stm.stem('buses'))
print(lzr.lemmatize('buses'))

print(stm.stem('bus'))
