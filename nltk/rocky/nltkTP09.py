# Tutorial playlist: https://www.youtube.com/watch?v=dZvCHz6lcGU&index=9&list=PLcTXcpndN-Sl9eYrKM6jtcOTgC52EJnqH

from nltk.corpus import webtext 
from nltk.collocations import BigramCollocationFinder 
from nltk.metrics import BigramAssocMeasures

textWords = [w.lower() for w in webtext.words('pirates.txt')]

# print(textWords)
# print(len(textWords))

finder = BigramCollocationFinder.from_words(textWords)

likeliestW = finder.nbest(BigramAssocMeasures.likelihood_ratio, 20)

# print(likeliestW)

# As we don't want stop words, let's add some filters 

from nltk.corpus import stopwords 

ignored_words = set(stopwords.words('english'))

filterStops = lambda w: len(w) < 3 or w in ignored_words

finder.apply_word_filter(filterStops)
likeliestW = finder.nbest(BigramAssocMeasures.likelihood_ratio, 20)

print(likeliestW)