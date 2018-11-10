# Tutorial playlist: https://www.youtube.com/watch?v=2XR7KTq3rco&list=PLcTXcpndN-Sl9eYrKM6jtcOTgC52EJnqH&index=10

from nltk.collocations import TrigramCollocationFinder
from nltk.corpus import webtext 
from nltk.metrics import TrigramAssocMeasures 

textWords = [w.lower() for w in webtext.words('grail.txt')]

finder = TrigramCollocationFinder.from_words(textWords)

likeliestW = finder.nbest(TrigramAssocMeasures.likelihood_ratio, 20)

print(likeliestW)

# Let's apply some filters

from nltk.corpus import stopwords 

ignored_words = set(stopwords.words('english'))

filterStops = lambda w: len(w) < 3 or w in ignored_words 

finder.apply_word_filter(filterStops)

likeliestW = finder.nbest(TrigramAssocMeasures.likelihood_ratio, 30)

print('\n')
print(likeliestW)

finder.apply_freq_filter(3)

likeliestW = finder.nbest(TrigramAssocMeasures.likelihood_ratio, 30)

print('\n')
print(likeliestW)