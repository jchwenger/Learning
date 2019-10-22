# Tutorial video: https://www.youtube.com/watch?v=Ib_G4ZllsEM&list=PLcTXcpndN-Sl9eYrKM6jtcOTgC52EJnqH&index=11

from nltk.stem import PorterStemmer 
from nltk.stem import LancasterStemmer 
from nltk.stem import RegexpStemmer 
# SnowballStemmer for languages other than English
# (Also sometimes called Porter2Stemmer)
# Aggressive stemmer: you cut more in words than other methods,
# the result of which might be forms that are unrecognizable
# to the untrained human reader

pstemmer = PorterStemmer()
print(pstemmer.stem('dancing'))
print(pstemmer.stem('dancer'))
print(pstemmer.stem('cooking'))
print(pstemmer.stem('cookery'))

# PorterSTemmer the least aggressive

lstemmer = LancasterStemmer()
print(lstemmer.stem('dancing'))
print(lstemmer.stem('dance'))
print(lstemmer.stem('dancer'))
print(lstemmer.stem('cooking'))
print(lstemmer.stem('cookery'))

# LancasterStemmer the most aggressive

rstemmer = RegexpStemmer('ing')

print(rstemmer.stem('skiing'))
print(rstemmer.stem('cooking'))
print(rstemmer.stem('king'))

# RegexStemmer does not have a pre-learnt dataset
# It will not distinguish between 'king' and 'cooking'

