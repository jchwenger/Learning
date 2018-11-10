from replacer import RegexReplacer 

givenpatterns = [
					(r'won\'t', 'will not'), 
					(r'\'s', ' is'), 
					(r'\'d', ' would'),
					(r'mad man', 'crazy arse mother fucking anthropoid')
				]

replacer = RegexReplacer(givenpatterns)
# replacer.patterns = givenpatterns

txt = replacer.replace("He's gone")

print(txt)


from replacer import RepeatReplacer

replacer = RepeatReplacer()

txt = replacer.replace("Anthhhhropoiiid")
print(txt)

# Tutorial 15: https://www.youtube.com/watch?v=r37OYsdH6Z8&list=PLcTXcpndN-Sl9eYrKM6jtcOTgC52EJnqH&index=15

# Thanks to the wordnet checking function, 'book' or
# 'cattle' will return as they are
txt = replacer.replace("Book")
print(txt)

txt = replacer.replace("cattle")
print(txt)

txt = replacer.replace("botttleeee")
print(txt)
 
# Tutorial 16: https://www.youtube.com/watch?v=K-L5yQxV7LY&index=16&list=PLcTXcpndN-Sl9eYrKM6jtcOTgC52EJnqH

from replacer import WordReplacer
from nltk.tokenize import word_tokenize

wordmapobj = {
				'bday' : 'birthday',
				'sup' : 'what\'s up',
				'brb' : 'be right back'
			}

replacer = WordReplacer(wordmapobj)

result = replacer.replace("bday")
print(result)

result = replacer.replace("sup")
print(result)

result = replacer.replace("brb")
print(result)

sw = word_tokenize('Sup! Awesome bday? brb!')

sw2 = ""

for word in sw:	
	result = replacer.replace(word)
	sw2 += result+" "

print(sw2)

# Tutorial 17 : https://www.youtube.com/watch?v=88yOCX4ZyoA&list=PLcTXcpndN-Sl9eYrKM6jtcOTgC52EJnqH&index=17

from replacer import AntonymReplacer

rep = AntonymReplacer()

antn = rep.replace('cowardice')
print(antn)

antn = rep.replace('heavy')
print(antn)

antn = rep.replace('weak')
print(antn)

antn = rep.replace('blue')
print(antn)

sent = rep.negreplace('this man is not salty')
print(sent)