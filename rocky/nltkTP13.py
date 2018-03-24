# Tutorial playlist: https://www.youtube.com/watch?v=Z-y5jWZza9Q&list=PLcTXcpndN-Sl9eYrKM6jtcOTgC52EJnqH&index=13

import re 

regex = re.compile(r'(?i)don\'t')
fst = "Don't you dare. I don't."
sst = regex.sub('do not', fst)
print(fst)
print(sst)

fst = "I won't go there. He's a mad man. He won't end that. He'd have to go now."
givenpatterns = [
					(r'won\'t', 'will not'), 
					(r'\'s', ' is'), 
					(r'\'d', ' would'),
					(r'mad man', 'crazy arse mother fucking anthropoid')
				]

def replace(text, patterns):
	for(raw, rep) in patterns:
		regex = re.compile(raw)
		text = regex.sub(rep,text)
	print(text)

print(fst)
replace(fst, givenpatterns)