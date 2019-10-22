import re
from nltk.corpus import wordnet 
from nltk.tokenize import word_tokenize

class RegexReplacer(object):
	def __init__(self, givenpatterns):
		self.patterns = givenpatterns

	def replace(self, text):
		for(raw, rep) in self.patterns:
			regex = re.compile(raw)
			text = regex.sub(rep, text)
		return text


class RepeatReplacer(object):
	def __init__(self):
		self.regex = re.compile(r'(\w*)(\w)\2(\w*)')
		self.repl = r'\1\2\3'

	def replace(self, word):
		# Let's add a method to check if the word actually has 
		# two letters (Tutorial 15)
		if wordnet.synsets(word):
			return word
		loop_res = self.regex.sub(self.repl, word)
		if (word == loop_res):
			return loop_res
		else:
			return self.replace(loop_res)

# Tutorial 16: https://www.youtube.com/watch?v=K-L5yQxV7LY&index=16&list=PLcTXcpndN-Sl9eYrKM6jtcOTgC52EJnqH
class WordReplacer(object):
	def __init__(self, word_map):
		self.word_map = word_map

	def replace(self, word):
		return self.word_map.get(word, word)

# Tutorial 17: https://www.youtube.com/watch?v=88yOCX4ZyoA&list=PLcTXcpndN-Sl9eYrKM6jtcOTgC52EJnqH&index=17

class AntonymReplacer(object):
	def replace(self, word):
		antonyms = set()
		for syn in wordnet.synsets(word):
			for lemma in syn.lemmas():
				for antonym in lemma.antonyms():
					antonyms.add(antonym.name())
		if len(antonyms) == 1:
			return antonyms.pop()
		else:
			return None

	def negreplace(self, string):
		i = 0
		sent = word_tokenize(string)
		len_sent = len(sent)
		words = []
		fsent = ""
		while i < len_sent:
			word = sent[i]
			if word == 'not' and i+1 < len_sent:
				ant = self.replace(sent[i+1])
				if ant:
					#words.append(ant)
					fsent += ant+" "
					i += 2
					continue
			# words.append(word)
			fsent += word+" "
			i += 1
		# return words
		return fsent