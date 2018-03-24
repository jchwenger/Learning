import nltk

# --------------------------------------------

## Lesson 1

#nltk.download()
# tokenizing - word tokenizers, sentence tokenizers

# from nltk.tokenize import sent_tokenize, word_tokenize


# example_text = "Hullo hullo, Mrs. Father Mucker! What's up in Trumplandia? Here the war is raging, the cunts are all like dogs with 666 tails." 

# print(sent_tokenize(example_text))
# print(word_tokenize(example_text))

# for i in word_tokenize(example_text):
# 	print(i)

# --------------------------------------------

## Lesson 2 - stop words

# from nltk.corpus import stopwords 
# from nltk.tokenize import word_tokenize

# example_text = "Hullo hullo, Mrs. Father Mucker! What's up in Trumplandia? Here the war is raging, the cunts are all like dogs with 666 tails." 
# example_text += " This is an example showing off stop fucking word filtration."

# stop_words = set(stopwords.words("english"))

# # print(stop_words)
# # print(len(stop_words))
# # sw = list(stop_words)
# # print(sw[0])

# words = word_tokenize(example_text)

# # filtered_sentence = []

# # for w in words:
# # 	if w not in stop_words:
# # 		filtered_sentence.append(w)

# filtered_sentence = [w for w in words if not w in stop_words]

# print(filtered_sentence)

# --------------------------------------------

# Lesson 3 - stemming

# from nltk.stem import PorterStemmer
# from nltk.tokenize import word_tokenize, sent_tokenize

# ps = PorterStemmer()

# # example_words = ["python", "pythoner", "pythoning", "pythoned", "pythonly"]

# # for w in example_words:
# # 	print(ps.stem(w))

# new_text = "It is very important to be pythonly while you are pythoning with python. All pythoners have pythoned poorly at least once."

# # print(sent_tokenize(new_text))
# words = word_tokenize(new_text)

# for w in words:
# 	print(ps.stem(w))

# --------------------------------------------

# Lesson 4 - Part of Speech Tagging

# from nltk.corpus import state_union
# from nltk.tokenize import PunktSentenceTokenizer 

# train_text = state_union.raw("2005-GWBush.txt")
# sample_text = state_union.raw("2006-GWBush.txt")

# custom_sent_tokenizer = PunktSentenceTokenizer(sample_text)

# tokenized = custom_sent_tokenizer.tokenize(sample_text)

# def process_content():
# 	try: 
# 		for i in tokenized:
# 			words = nltk.word_tokenize(i)
# 			tagged = nltk.pos_tag(words)

# 			print(tagged)

# 	except Exception as e:
# 		print(str(e))

# process_content()

# nltk.help.upenn_tagset()

# $: dollar
#     $ -$ --$ A$ C$ HK$ M$ NZ$ S$ U.S.$ US$
# '': closing quotation mark
#     ' ''
# (: opening parenthesis
#     ( [ {
# ): closing parenthesis
#     ) ] }
# ,: comma
#     ,
# --: dash
#     --
# .: sentence terminator
#     . ! ?
# :: colon or ellipsis
#     : ; ...
# CC: conjunction, coordinating
#     & 'n and both but either et for less minus neither nor or plus so
#     therefore times v. versus vs. whether yet
# CD: numeral, cardinal
#     mid-1890 nine-thirty forty-two one-tenth ten million 0.5 one forty-
#     seven 1987 twenty '79 zero two 78-degrees eighty-four IX '60s .025
#     fifteen 271,124 dozen quintillion DM2,000 ...
# DT: determiner
#     all an another any both del each either every half la many much nary
#     neither no some such that the them these this those
# EX: existential there
#     there
# FW: foreign word
#     gemeinschaft hund ich jeux habeas Haementeria Herr K'ang-si vous
#     lutihaw alai je jour objets salutaris fille quibusdam pas trop Monte
#     terram fiche oui corporis ...
# IN: preposition or conjunction, subordinating
#     astride among uppon whether out inside pro despite on by throughout
#     below within for towards near behind atop around if like until below
#     next into if beside ...
# JJ: adjective or numeral, ordinal
#     third ill-mannered pre-war regrettable oiled calamitous first separable
#     ectoplasmic battery-powered participatory fourth still-to-be-named
#     multilingual multi-disciplinary ...
# JJR: adjective, comparative
#     bleaker braver breezier briefer brighter brisker broader bumper busier
#     calmer cheaper choosier cleaner clearer closer colder commoner costlier
#     cozier creamier crunchier cuter ...
# JJS: adjective, superlative
#     calmest cheapest choicest classiest cleanest clearest closest commonest
#     corniest costliest crassest creepiest crudest cutest darkest deadliest
#     dearest deepest densest dinkiest ...
# LS: list item marker
#     A A. B B. C C. D E F First G H I J K One SP-44001 SP-44002 SP-44005
#     SP-44007 Second Third Three Two * a b c d first five four one six three
#     two
# MD: modal auxiliary
#     can cannot could couldn't dare may might must need ought shall should
#     shouldn't will would
# NN: noun, common, singular or mass
#     common-carrier cabbage knuckle-duster Casino afghan shed thermostat
#     investment slide humour falloff slick wind hyena override subhumanity
#     machinist ...
# NNP: noun, proper, singular
#     Motown Venneboerger Czestochwa Ranzer Conchita Trumplane Christos
#     Oceanside Escobar Kreisler Sawyer Cougar Yvette Ervin ODI Darryl CTCA
#     Shannon A.K.C. Meltex Liverpool ...
# NNPS: noun, proper, plural
#     Americans Americas Amharas Amityvilles Amusements Anarcho-Syndicalists
#     Andalusians Andes Andruses Angels Animals Anthony Antilles Antiques
#     Apache Apaches Apocrypha ...
# NNS: noun, common, plural
#     undergraduates scotches bric-a-brac products bodyguards facets coasts
#     divestitures storehouses designs clubs fragrances averages
#     subjectivists apprehensions muses factory-jobs ...
# PDT: pre-determiner
#     all both half many quite such sure this
# POS: genitive marker
#     ' 's
# PRP: pronoun, personal
#     hers herself him himself hisself it itself me myself one oneself ours
#     ourselves ownself self she thee theirs them themselves they thou thy us
# PRP$: pronoun, possessive
#     her his mine my our ours their thy your
# RB: adverb
#     occasionally unabatingly maddeningly adventurously professedly
#     stirringly prominently technologically magisterially predominately
#     swiftly fiscally pitilessly ...
# RBR: adverb, comparative
#     further gloomier grander graver greater grimmer harder harsher
#     healthier heavier higher however larger later leaner lengthier less-
#     perfectly lesser lonelier longer louder lower more ...
# RBS: adverb, superlative
#     best biggest bluntest earliest farthest first furthest hardest
#     heartiest highest largest least less most nearest second tightest worst
# RP: particle
#     aboard about across along apart around aside at away back before behind
#     by crop down ever fast for forth from go high i.e. in into just later
#     low more off on open out over per pie raising start teeth that through
#     under unto up up-pp upon whole with you
# SYM: symbol
#     % & ' '' ''. ) ). * + ,. < = > @ A[fj] U.S U.S.S.R * ** ***
# TO: "to" as preposition or infinitive marker
#     to
# UH: interjection
#     Goodbye Goody Gosh Wow Jeepers Jee-sus Hubba Hey Kee-reist Oops amen
#     huh howdy uh dammit whammo shucks heck anyways whodunnit honey golly
#     man baby diddle hush sonuvabitch ...
# VB: verb, base form
#     ask assemble assess assign assume atone attention avoid bake balkanize
#     bank begin behold believe bend benefit bevel beware bless boil bomb
#     boost brace break bring broil brush build ...
# VBD: verb, past tense
#     dipped pleaded swiped regummed soaked tidied convened halted registered
#     cushioned exacted snubbed strode aimed adopted belied figgered
#     speculated wore appreciated contemplated ...
# VBG: verb, present participle or gerund
#     telegraphing stirring focusing angering judging stalling lactating
#     hankerin' alleging veering capping approaching traveling besieging
#     encrypting interrupting erasing wincing ...
# VBN: verb, past participle
#     multihulled dilapidated aerosolized chaired languished panelized used
#     experimented flourished imitated reunifed factored condensed sheared
#     unsettled primed dubbed desired ...
# VBP: verb, present tense, not 3rd person singular
#     predominate wrap resort sue twist spill cure lengthen brush terminate
#     appear tend stray glisten obtain comprise detest tease attract
#     emphasize mold postpone sever return wag ...
# VBZ: verb, present tense, 3rd person singular
#     bases reconstructs marks mixes displeases seals carps weaves snatches
#     slumps stretches authorizes smolders pictures emerges stockpiles
#     seduces fizzes uses bolsters slaps speaks pleads ...
# WDT: WH-determiner
#     that what whatever which whichever
# WP: WH-pronoun
#     that what whatever whatsoever which who whom whosoever
# WP$: WH-pronoun, possessive
#     whose
# WRB: Wh-adverb
#     how however whence whenever where whereby whereever wherein whereof why
# ``: opening quotation mark
#     ` ``


# --------------------------------------------

# Lesson 5 - Chunking

# from nltk.corpus import state_union
# from nltk.tokenize import PunktSentenceTokenizer 

# train_text = state_union.raw("2005-GWBush.txt")
# sample_text = state_union.raw("2006-GWBush.txt")

# custom_sent_tokenizer = PunktSentenceTokenizer(sample_text)

# tokenized = custom_sent_tokenizer.tokenize(sample_text)

# def process_content():
# 	try: 
# 		for i in tokenized:
# 			words = nltk.word_tokenize(i)
# 			tagged = nltk.pos_tag(words)

# 			# A regular expression parser: zero or more adverbs of any kind
# 			# followed by zero or more verbs of any kind, followed by a proper
# 			# singular noun and finally zero or one singular noun
 
# 			chunkGram = r"""Chunk: {<RB.?>*<VB.?>*<NNP>+<NN>?}"""

# 			chunkParser = nltk.RegexpParser(chunkGram)

# 			chunked = chunkParser.parse(tagged)

# 			# print(chunked)
# 			chunked.draw()


# 	except Exception as e:
# 		print(str(e))

# process_content()


# --------------------------------------------

# Lesson 6 - Chinking

# from nltk.corpus import state_union
# from nltk.tokenize import PunktSentenceTokenizer 

# train_text = state_union.raw("2005-GWBush.txt")
# sample_text = state_union.raw("2006-GWBush.txt")

# custom_sent_tokenizer = PunktSentenceTokenizer(sample_text)

# tokenized = custom_sent_tokenizer.tokenize(sample_text)



# def process_content():
# 	try: 
# 		for i in tokenized:
# 			words = nltk.word_tokenize(i)
# 			tagged = nltk.pos_tag(words)


# 			chunkGram = r"""Chunk: {<.*>+}
# 									}<VB.?|IN|DT>+{"""

# 			chunkParser = nltk.RegexpParser(chunkGram)

# 			chunked = chunkParser.parse(tagged)

# 			# print(chunked)
# 			# chunked.draw()


# 	except Exception as e:
# 		print(str(e))

# process_content()


# --------------------------------------------

# Lesson 7 - Named Entity Recognition


# from nltk.corpus import state_union
# from nltk.tokenize import PunktSentenceTokenizer 

# train_text = state_union.raw("2005-GWBush.txt")
# sample_text = state_union.raw("2006-GWBush.txt")

# custom_sent_tokenizer = PunktSentenceTokenizer(sample_text)

# tokenized = custom_sent_tokenizer.tokenize(sample_text)



# def process_content():

# 	for i in tokenized[:2]:
# 		words = nltk.word_tokenize(i)
# 		tagged = nltk.pos_tag(words)

# 		namedEnt = nltk.ne_chunk(tagged)
# 		# Only to tag NEs without the tag included (saying if it's a person, a place, etc.)
# 		# namedEnt = nltk.ne_chunk(tagged, binary = True) 


# 		namedEnt.draw()


# process_content()

# What are the Named Entity labels, found here:
# https://stackoverflow.com/questions/45210930/what-are-the-entity-types-for-nltk

# chunker = nltk.data.load(nltk.chunk._MULTICLASS_NE_CHUNKER) # cf. nltk/chunk/__init__.py
# for l in sorted(chunker._tagger._classifier.labels()):
# 	print(l)

# B-FACILITY
# B-GPE
# B-GSP
# B-LOCATION
# B-ORGANIZATION
# B-PERSON
# I-FACILITY
# I-GPE
# I-GSP
# I-LOCATION
# I-ORGANIZATION
# I-PERSON

# Other list found here: http://www.nltk.org/book/ch07.html 

# ORGANIZATION 	Georgia-Pacific Corp., WHO
# PERSON 	Eddy Bonte, President Obama
# LOCATION 	Murray River, Mount Everest
# DATE 	June, 2008-06-29
# TIME 	two fifty a m, 1:30 p.m.
# MONEY 	175 million Canadian Dollars, GBP 10.40
# PERCENT 	twenty pct, 18.75 %
# FACILITY 	Washington Monument, Stonehenge
# GPE 	South East Asia, Midlothian

# --------------------------------------------

# Lesson 8 - Lemmatizing

# from nltk.stem import WordNetLemmatizer

# lemmatizer = WordNetLemmatizer()

# print(lemmatizer.lemmatize("cats"))
# print(lemmatizer.lemmatize("cacti"))
# print(lemmatizer.lemmatize("rocks"))
# print(lemmatizer.lemmatize("fell"))
# print(lemmatizer.lemmatize("me"))

# print(lemmatizer.lemmatize("better", pos="a"))
# print(lemmatizer.lemmatize("best", pos="a"))
# print(lemmatizer.lemmatize("runs"))
# print(lemmatizer.lemmatize("ran", 'v'))

# --------------------------------------------

# Lesson 9 - Corpora

# Check where modules are installed
# help('modules')

# Where's the init file
# print(nltk.__file__)

# from nltk.corpus import gutenberg
# from nltk.tokenize import sent_tokenize

# sample = gutenberg.raw("bible-kjv.txt")

# tok = sent_tokenize(sample)

# print(tok[5:15])


# --------------------------------------------

# Lesson 10 - WordNet

# from nltk.corpus import wordnet

# syns = wordnet.synsets("program")

# synset
# print(syns[2])

# the lemmas within it
# print(syns[2].lemmas())

# just the word for one of the lemmas
# print(syns[2].lemmas()[0].name())

# definition
# print(syns[2].definition())

# examples
# print(syns[2].examples())

# synonyms = []
# antonyms = []

# for syn in wordnet.synsets("good"):
# 	for l in syn.lemmas():
# 		# print("l synonyms: ", l)
# 		synonyms.append(l.name())
# 		if l.antonyms():
# 			# print("l antonym: ", l.antonyms())	
# 			antonyms.append(l.antonyms()[0].name())

# print(set(synonyms))
# print(set(antonyms))

# meaning similarity 

# w1 = wordnet.synset("ship.n.01")
# w2 = wordnet.synset("boat.n.01")

# w1 = wordnet.synset("ship.n.01")
# w2 = wordnet.synset("car.n.01")

# w1 = wordnet.synset("faith.n.01")
# w2 = wordnet.synset("belief.n.01")

# w1 = wordnet.synset("satan.n.01")
# w2 = wordnet.synset("devil.n.01")

# w1 = wordnet.synset("banana.n.01")
# w2 = wordnet.synset("love.n.01")

# print(w1.wup_similarity(w2))

# --------------------------------------------

# Lesson 11 - Text classification

# import random
# from nltk.corpus import movie_reviews

# documents = [(list(movie_reviews.words(fileid)), category)
# 			for category in movie_reviews.categories()
# 			for fileid in movie_reviews.fileids(category)]

# one liner equivalent to this :
# documents = []

# for category in movie_reviews.categories():
# 	for fileid in movie_reviews.fileids(category):
# 		documents.append(list(movie_reviews.words(fileid)), category)

# random.shuffle(documents)

# print(documents[1])

# all_words = []

# for w in movie_reviews.words():
# 	all_words.append(w.lower())

# all_words = nltk.FreqDist(all_words)

# print(all_words.most_common(15))

# print(all_words["stupid"])

# --------------------------------------------

# Lesson 12 - Words as features


# import random
# from nltk.corpus import movie_reviews

# documents = [(list(movie_reviews.words(fileid)), category)
# 			for category in movie_reviews.categories()
# 			for fileid in movie_reviews.fileids(category)]


# random.shuffle(documents)
 
# all_words = []

# for w in movie_reviews.words():
# 	all_words.append(w.lower())

# all_words = nltk.FreqDist(all_words)

# word_features = list(all_words.keys())[:3000]

# def find_features(document):
# 	words = set(document)
# 	features = {}
# 	for w in word_features:
# 		features[w] = (w in words)

# 	return features

# print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))

# featuresets = [(find_features(rev), category) for (rev, category) in documents]

# --------------------------------------------

# Lesson 13  - Naive Bayes


# import random
# from nltk.corpus import movie_reviews

# documents = [(list(movie_reviews.words(fileid)), category)
# 			for category in movie_reviews.categories()
# 			for fileid in movie_reviews.fileids(category)]


# random.shuffle(documents)
 
# all_words = []

# for w in movie_reviews.words():
# 	all_words.append(w.lower())

# all_words = nltk.FreqDist(all_words)

# word_features = list(all_words.keys())[:3000]

# def find_features(document):
# 	words = set(document)
# 	features = {}
# 	for w in word_features:
# 		features[w] = (w in words)

# 	return features

# # print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))

# featuresets = [(find_features(rev), category) for (rev, category) in documents]

# training_set = featuresets[:1900]
# testing_set = featuresets[1900:]

# # posterior = prior occurrences x likelihood / evidence

# classifier = nltk.NaiveBayesClassifier.train(training_set)

# print("Naive Bayes Algo accuracy: ", (nltk.classify.accuracy(classifier, testing_set))*100)

# classifier.show_most_informative_features(15)


# --------------------------------------------

# Lesson 14 - Save classifier with Pickle

# import random
# from nltk.corpus import movie_reviews
# import pickle

# documents = [(list(movie_reviews.words(fileid)), category)
# 			for category in movie_reviews.categories()
# 			for fileid in movie_reviews.fileids(category)]


# random.shuffle(documents)
 
# all_words = []

# for w in movie_reviews.words():
# 	all_words.append(w.lower())

# all_words = nltk.FreqDist(all_words)

# word_features = list(all_words.keys())[:3000]

# def find_features(document):
# 	words = set(document)
# 	features = {}
# 	for w in word_features:
# 		features[w] = (w in words)

# 	return features

# print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))

# featuresets = [(find_features(rev), category) for (rev, category) in documents]

# training_set = featuresets[:1900]
# testing_set = featuresets[1900:]

# # posterior = prior occurrences x likelihood / evidence

# classifier = nltk.NaiveBayesClassifier.train(training_set)


# print("Naive Bayes Algo accuracy: ", (nltk.classify.accuracy(classifier, testing_set))*100)
# classifier.show_most_informative_features(15)

# # Save the classifier

# save_classifier = open("naivebayes.pickle", "wb")
# pickle.dump(classifier, save_classifier)
# save_classifier.close()

# # Load it afterward

# classifier_f = open("naivebayes.pickle", "rb")
# classifier = pickle.load(classifier_f)
# classifier_f.close()

# --------------------------------------------

# Lesson 15 - Scikit-learn incorporation


# import random
# from nltk.corpus import movie_reviews
# from nltk.classify.scikitlearn import SklearnClassifier
# import pickle

# from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB

# from sklearn.linear_model import LogisticRegression, SGDClassifier
# from sklearn.svm import SVC, LinearSVC, NuSVC

# documents = [(list(movie_reviews.words(fileid)), category)
# 			for category in movie_reviews.categories()
# 			for fileid in movie_reviews.fileids(category)]


# random.shuffle(documents)
 
# all_words = []

# for w in movie_reviews.words():
# 	all_words.append(w.lower())

# all_words = nltk.FreqDist(all_words)

# word_features = list(all_words.keys())[:3000]

# def find_features(document):
# 	words = set(document)
# 	features = {}
# 	for w in word_features:
# 		features[w] = (w in words)

# 	return features

# featuresets = [(find_features(rev), category) for (rev, category) in documents]

# training_set = featuresets[:1900]
# testing_set = featuresets[1900:]

# # posterior = prior occurrences x likelihood / evidence

# classifier = nltk.NaiveBayesClassifier.train(training_set)

# # Load it afterward

# classifier_f = open("naivebayes.pickle", "rb")
# classifier = pickle.load(classifier_f)
# classifier_f.close()


# print("Naive Bayes Algo accuracy: ", (nltk.classify.accuracy(classifier, testing_set))*100)
# classifier.show_most_informative_features(15)

# # Multinomial 
# MNB_classifier = SklearnClassifier(MultinomialNB())
# MNB_classifier.train(training_set)
# print("MNB_classifier accuracy: ", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)

# # Gaussian
# # Error regarding a sparse matrix being passed instead of a dense one

# # GaussianNB_classifier = SklearnClassifier(GaussianNB())
# # GaussianNB_classifier.train(training_set)
# # print("GaussianNB accuracy: ", (nltk.classify.accuracy(GaussianNB_classifier, testing_set))*100)

# # Bernoulli

# Bernoulli_classifier = SklearnClassifier(BernoulliNB())
# Bernoulli_classifier.train(training_set)
# print("BernoulliNB accuracy: ", (nltk.classify.accuracy(Bernoulli_classifier, testing_set))*100)


# # LogisticRegression

# LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
# LogisticRegression_classifier.train(training_set)
# print("LogisticRegression accuracy: ", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)

# # SGDClassifier

# SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
# SGDClassifier_classifier.train(training_set)
# print("SGDClassifier accuracy: ", (nltk.classify.accuracy(SGDClassifier_classifier, testing_set))*100)

# # SVC 

# SVC_classifier = SklearnClassifier(SVC())
# SVC_classifier.train(training_set)
# print("SVC accuracy: ", (nltk.classify.accuracy(SVC_classifier, testing_set))*100)

# # LinearSVC

# LinearSVC_classifier = SklearnClassifier(LinearSVC())
# LinearSVC_classifier.train(training_set)
# print("LinearSVC accuracy: ", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)

# # NuSVC

# NuSVC_classifier = SklearnClassifier(NuSVC())
# NuSVC_classifier.train(training_set)
# print("NuSVC accuracy: ", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)



# --------------------------------------------

# Lesson 16 - Combining Algos with a Vote



import random
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
import pickle

from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

from nltk.classify import ClassifierI
from statistics import mode

class VoteClassifier(ClassifierI):
	def __init__(self, *classifiers):
		self._classifiers = classifiers 

	def classify(self, features):
		votes = []
		for c in self._classifiers:
			v = c.classify(features)
			votes.append(v)
		return mode(votes)

	def confidence(self, features):
		votes = []
		for c in self._classifiers:
			v = c.classify(features)
			votes.append(v)

		choice_votes = votes.count(mode(votes))
		conf = choice_votes / len(votes)
		return conf

documents = [(list(movie_reviews.words(fileid)), category)
			for category in movie_reviews.categories()
			for fileid in movie_reviews.fileids(category)]


random.shuffle(documents)
 
all_words = []

for w in movie_reviews.words():
	all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)

word_features = list(all_words.keys())[:3000]

def find_features(document):
	words = set(document)
	features = {}
	for w in word_features:
		features[w] = (w in words)

	return features

featuresets = [(find_features(rev), category) for (rev, category) in documents]

training_set = featuresets[:1900]
testing_set = featuresets[1900:]

# posterior = prior occurrences x likelihood / evidence

classifier = nltk.NaiveBayesClassifier.train(training_set)

# Load it afterward

classifier_f = open("naivebayes.pickle", "rb")
classifier = pickle.load(classifier_f)
classifier_f.close()


print("Naive Bayes Algo accuracy: ", (nltk.classify.accuracy(classifier, testing_set))*100)
classifier.show_most_informative_features(15)

# Multinomial 
MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MNB_classifier accuracy: ", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)

# Gaussian
# Error regarding a sparse matrix being passed instead of a dense one

# GaussianNB_classifier = SklearnClassifier(GaussianNB())
# GaussianNB_classifier.train(training_set)
# print("GaussianNB accuracy: ", (nltk.classify.accuracy(GaussianNB_classifier, testing_set))*100)

# Bernoulli

Bernoulli_classifier = SklearnClassifier(BernoulliNB())
Bernoulli_classifier.train(training_set)
print("BernoulliNB accuracy: ", (nltk.classify.accuracy(Bernoulli_classifier, testing_set))*100)


# LogisticRegression

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression accuracy: ", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)

# SGDClassifier

SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
print("SGDClassifier accuracy: ", (nltk.classify.accuracy(SGDClassifier_classifier, testing_set))*100)

# SVC 

SVC_classifier = SklearnClassifier(SVC())
SVC_classifier.train(training_set)
print("SVC accuracy: ", (nltk.classify.accuracy(SVC_classifier, testing_set))*100)

# LinearSVC

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC accuracy: ", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)

# NuSVC

NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print("NuSVC accuracy: ", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)


voted_classifier = VoteClassifier(classifier, 
									MNB_classifier, 
									Bernoulli_classifier, 
									LogisticRegression_classifier, 
									SGDClassifier_classifier, 
									LinearSVC_classifier, 
									NuSVC_classifier)
print("Voted Classifier accuracy: ", (nltk.classify.accuracy(voted_classifier, testing_set))*100)

print("Classification: ", voted_classifier.classify(testing_set[0][0]), "Confidence %: ", voted_classifier.confidence(testing_set[0][0]))
print("Classification: ", voted_classifier.classify(testing_set[1][0]), "Confidence %: ", voted_classifier.confidence(testing_set[1][0]))
print("Classification: ", voted_classifier.classify(testing_set[2][0]), "Confidence %: ", voted_classifier.confidence(testing_set[2][0]))
print("Classification: ", voted_classifier.classify(testing_set[3][0]), "Confidence %: ", voted_classifier.confidence(testing_set[3][0]))
print("Classification: ", voted_classifier.classify(testing_set[4][0]), "Confidence %: ", voted_classifier.confidence(testing_set[4][0]))
print("Classification: ", voted_classifier.classify(testing_set[5][0]), "Confidence %: ", voted_classifier.confidence(testing_set[5][0]))


# --------------------------------------------

# Lesson 17 - Investigating bias


