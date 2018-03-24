# Tutorial bz SentDex creator here : https://www.youtube.com/watch?v=FLZvOKSCkxY&list=PLQVvvaa0QuDf2JswnfiGkliBInZnIC4HL

import nltk


# Lesson 19 - Sentiment Analysis Module

import random
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
import pickle

from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

from nltk.classify import ClassifierI
from statistics import mode

from nltk.tokenize import word_tokenize

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

short_pos = open("short_reviews/positive.txt", "r").read()
short_neg = open("short_reviews/negative.txt", "r").read()

all_words = []
documents = []

# j for adjective, r adverb, v verb
# allowed_word_types = ["J", "R", "V"]
allowed_word_types = ["J"]

for r in short_pos.split('\n'):
	documents.append((r, "pos"))
	words = word_tokenize(r)
	pos = nltk.pos_tag(words)
	for w in pos:
		if w[1][0] in allowed_word_types:
			all_words.append(w[0].lower())

for r in short_neg.split('\n'):
	documents.append((r, "neg"))
	words = word_tokenize(r)
	pos = nltk.pos_tag(words)
	for w in pos:
		if w[1][0] in allowed_word_types:
			all_words.append(w[0].lower())

save_documents = open("picked_algorithms/documents.pickle", "wb")
pickle.dump(documents, save_documents)
save_documents.close()

all_words = nltk.FreqDist(all_words)

word_features = list(all_words.keys())[:5000]

save_word_features = open("picked_algorithms/word_features5k.pickle", "wb")
pickle.dump(word_features, save_word_features)
save_word_features.close()

def find_features(document):
	words = word_tokenize(document)
	features = {}
	for w in word_features:
		features[w] = (w in words)

	return features

featuresets = [(find_features(rev), category) for (rev, category) in documents]

save_featuresets = open("picked_algorithms/featuresets.pickle", "wb")
pickle.dump(featuresets, save_featuresets)
save_featuresets.close()

random.shuffle(featuresets)

# print(len(featuresets))

# Positive data example

testing_set = featuresets[10000:]
training_set = featuresets[:10000]

# posterior = prior occurrences x likelihood / evidence

classifier = nltk.NaiveBayesClassifier.train(training_set)
print("Naive Bayes Algo accuracy: ", (nltk.classify.accuracy(classifier, testing_set))*100)
classifier.show_most_informative_features(15)

save_classifier = open("picked_algorithms/naivebayes5k.pickle", "wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()


# Multinomial 

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MNB_classifier accuracy: ", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)

save_MNB_classifier = open("picked_algorithms/MNB_classifier5k.pickle", "wb")
pickle.dump(MNB_classifier, save_MNB_classifier)
save_MNB_classifier.close()


# Bernoulli

Bernoulli_classifier = SklearnClassifier(BernoulliNB())
Bernoulli_classifier.train(training_set)
print("BernoulliMNB accuracy: ", (nltk.classify.accuracy(Bernoulli_classifier, testing_set))*100)

save_Bernoulli_classifier = open("picked_algorithms/Bernoulli_classifier5k.pickle", "wb")
pickle.dump(Bernoulli_classifier, save_Bernoulli_classifier)
save_Bernoulli_classifier.close()


# LogisticRegression

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression accuracy: ", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)

save_LogisticRegression_classifier = open("picked_algorithms/LogisticRegression_classifier5k.pickle", "wb")
pickle.dump(LogisticRegression_classifier, save_LogisticRegression_classifier)
save_LogisticRegression_classifier.close()


# SGDClassifier

SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
print("SGDClassifier accuracy: ", (nltk.classify.accuracy(SGDClassifier_classifier, testing_set))*100)

save_SGDClassifier_classifier = open("picked_algorithms/SGDClassifier_classifier5k.pickle", "wb")
pickle.dump(SGDClassifier_classifier, save_SGDClassifier_classifier)
save_SGDClassifier_classifier.close()


# SVC 

SVC_classifier = SklearnClassifier(SVC())
SVC_classifier.train(training_set)
print("SVC accuracy: ", (nltk.classify.accuracy(SVC_classifier, testing_set))*100)

save_SVC_classifier = open("picked_algorithms/SVC_classifier5k.pickle", "wb")
pickle.dump(SVC_classifier, save_SVC_classifier)
save_SVC_classifier.close()


# LinearSVC

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC accuracy: ", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)

save_LinearSVC_classifier = open("picked_algorithms/LinearSVC_classifier5k.pickle", "wb")
pickle.dump(LinearSVC_classifier, save_LinearSVC_classifier)
save_LinearSVC_classifier.close()


# NuSVC

NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print("NuSVC accuracy: ", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)

save_NuSVC_classifier = open("picked_algorithms/NuSVC_classifier5k.pickle", "wb")
pickle.dump(NuSVC_classifier, save_NuSVC_classifier)
save_NuSVC_classifier.close()


# Voted classifier 

voted_classifier = VoteClassifier(classifier, 
									MNB_classifier, 
									Bernoulli_classifier, 
									LogisticRegression_classifier, 
									SGDClassifier_classifier, 
									LinearSVC_classifier, 
									NuSVC_classifier)
print("Voted Classifier accuracy: ", (nltk.classify.accuracy(voted_classifier, testing_set))*100)


def sentiment(text):
	feats = find_features(text)
	return voted_classifier.classify(feats)