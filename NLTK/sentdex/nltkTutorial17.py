# Tutorial bz SentDex creator here : https://www.youtube.com/watch?v=FLZvOKSCkxY&list=PLQVvvaa0QuDf2JswnfiGkliBInZnIC4HL

import nltk

# Lesson 17 - Investigating bias

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


# random.shuffle(documents)
 
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

# Positive data example
training_set = featuresets[:1900]
testing_set = featuresets[1900:]

# Negative data example
training_set = featuresets[100:]
testing_set = featuresets[:100]

# posterior = prior occurrences x likelihood / evidence

# classifier = nltk.NaiveBayesClassifier.train(training_set)

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

# print("Classification: ", voted_classifier.classify(testing_set[0][0]), "Confidence %: ", voted_classifier.confidence(testing_set[0][0]))
# print("Classification: ", voted_classifier.classify(testing_set[1][0]), "Confidence %: ", voted_classifier.confidence(testing_set[1][0]))
# print("Classification: ", voted_classifier.classify(testing_set[2][0]), "Confidence %: ", voted_classifier.confidence(testing_set[2][0]))
# print("Classification: ", voted_classifier.classify(testing_set[3][0]), "Confidence %: ", voted_classifier.confidence(testing_set[3][0]))
# print("Classification: ", voted_classifier.classify(testing_set[4][0]), "Confidence %: ", voted_classifier.confidence(testing_set[4][0]))
# print("Classification: ", voted_classifier.classify(testing_set[5][0]), "Confidence %: ", voted_classifier.confidence(testing_set[5][0]))