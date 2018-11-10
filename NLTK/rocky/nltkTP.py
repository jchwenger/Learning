# Tutorial playlist here: https://www.youtube.com/watch?v=IqPWZL5f-7g&list=PLcTXcpndN-Sl9eYrKM6jtcOTgC52EJnqH

import nltk 

parag = "WASHINGTON — In his first full cabinet meeting last June, President Trump invited a chorus of gushing praise from his top aides by boasting that he had assembled a “phenomenal team of people, a great group of talent.” \
But in the nine months since then, Mr. Trump has fired or forced out a half-dozen of the “incredible, talented” people in the Cabinet Room that day: his secretaries of state and health, along with his chief strategist, his chief of staff, his top economic aide and his press secretary. \
And the purge at the top may not be over. Mr. Trump, who is famously fickle, appears to have soured on additional members of his senior leadership team — and his frequent mulling about making changes has some people around him convinced that he could act soon. \
“There will always be change. I think you want to see change,” Mr. Trump said, ominously, on Thursday. “I want to also see different ideas.”"

from nltk.tokenize import sent_tokenize

tokenized_parag = sent_tokenize(parag)

print(tokenized_parag)
print(len(tokenized_parag))
# print('\n'+tokenized_parag[2]+'\n')
# print(tokenized_parag[5])

import nltk.data

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

for w in tokenizer.tokenize(parag):
	print(w+'\n')

arrayT = tokenizer.tokenize(parag)

print(arrayT[5])
