# Tutorial playlist here: https://www.youtube.com/watch?v=OrJqt8zHTX4&list=PLcTXcpndN-Sl9eYrKM6jtcOTgC52EJnqH&index=4

from nltk.corpus import stopwords 

ensw = stopwords.words('english')

print(ensw)
print(len(ensw))

from nltk.tokenize import word_tokenize

parag1 = "WASHINGTON — In his first full cabinet meeting last June, President Trump invited a chorus of gushing praise from his top aides by boasting that he had assembled a “phenomenal team of people, a great group of talent.” \
But in the nine months since then, Mr. Trump has fired or forced out a half-dozen of the “incredible, talented” people in the Cabinet Room that day: his secretaries of state and health, along with his chief strategist, his chief of staff, his top economic aide and his press secretary. \
And the purge at the top may not be over. Mr. Trump, who is famously fickle, appears to have soured on additional members of his senior leadership team — and his frequent mulling about making changes has some people around him convinced that he could act soon. \
“There will always be change. I think you want to see change,” Mr. Trump said, ominously, on Thursday. “I want to also see different ideas.”"

# Notice the difference in the final paragraphs's number of words 
# if the text has been all processed into lower case

# paragArr = word_tokenize(parag1)

paragArr = word_tokenize(parag1.lower())

print(paragArr)
print(len(paragArr))

filterArr = [item for item in paragArr if item not in ensw]

print(filterArr)
print(len(filterArr))

