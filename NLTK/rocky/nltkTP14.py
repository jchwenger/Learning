# Tutorial playlist: https://www.youtube.com/watch?v=9WOLqrDfM-Y&list=PLcTXcpndN-Sl9eYrKM6jtcOTgC52EJnqH&index=14

import re 

regex = re.compile(r'(\w*)(\w)\2(\w*)')

# The above regular expression will find the first occurrence
# of a repeating character in a word:
# This can be read in this way:
# (\w*): find any letter, zero or more times (* operator)
# (\w): find one letter
# \2: calls (\w) again, which finds a double character
# (\w*): again find any letter, any number of times
# In the following example, the computer will start by
# replacing the double 'i's by single ones, then go on
# by doing the same job with the 'a's

fw = 'dramaaaatiiiic'
sw = regex.sub(r'\1\2\3', fw)

print(sw)

def looper(word):
	loop_res = regex.sub(r'\1\2\3', word)
	if (word == loop_res):
		return loop_res
	else:
		# Too see the process, uncomment the line:
		# print(loop_res)
		return looper(loop_res)

sw = looper(fw)

print(sw)