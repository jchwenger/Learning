# Tutorial 18 : https://www.youtube.com/watch?v=JELNYAAaWyU&list=PLcTXcpndN-Sl9eYrKM6jtcOTgC52EJnqH&index=18

import os, os.path

mypath = os.path.expanduser('~/bin/Python/nltk/nltk_data')

if not os.path.exists(mypath):
	os.mkdir(mypath)
	print("folder has been created")
else:
	print("folder already exists")

import nltk.data 

print(nltk.data.path)

varbool = mypath in nltk.data.path
print(varbool)

newfile = open('../nltk_data/wow', 'r')

print(newfile.read())
