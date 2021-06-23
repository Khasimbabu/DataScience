#*******************
#******************
# Text Mining
#*****************

-> An exploration and analysis of textual(natural language) data to identify facts, relationships and assertions.
-> Process of analysing collections of textual materials in order to capture key concepts and themes and uncover hidden relationships and trends

Examples: 
Classical Spam Filtering
Review Analysis
Tweets Analysis

Natural Language Processing(NLP)
NLP is based on latent features of the text and uses those in its methods and text mining is based on observed features.
NLP considers the context in the text, while text mining does not.

Text mining sources:
Comments(Social networking sites)
Tweets
Sales Reports
Emails
Blogs
Word Documents

Types of documents in Text mining:
Structured documents(survey forms,claims)
Semi Structured documents(Job listings, Invoices)
Unstructured documents(blogs)

#******************
# NLP

#corpus is a collection of texts
#corpora is a plural of corpus

# in cmd prompt
# pip install nltk (or) !conda install nltk

import nltk

# some names of the corpus provided in the nltk module
# Gutenberg corpus - import format --> nltk.corpus.gutenberg
# Web and chat text - import format --> nltk.corpus.webtext
# NPS chat - import format --> nltk.corpus.nps_chat
# Brown corpus - import format --> nltk.corpus.brown
# Reuters corpus - import format --> nltk.corpus.reuters
# Inaugural Address Corpus - import format --> nltk.corpus.inaugural

from nltk.corpus import gutenberg as gt
print(gt.fileids)
# gives all the files that the corpus has

#words func
shkspr_hmlt = gt.words('shakespeare-hamlet.txt')
print(shkspr_hmlt, end='\n\n\n')
# gives all the words

print(len(shkspr_hmlt))

shkspr_hmlt = gt.raw('shakespeare-hamlet.txt')
print(shkspr_hmlt, end='\n\n\n')
# gives the raw text

# functions - raw() - words() - sents()

for fileid in gt.fileids():
			# raw_data = gt.raw(fileid)
			num_words = len(gt.words(fileid))
			num_sents = len(gt.sents(fileid))
			# vocabulory = set([w.lower() for w in gt.words(fileid)])
			print('Data for File Id :', fileid)
			print('Number of words :',num_words,'\nNumber of sentences:',num_sents)
			#print('Vocabulary:\n',vocabulory, end='\n\n\n')
			print('Words :',gt.words(fileid), end='\n\n\n')

from nltk.corpus import PlaintextCorpusReader
# download plays in txt format of shakespeare
# http://www.textfiles.com/etext/AUTHORS/SHAKESPEARE/
import os
#corpus_root = os.getcwd() + '/'
corpus_root = 'C:\Users\kshaik\Documents\Khasim2020\Khasim2021\DataScience\'
corpus_root

file_ids = '.*.txt'

corpus = PlaintextCorpusReader(corpus_root, file_ids)
print(corpus.fileids())
# This will take a long time if we have a lot of files

print(corpus.words('shakespeare-taming-of-the-shrew.txt'))
print(corpus.raw('shakespeare-taming-of-the-shrew.txt'))

#**************
import nltk
from nttk.corpus import gutenberg as 
from nltk.corpus import plaintextCorpusReader
import os
from nltk.tokenize import word_tokenize, sent_tokenize

def read_file(filename):
	with open(filename, 'r') as file:
		text=file.read()
	return text
	
import os
os.chdir('C:\Users\kshaik\Documents\Khasim2020\Khasim2021\DataScience\'))
text = read_file(shakespeare-taming-of-the-shrew.txt')
text[:100]

#words_tokenize - inputs --> string containing the text
#				- outputs --> list of words

words = word_tokenize(text)
print('size as a list: ', len(words))
print('size as a set: 	', len(set(words)))
print(words[:100])

#sent_tokenize - inputs --> string containing the text
#				- outputs --> list of sentences

sentences = sent_tokenize(text)
print('No of sentences: ', len(sentences),'\n')
print('size as a set: 	', len(set(words)))
for sentence in sentences[:5]:
	print(sentence.strip())
print('Size as a list :',len(sentences))
print('Size as a set :',len(set(sentences)), end='\n\n')
print(sentences[:10])

import re	# regular expression

# search function
'''
if((re.search('^a','abc'))):
	print('Found it !')

re.search('^a','abc')
re.search('^a','Abc')

if((re.search('^a','Abc'))):
	print('Found it !')
else:
	print('Not Found')

print(len(set([w for w in words if re.search('^a+',w)])))
print(set([w for w in words if re.search('^a+',w)]))
print([w for w in words if re.search('^at+',w)])
print([w for w in words if re.search('^at*',w)])
print([w for w in words if re.search('^a*t*',w)])
print([w for w in words if re.search('^a+t+',w)])
print([w for w in words if re.search('^a*t+',w)])
'''

from nltk.corpus import gutenberg, brown, nps_chat
# import nltk

moby = nltk.Text(gutenberg.words('melville-moby_dick.txt'))
print(moby.findall(r'<a><.*><man>'))

#a __ man # 3 tokens each word is a tokenize

# Creating a Text Object
chat_obj = nltk.Text(nps_chat.words())
print(chat_obj.findall(r'<.*><.*><bro>'))

hobbies_learned = nltk.Text(brown.words(categories=['hobbies','learned']))
print(hobbies_learned.findall(r'</w*><and><other><\w*s>'))

# passing your own list of words
text = 'Hello, I am an electrical engineer, who is currently learning DataScience'

obj = nltk.Text(nltk.word_tokenize(text))
print(obj.findall(r'<.*ing>+'))

#****************
#Stemming & Lemmatization
#***********

import nltk
from nltk.corpus import gutenberg as gt
from nltk.corpus import PlaintextCorpusReader
import os
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import PorterStemmer, LancasterStemmer

#tokens = nltk.corpus.brown.words(categories=['romance'])
#off the shelf stemmers
# The porter stemmer is a good choice if you are indexing some texts and want to support search using alternative forms of words
# crying --> cry

#Stemming does not do a dictionary lookup but lemmatization does

porter = PorterStemmer()
# stem() function - takes a token as input

tokens=['lying']
print(porter.stem(tokens[0]))	#gives 'lie'

lancaster = LancasterStemmer()
print(lancaster.stem(token[0]))	# gives 'lying'

from nltk import WordNetLemmatizer
from nltk.corpus import brown

# import nltk
# Difference between Stemming & Lemmatization
# Stemming does not do a dictionary lookup  but lemmatization does
# We are normalizing the words as we cannot do every word we use Lemma

tokens = brown.words(categories=['religion'])
wnl = WordNetLemmatizer()
print(set([wnl.lemmatize(t) for t in tokens]))

tokens = nltk.word_tokenize('the women are not lying')	#women --> woman
tokens

print([wnl.lemmatize(t) for t in tokens]}

tokens = nltk.word_tokenize('the children are not lying')
print([wnl.lemmatize(t) for t in tokens])


#****************************************************










																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																										







