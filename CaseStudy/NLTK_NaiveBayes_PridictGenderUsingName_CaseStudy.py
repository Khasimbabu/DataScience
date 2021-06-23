#*******************
# NLP CaseStudy
#*****************

NLP (Natural Language Processing) = Is an automated way to understand and analyze natural human languages and extract information from such data by applying machine algorithms.

- Analyzing tons of data
- Identifying various languages and dialects
- Applying quatitative analysis
- Handling ambiguities

NLP Terminology
-Tokenization = split
-Stemming = Map to valid root word
-Tf-Idf = Term frequency-inverse document frequency
-Semantic analytics = compare words/pharases...
-Disambiguation = determines meaning and sense of words = context vs intent
-Topic models = discover topics in a collection of documents
-word bounderies = Determine which one word ends and the other begins

NLP approach for Text Data
Basic Text processing -> Categorize & Tag words -> classify text -> extract info -> analyze sentence structure -> build feature based structure -> Analyze the meaning

NLP Libraries
- NLTK
- Scikit-learn
- TextBlob
- spaCy

Feature Extraction = Is a technique to convert the content into the numerical vectors to perform machine learning

Bag of words(BOW) = Each word how many times appearing = Each word has some weight(its number) = Is used to convert text data into numerical feature vectors with a fixed size.

Tokenizing -> Counting -> Store

Text Summarization:
1) convert the sentences to lower case
2) Building the bag of words - Tokenize and count each word occurences
3) Take the word count value, which has highest number count & devide that value for all the values
& you get weight for each word.
4) For each line calculate (sum) the weight. The line with highest weightage is the Text summerization.

word2vector = In this model, each word is represented as vector of 32 or more dimentions
instead of single number

word2vector = gives symantic/similar meaning = Its a deep learning technique for extracting the sentence meaning.

word2vector represent the words in multiple directions/dimentions

Relation between different words is preserved

Overfitting a senario = when the model performs very well with your dataset but fails when applied to any new dataset

#***********************
# Naive Bayes Classifier
#*************************

- It is efficient as it uses limited CPU and memory
- It is fast as Model training takes less time
- Works well with noisy & missing data
- It uses principles of probabilities
- Works good for categorical data but not good with large numeric features
- simple and fast
 
- Used for Sentiment Analysis, Email spam detection, Language detection, categorization of documents, Medical diagnosis 
- MultiNomial NaiveBayes is when multiple occurences of word matter


Theorem: Gives the conditional probability of an event A, given event B has already occured.

P(A) = Probability of an event Analysis = value between 0 & 1

P(A/B) = P(A intersect B) * P(A)/P(B)

Target variable = A = dependent variable
Predictors becomes events = B1 to Bn

-> You give the data, Algorithm will generate a Table with different probabilities.


#**********************************************************	

# Build a small application which will take input a name of a person and 
# tell/predict if the name is female or male
# choosing features which our algorithm will use to identify
# whether the person is a male or female

def gender_features_part1(name):
	name = str(name).lower()
	return {'last_letter': name[-1:]}
print(gender_features_part1('sam'))
print(gender_features_part1('Dara'))

import nltk
from nltk.corpus import gutenberg as gt
from nltk.corpus import PlaintextCorpusReader
import os
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk import PorterStemmer, LancasterStemmer

# sample of names using the nltk built-in module
from nltk.corpus import names as names_sample
import random
names = [(name,'male') for name in names_sample.words('male.txt')] + [(name,'female') for name in names_sample.words('female.txt')]

print(names)

# run multy times v get shuffled names to reduce bias 
random.shuffle(names)

for name,gender in names[:20]:
	print('Name: ', name, ' Gender:',gender)
	
# make a feature set for all the names
feature_sets = [(gender_features_part1(name.lower()), gender)
								for name, gender in names]

for dict,gender in feature_sets[:20]:
		print(dict,gender)

print(len(feature_sets))

# Making a testing dataset and training dataset
train_set = feature_sets[3000:]
test_set = feature_sets[:3000]

# Now we use the Naive Bayes classifier and train it using the train set
import nltk
classifier = nltk.NaiveBayesClassifier.train(train_set)

# Now we test it against names 
gender_features_part1('Samy')
print(classifier.classify(gender_features_part1('Samy')))
print(classifier.classify(gender_features_part1('Sam')))

# Testing the accuracy of our classifier
print(nltk.classify.accuracy(classifier, test_set)*100)

# nltk.classify.accuracy(ML Classifier, testing data set)

#
# show_most_informative_features function
# no of features we want to see - default value of 10

print(classifier.show_most_informative_features())  # it will build a Table of probabilities

# last letter = 'a'     female : male = 33.2 : 1.0
# for every 33 females whose name ends with a, there is 1 male ending with a
# last letter = 'k'     male : female = 22.7 : 1.0


#********************
# Build a complex App with complex features

class GenderApp(object):
    def __init__(self):
        names_sample = nltk.corpus.names
        self.names = [(name.lower(), 'male') for name in names_sample.words('male.txt')] + [(name,'female') for name in names_sample.words('female.txt')]
        
        random.shuffle(self.names)
        
    @staticmethod
    def gender_features_cmplx(word):
        name = word.lower()     # normalise
        features = {}   # dictionary()
        features['first_letter'] = name[0]  # get first letter
        features['last_letter'] = name[-1]  # get last letter

        for letter in 'abcdefghijklmnopqrstuvwxyz':
            features['count' + letter] = name.count(letter)
        # how many times each letter occur in a name
            features['has' + letter] = letter in name
            # sam -->
            #counta = 1 and hasa = true and
            #countb = 0 and hasb = false
            # ea of this becomes a feature
        return features
        
    if __name__ == '__main__':
        app = GenderApp()
        print(GenderApp.gender_features_cmplx('Sam'))
        
#**********************
class GenderApp(object):
    def __init__(self):
        names_sample = nltk.corpus.names
        self.names = [(name.lower(), 'male') for name in names_sample.words('male.txt')] + [(name,'female') for name in names_sample.words('female.txt')]
        
        random.shuffle(self.names)
        self.feature_sets = [(GenderApp.gender_features_cmplx(name),gender) for name, gender in names] 
        self.train_set = self.feature_sets[:4000]
        self.test_set = self.feature_sets[4000:]
        self.classifier = nltk.NaiveBayesClassifier.train(self.train_set)
    
    def check_gender(self,name):
        name = name.lower()
        print('Gender for ' + name + ' : ' + self.classifier.classify(GenderApp.gender_features_cmplx(name)))
                
    @staticmethod
    def gender_features_cmplx(word):
        name = word.lower()     # normalise
        features = {}   # dictionary()
        features['first_letter'] = name[0]  # get first letter
        features['last_letter'] = name[-1]  # get last letter

        for letter in 'abcdefghijklmnopqrstuvwxyz':
            features['count' + letter] = name.count(letter)
        # how many times each letter occur in a name
            features['has' + letter] = letter in name
            # sam -->
            #counta = 1 and hasa = true and
            #countb = 0 and hasb = false
            # ea of this becomes a feature
        return features
        
    if __name__ == '__main__':
        app = GenderApp()
        app.check_gender('Sam')



# You can use the similar algorithm model for checking the Movie review...like many applications
# Spam/Ham emails prediction...Stock News and the prices variation prediction
# Concept is the same but the data is different












