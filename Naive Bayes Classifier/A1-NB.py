#Aditya Agarwal | Manisha Katariya | Jalaj Bansal
#Assignment - 1B: Naïve – Bayes Classifier


import pandas as pd
import numpy as np

data = pd.read_csv('dataset_NB.txt', sep='\t', header=None, names=['email', 'label'])

#data cleaning

data['label'] = data.apply(lambda row: row.email[-1], axis = 1)
data['email'] = data['email'].str.replace('\W', ' ') #Removes punctuation
data['email'] = data['email'].str.replace('\d+', ' ') #Removes numbers
data['email'] = data['email'].str.lower() #convert to lowercase
data['email'] = data['email'].str.split()
stopwords = ['ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than']

#making the vocabulary
vocabulary = []
for email in data['email']:
	for word in email:
	    vocabulary.append(word)

vocabulary = list(set(vocabulary))
n_vocabulary = len(vocabulary)

#removing stopwords from vocab
vocabulary = [word for word in vocabulary if word not in stopwords]
	

#creating new dataframe in desired format
#column are individual words and rows are emails; if word belongs in email, value = 1 else 0
word_counts_per_email = {unique_word: [0] * len(data['email']) for unique_word in vocabulary}
for index, email in enumerate(data['email']):
   for word in email:
   	if word not in stopwords:
   		word_counts_per_email[word][index] += 1
word_counts = pd.DataFrame(word_counts_per_email)
#final dataset
training_set_clean = pd.concat([data, word_counts], axis=1)


#function called during every iteration of 7-fold cross validation
def fit(training_set_clean, testing_set, starting_row, vocabulary):

	stopwords = ['ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than']

	n_vocabulary = len(vocabulary)

	#calculating constants to train model
	spam_messages = training_set_clean[training_set_clean['label'] == '1']
	ham_messages = training_set_clean[training_set_clean['label'] == '0']

	p_spam = len(training_set_clean[training_set_clean['label'] == '1']) / len(training_set_clean)
	p_ham = len(training_set_clean[training_set_clean['label'] == '0']) / len(training_set_clean)

	n_words_per_spam_message = spam_messages['email'].apply(len)
	n_spam = n_words_per_spam_message.sum()

	n_words_per_ham_message = ham_messages['email'].apply(len)
	n_ham = n_words_per_ham_message.sum()
	#laplace smoothing parameter
	alpha = 1.0

	# Initialising parameters
	parameters_spam = {unique_word:0 for unique_word in vocabulary}
	parameters_ham = {unique_word:0 for unique_word in vocabulary}

	# Calculate parameters
	for word in vocabulary:
	   if word == 'email': continue	
	   if word == 'label': continue

	   if word in spam_messages: n_word_given_spam = spam_messages[word].sum()
	   else: n_word_given_spam = 0 # spam_messages already defined 
	   p_word_given_spam = (n_word_given_spam + alpha) / (n_spam + alpha*n_vocabulary)
	   parameters_spam[word] = p_word_given_spam

	   if word in ham_messages: n_word_given_ham = ham_messages[word].sum() # ham_messages already defined
	   else: n_word_given_ham = 0
	   p_word_given_ham = (n_word_given_ham + alpha) / (n_ham + alpha*n_vocabulary)
	   parameters_ham[word] = p_word_given_ham
	
	#testing the model
	y_same = 0 #holds the number of correctly classified samples
	for index, email in enumerate(testing_set['email']):
		p_spam_given_message = p_spam
		p_ham_given_message = p_ham
		for word in email:
			
			if word in parameters_spam:
				p_spam_given_message *= parameters_spam[word]
			if word in parameters_ham:
				p_ham_given_message *= parameters_ham[word]
		if p_ham_given_message > p_spam_given_message:
			if testing_set.loc[index+starting_row, 'label'] == '0': y_same+=1 #if sample is correctly classified as ham
		else:
			if testing_set.loc[index+starting_row, 'label'] == '1': y_same+=1 #if sample is correctly classified as spam
	#returning accuracy
	return (y_same / (1000/7)) 

#initialising accuracy
accuracy = 0.0

for i in range(7):
	#splitting the data into 7 folds and looping over 7 times
	testingset = training_set_clean.iloc[int(i * 1000 / 7): int((i+1) * 1000 / 7), ]
	training_set = training_set_clean.drop(training_set_clean.index[int(i * 1000 / 7): int((i+1) * 1000 / 7)]) #.drop(table.index[:2], inplace = True) 
	#finding the vocabulary for each training set
	vocabulary = []
	for email in training_set['email']:
		for word in email:
			vocabulary.append(word)
	vocabulary = list(set(vocabulary))

	value = fit(training_set, testingset, int(i * 1000 / 7), vocabulary)
	print("Fold ", i, ": ",value)
	accuracy += value/7

print("Final Accuracy: ", accuracy)
