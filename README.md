# Data Challenge - Text Classification 
* (This code and report is part of my assessment during Master's for unit Applied Data Science) *

Download data here - https://drive.google.com/drive/folders/1d60aP0KjoC_Xj0euvCWqLLkCeyzWhw29?usp=sharing

Data Analysis Task
With the enormous evolution in data since last few decades, text classification is key in organizing and handling text data. Text classification can be used for news stories classification, search information on the internet and guide a user’s search through hypertext. This report will guide through how the task of text classification has been done.
## R Libraries Used 
● caret 
● e1071
## Pre-processing steps 
The most intense step of knowledge discovery is pre-processing of data. It depends on the data source being used. The main aim of pre-processing data is to achieve sequential patterns in documents. Enormous amount of data can have a low prediction value thus data needs pre-processing to increase its value and attain better outcomes.
We have been provided with three different files, the first file is training_docs which has the data which will be used to train and test our model, it also has a unique document id for each document, the second file is final_training_labels which has the classes for all the text which has been provided in the first document in the same order as the text in the first document(training_docs).
All the pre-processing has been done using Python. The three text files are read and stored as a string. A data frame is created in which the training_docs are stored, the first column has the ID and the other column has text, all the unnecessary text such as ID, TEXT, EOD has been removed. The same thing done is done for testing_docs and stored in a data frame and the two data frames are concatenated.
In the next step we lemmatize the text in the data frame, stop words are removed and all the text is changed into lower case. After which all the text is tokenized, and the data is stripped of all numbers and punctuations. After tokenizing the data, we have found n-grams for the data and found unigrams, bi-grams and tri-grams and stored them in different columns.
## Feature selection 
The process of choosing a subset of words present in the training set and applying this set of words as features is called feature selection. 2 important problems are solved by doing so. Number one it reduces the size of the effective vocabulary and secondly it helps in eliminating or reducing noise features. Selecting features which do not help in classification usually end up in overfitting of the model on the training set.Our features have been selected using frequency-based feature section, the words that are the most
common in a class are selected as features. Frequency-based frequency selection can select few features that may be frequent but does not provide any information about any specific class, this method is used for selection of thousands of features. It is a good substitute for complex methods. 
### How we sparsed By finding n-grams features i.e. applying feature engineering:
• trigrams - words that occur in cohort of 3 words and make verb, noun, adjective that makes sense when clubbed together. Hence, if they are together in a corpus they make distinguish pattern which helps in classification of group of corpora. We took the trigrams that occurred more than 1064 times in the whole corpus. There are total 12 trigrams that we took in our feature selection.
Some of them were:
('prime', 'minister', 'john')
('minister', 'john', 'howard')
('south', 'wale', 'government')
• bigrams - words that occur in cohort of 2 words and make sense in only when together and helps in classification of corpus. We took the bigrams that occurred more than 2112 and less than 104313 times in corpora. There are total 95 bigrams in our features. Some of the most common bigrams were:
('northern', 'territory')
('western', 'australia')
('south', 'wale')
• unigrams: single word that makes distinct feature in identifying the corpora. We selected unigrams that occurrence of more than 2539 and less than 103600 in the whole corpora. There were total 1219 unigrams. 

### Method used to develop model We used support vector machine to classify our data. Why we used this algorithm:
#### 1. High Dimension input space: When learning text classifier, one has to deal with very many (1000) features. Since, SVMs use overfitting protection, which does not necessarily depend on the number of features. They have potential to handle these large feature space.
#### 2. Document vectors are sparse: For each document, the corresponding document vector contains only few entries which are non-zero. For a mistake bound model the “additive” algorithms, which have a similar inductive bias like SVMs are well suited for problems with dense concepts and sparse instances.
## Description of the model used for learning 
Since last two decades, enhancing classification of machine learning techniques has been very important. The development has led to the creation of support vector machines which are a state of the art classifier. A support vector machine is vector space-based machine learning method where the end goal is to identify the decision boundary between 2 classes that is maximally far from any point in the training data which also helps in eradicating any outliers and noise. SVMs work on the principle of Structural Risk Minimization principle from computational learning theory, this principle asks to fund a hypothesis h for which we can promise the minimum test error. The actual error of h is the probability that h on the training set and the complexity of H which is measured by VC-Dimension, the hypothesis space containing h. SVM find the hypothesis h which approximately minimizes this bound on true error by effectually and competently controlling the VC-Dimension of H.
Support vector machines provide good results for text classification as it can provide a high dimensional input space, text classifiers may have to use more than ten thousand features, but support vector machines provide overfitting protection which does is not affected by the number of variables or features, support vector machines can lever large variable spaces. Many features in the high dimensional space are immaterial, these irrelevant features are identified by feature selection. If a classifier takes input of the unrequired features performs better than just taking random classifiers.
A document vector usually consists of only few entries which are not zero. There are theoretical and empirical proves that support vector machines can handle issues with dense concepts and sparse instances. A lot of text classification problems are linearly separable, support vector machines can find this linear or polynomial separators. Support vector machines take almost same time as C4.5 in training the model but when compared to Naïve Bayes and k-NN they are more expensive when it comes to time, but research is being done in this area to enhance support vector machines.
##### After training the support vector machine for the training data, we used this trained model to predict the labels for the test data taken from the training data and came up with an accuracy of 74.36%. Recall 0.74 and F-score 0.75.
