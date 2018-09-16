# Imports

import re
import pandas as pd
import json
from collections import defaultdict
import numpy as np
import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn import metrics

###################################
## Create some utilities methods ##
###################################

# Function to filter valid tweets or not for this process
# Tweets that start with RT @ means a tweet reply
def isReplyTweet(tweet):
    if (re.search(r"(^([@]{1}[a-zA-Z0-9_]{3}))|(^([RT ]{3}[@]))+", tweet, re.IGNORECASE) != None):
        return True
    return False


# Function to filter the #
def isHasTag(tweet):
    if (re.search(r"(^[#]{1}[a-zA-Z0-9_]{3})+", tweet, re.IGNORECASE) != None):
        return True
    return False

# Create a function in order to check if the token is one of the special characters or not
def isSpecialChar(char):
    isSpecialChar = False
    #Define an array of special characteres
    specialChars = ['.', ',', ';', '...', '!', '?', ':', '/', '|', '\\', '&', '(', ')', '[', ']', '{', '}', '-', '_', '+', '=', '*', '%', '$', '@', '\'', '<', '>', '"']
    
    if char in specialChars:
        isSpecialChar = True
    
    return isSpecialChar


# Create a function in order to check if the token is one of the prepositions of the special words or not
def isUnnecessary(text):
    isUnnecessary = False
    #Define an array of special characteres
    isUnnecessaryText = ['a', 'the', 'of', 'it', 's', 're', 'are', 'is', 'll', 'be', 'to', 'i', 'all', 'do', 'does', 'did', 'will']
    
    if text in isUnnecessaryText:
        isUnnecessary = True
    
    return isUnnecessary


# Create a function in order to generate the matrix with 0 or 1 for the specific words
def isWordPresent(text, words):
    isPresent = False
    if text in words:
        isUnnecessary = True
    
    return isUnnecessary


def isEmojiPresent(text):
    # a_list = ['😤 😢 😭 😦 😧 😨 😩 🤯 😬 😰 😱😍 🤔 🙈 me así, bla es se 😌 ds 💕👭👙 🔪 🗡 ⚔️ ♾ 🏴‍☠️']
    if (len(re.findall(r'[^\w\s,]', text)) > 0):
        return True
    return False

def tokensHasEmoji(tokens):
    for token in tokens:
        if (isEmojiPresent(token)):
            return True
        
    return False

## Read the Dataset into a Panda Dataframe
# read the data

dataset = pd.read_json('../database/tweetsTest.json', lines=True)
#dataset = pd.read_json('../database/tweets.json', lines=True)

# load the data in a dataframe
data = pd.DataFrame(dataset)
data = data.dropna(subset=['user'])

##########################################
# Dictionary to group the tweets by user #
##########################################
userTweetsDict = dict()

for row in data.itertuples(index=True, name='Pandas'):
    user = getattr(row, "user")
    userId = user["_id"]['$numberLong']
    if userTweetsDict.get(userId) == None:
        tweets = list()
        if not (isReplyTweet(getattr(row, "text"))):
            tweets.append({"text": getattr(row, "text"), "isPositiveTweet": getattr(row, "isPositiveTweet")})
            userTweetsDict[userId] = tweets
    else:
        tweets = userTweetsDict.get(userId)
        if not (isReplyTweet(getattr(row, "text"))):
            tweets.append({"text": getattr(row, "text"), "isPositiveTweet": getattr(row, "isPositiveTweet")})
            userTweetsDict[userId] = tweets


#################
#  Tokenization #
#################

# Start the tokenization process, by creating an array of phrases. I mean, spliting by pontuation
for k, v in userTweetsDict.items():
    for i in range(0, len(v)):
        tweet = v[i].get("text")
        v[i]["text"] = sent_tokenize(tweet)

# Now get the tokenized phrases and performa word_tokenization, I mean split the phrases by word
for k, v in userTweetsDict.items():
    for i in range(0, len(v)):
        tweet = v[i].get("text")
        tokenized_docs = [word_tokenize(doc) for doc in tweet]

# Perform the lemmatization of the words
# Remove the special characteres from the twitter's text
lemmer=WordNetLemmatizer()

for k, v in userTweetsDict.items():
    for i in range(0, len(v)):
        tweet = v[i].get("text")
        tokenized_docs = [word_tokenize(doc) for doc in tweet]
        v[i]["text"] = tokenized_docs
        for j in range(0, len(tokenized_docs)):
            isUnnecessaryToRemove = list()
            specialCharsToRemove = list()
            for k in range(0, len(tokenized_docs[j])):
                if (isSpecialChar(tokenized_docs[j][k])):
                    specialCharsToRemove.append(tokenized_docs[j][k])
                    
                # lemmatize the word
                tokenized_docs[j][k] = lemmer.lemmatize(tokenized_docs[j][k].lower())
                if (isUnnecessary(tokenized_docs[j][k])):
                    isUnnecessaryToRemove.append(tokenized_docs[j][k])
            
            for specialChar in isUnnecessaryToRemove:
                while specialChar in tokenized_docs[j]:
                    tokenized_docs[j].remove(specialChar)
            
            for specialChar in specialCharsToRemove:
                while specialChar in tokenized_docs[j]:
                    tokenized_docs[j].remove(specialChar)
                    
            print(tokenized_docs[j])

        v[i]["text"] = tokenized_docs
        print(tokenized_docs)



#####################
# Matrix ############
#####################

#badWords = ["suicidal", "suicide", "kill", "myself", "note", "letter", "end", "life", "never", "wake", "jump", "sleep", "forever", "die", "dead", "pact", "tired", "living", "alone", "bullied", "bullyng"]
#positiveWords = ["happy", "happiness", "enjoy", "love", "news", "plans", "vacation", "live"]

#badWords = ["suicidal", "suicide", "kill", "myself", "end", "die", "dead", "pact", "living", "alone", "bullied", "bullyng"]
#positiveWords = ["happy", "happiness", "enjoy", "love", "news", "vacation", "live"]

badWords = ["suicidal", "suicide", "kill", "myself", "end", "die", "dead", "bullied", "bullyng"]
positiveWords = ["happy", "happiness", "enjoy", "love", "news", "live"]

matrix = []
for k, v in userTweetsDict.items():
    for i in range(0, len(v)):
        tweet = v[i].get("text")
        isPositiveTweet = v[i].get("isPositiveTweet")
        print(tweet)
        
        for tokens in tweet:
            vector = []
            vector.append(k)
            
            #if isPositiveTweet == True:
            for positiveWord in positiveWords:
                if positiveWord in tokens:
                    vector.append(1)
                else:
                    vector.append(0)
            #else:
            for badWord in badWords:
                if badWord in tokens:
                    vector.append(1)
                else:
                    vector.append(0)

            # Check if the tweet has emoji
            hasEmoji = tokensHasEmoji(tokens)
            
            if hasEmoji == True:
                vector.append(1)
            else:
                vector.append(0)
            
            if isPositiveTweet == True:
                #vector.append(1)
                #if hasEmoji == True:
                vector.append(0)
                #else:
                #    vector.append(1)
            else:
                #vector.append(0)
                if hasEmoji == True:
                    vector.append(0)
                else:
                    vector.append(1)
                
            matrix.append(vector)
            #for token in tokens:
            #    result = lemmer.lemmatize(token)
            #    token = result
            #    print(result)
    print("Matrix: " + str(matrix))

######################################################
## Re-create the Panda Dataframe only with the data ##
######################################################

# Create the matrix with only the values
dataValues = []
targetValues = []
for element in matrix:
    dataValues.append(element[1:])
    targetValues.append(element[-1])

columns = ["suicidal", "suicide", "kill", "myself", "end", "die", "dead", "bullied", "bullyng", "happy", "happiness", "enjoy", "love", "news", "live"] 
columns.extend(["hasEmoji", "isSuicidal"])
print(len(columns))

dataFrame = pd.DataFrame(dataValues)
dataFrame.columns = columns

############################
# Dimensionality Reduction #
############################
# Create correlation matrix
corr_matrix = dataFrame.corr().abs()
corr_matrix

# Execute the PCA in order to reduce the problem's dimension
pca = PCA(n_components=17)
pca.fit(dataFrame)
dataPCA = pca.transform(dataFrame)
print(pca.explained_variance_ratio_)
print(pca.singular_values_)

# remonta o DataFrame
dataFiltered = pd.DataFrame(dataPCA)
dataFiltered.head()

# Selects only the upper matrix's triangle 
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
# Find the coluns which the variance is lesser than 0.85
to_drop = [column for column in upper.columns if any(upper[column] > 0.75)]
dataFiltered = dataFrame.drop(dataFrame.columns[to_drop], axis=1)
dataFiltered['target'] = targetValues

##########################
## Let's train our model #
##########################
dataTrain, dataTest = train_test_split(dataFiltered, test_size=0.2)
# create an cross validation model
seed = 7
kfold = model_selection.KFold(n_splits=7, random_state=seed)

##########################
## Split the TRAIN DATA ##
##########################
nColumns = dataTrain.shape[1] - 1
trainValues = dataTrain.values
trainFeatures = trainValues[:,0:nColumns-1]
trainTargets = trainValues[:,nColumns]

#########################
## Split the TEST DATA ##
#########################
nColumns = dataTest.shape[1] - 1
testValues = dataTest.values
testFeatures = testValues[:,0:nColumns-1]
testTargets = testValues[:,nColumns]

#############################
## ENSEMBLE CLASSIFICATION ##
#############################
estimators = []
model1 = LogisticRegression()
estimators.append(('logistic', model1))
model2 = DecisionTreeClassifier()
estimators.append(('cart', model2))
model3 = SVC()
estimators.append(('svm', model3))

# create the ensemble
ensemble = VotingClassifier(estimators)
ensemble.fit(trainFeatures, trainTargets)
results = model_selection.cross_val_score(ensemble, trainFeatures, trainTargets, cv=kfold)
print('Essemble Accuracy: {0:0.2f}'.format(results.mean()))

#######################
# Essemble Prediction #
#######################
# perform the prediction under the test database
predictions = ensemble.predict(testFeatures)

# Evaluate the results
accuracy = accuracy_score(testTargets, predictions)
recall = recall_score(testTargets, predictions) 
precision = precision_score(testTargets, predictions) 
f1Score = f1_score(testTargets, predictions)

print('Essemble Prediction Accuracy: {0:0.2f}'.format(accuracy))
print('Essemble Prediction Recall: {0:0.2f}'.format(recall))
print('Essemble Prediction Precision: {0:0.2f}'.format(precision))
print('Essembke Prediction F1 score: {0:0.2f}'.format(f1Score))

# Confusion matrix
print(confusion_matrix(testTargets, predictions))
print(classification_report(testTargets, predictions))


# AUC
fpr, tpr, thresholds = metrics.roc_curve(testTargets, predictions)
roc_auc = metrics.auc(fpr, tpr)

print('Essemble AUC FPR:')
print(fpr)
print('Essemble AUC TPR:')
print(tpr)
print('Essemble AUC Thresholds:')
print(thresholds)
print('Essemble AUC ROC:')
print(metrics.auc(fpr, tpr))

######################
# SVM Classification #
######################
#clf = svm.SVC(kernel='linear', C=1)
#clf = clf.fit(dataFiltered, targetValues)
#scores = cross_val_score(clf, dataFiltered, targetValues, cv=5)

#print("SVM Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

##################
# SVM Prediction #
##################
# perform the prediction under the test database
#predictions = clf.predict(testFeatures)

# Evaluate the results
#accuracy = accuracy_score(testTargets, predictions)
#recall = recall_score(testTargets, predictions) 
#precision = precision_score(testTargets, predictions) 
#f1Score = f1_score(testTargets, predictions)

#print('Essemble Prediction Accuracy: {0:0.2f}'.format(accuracy))
#print('Essemble Prediction Recall: {0:0.2f}'.format(recall))
#print('Essemble Prediction Precision: {0:0.2f}'.format(precision))
#print('Essembke Prediction F1 score: {0:0.2f}'.format(f1Score))


############################
# Decision Tree Prediction #
############################
clf = tree.DecisionTreeClassifier()
clf = clf.fit(trainFeatures, trainTargets)
predictions = clf.predict(testFeatures)

# Evaluate the results
accuracy = accuracy_score(testTargets, predictions)
recall = recall_score(testTargets, predictions) 
precision = precision_score(testTargets, predictions) 
f1Score = f1_score(testTargets, predictions)

print('Tree Prediction Accuracy: {0:0.2f}'.format(accuracy))
print('Tree Prediction Recall: {0:0.2f}'.format(recall))
print('Tree Prediction Precision: {0:0.2f}'.format(precision))
print('Tree Prediction F1 score: {0:0.2f}'.format(f1Score))

# Confusion matrix
print(confusion_matrix(testTargets, predictions))
print(classification_report(testTargets, predictions))


# AUC
fpr, tpr, thresholds = metrics.roc_curve(testTargets, predictions)
roc_auc = metrics.auc(fpr, tpr)

print('Tree AUC FPR:')
print(fpr)
print('Tree AUC TPR:')
print(tpr)
print('Tree AUC Thresholds:')
print(thresholds)
print('Tree AUC ROC:')
print(metrics.auc(fpr, tpr))


#############################
# Neural Network Prediction #
#############################
neuralNetwork = MLPClassifier(solver='adam', alpha=1e-5,
                    hidden_layer_sizes=(5, 2), random_state=1)
neuralNetwork = neuralNetwork.fit(trainFeatures, trainTargets)
predictions = neuralNetwork.predict(testFeatures)

# Evaluate the results
accuracy = accuracy_score(testTargets, predictions)
recall = recall_score(testTargets, predictions) 
precision = precision_score(testTargets, predictions) 
f1Score = f1_score(testTargets, predictions)

print('Neural Network Prediction Accuracy: {0:0.2f}'.format(accuracy))
print('Neural Network Prediction Recall: {0:0.2f}'.format(recall))
print('Neural Network Prediction Precision: {0:0.2f}'.format(precision))
print('Neural Network Prediction F1 score: {0:0.2f}'.format(f1Score))

# Confusion matrix
print(confusion_matrix(testTargets, predictions))
print(classification_report(testTargets, predictions))


# AUC
fpr, tpr, thresholds = metrics.roc_curve(testTargets, predictions)
roc_auc = metrics.auc(fpr, tpr)

print('Neural Network AUC FPR:')
print(fpr)
print('Neural Network AUC TPR:')
print(tpr)
print('Neural Network AUC Thresholds:')
print(thresholds)
print('Neural Network AUC ROC:')
print(metrics.auc(fpr, tpr))

###########################
## GaussianNB Prediction ##
###########################
gnb = GaussianNB()
gnb = gnb.fit(trainFeatures, trainTargets)
predictions = gnb.predict(testFeatures)

# Evaluate the results
accuracy = accuracy_score(testTargets, predictions)
recall = recall_score(testTargets, predictions) 
precision = precision_score(testTargets, predictions) 
f1Score = f1_score(testTargets, predictions)

print('GaussianNB Accuracy: {0:0.2f}'.format(accuracy))
print('GaussianNB Recall: {0:0.2f}'.format(recall))
print('GaussianNB Precision: {0:0.2f}'.format(precision))
print('GaussianNB F1 score: {0:0.2f}'.format(f1Score))

# Confusion matrix
print(confusion_matrix(testTargets, predictions))
print(classification_report(testTargets, predictions))


# AUC
fpr, tpr, thresholds = metrics.roc_curve(testTargets, predictions)
roc_auc = metrics.auc(fpr, tpr)

print('GaussianNB AUC FPR:')
print(fpr)
print('GaussianNB AUC TPR:')
print(tpr)
print('GaussianNB AUC Thresholds:')
print(thresholds)
print('GaussianNB AUC ROC:')
print(metrics.auc(fpr, tpr))


# Setup the parameters and distributions to sample from: param_dist
param_dist = {"max_depth": [3, None],
              "criterion": ["gini", "entropy"]}

# Classify using decision tree, but now with hyper parameters
dtf_hps= GridSearchCV(clf,param_dist,cv=5)
dtf_hps.fit(trainFeatures, trainTargets)
predictions = dtf_hps.predict(testFeatures)

print("Best value for max_depth parameter: {0}".format(dtf_hps.best_params_['max_depth']))
print("Best value for criterion parameter: {0}".format(dtf_hps.best_params_['criterion']))

# Evaluate the results
accuracy = accuracy_score(testTargets, predictions)
recall = recall_score(testTargets, predictions) 
precision = precision_score(testTargets, predictions) 
f1Score = f1_score(testTargets, predictions)

print('GridSearch CV Accuracy: {0:0.2f}'.format(accuracy))
print('GridSearch CV Recall: {0:0.2f}'.format(recall))
print('GridSearch CV Precision: {0:0.2f}'.format(precision))
print('GridSearch CV F1 score: {0:0.2f}'.format(f1Score))

# Confusion matrix
print(confusion_matrix(testTargets, predictions))
print(classification_report(testTargets, predictions))

# AUC
fpr, tpr, thresholds = metrics.roc_curve(testTargets, predictions)
roc_auc = metrics.auc(fpr, tpr)

print('GridSearch CV AUC FPR:')
print(fpr)
print('GridSearch CV AUC TPR:')
print(tpr)
print('GridSearch CV AUC Thresholds:')
print(thresholds)
print('GridSearch CV AUC ROC:')
print(metrics.auc(fpr, tpr))