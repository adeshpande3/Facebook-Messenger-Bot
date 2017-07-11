import tensorflow as tf 
import numpy as np 
import re
from collections import Counter
import sys
import pickle
import os

# This Word2Vec implementation is largely based on this paper
# https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf
# It's a bit old, but Word2Vec is still SOTA and relatively simple, so I'm going with it

wordVecDimensions = 100
batchSize = 32
numNegativeSample = 64
windowSize = 5
numIterations = 10000

def cleanDataset(filename):
	openedFile = open(filename, 'r')
	allLines = openedFile.readlines()
	myStr = ""
	for line in allLines:
	    tempStr = line.replace('\n',' ').lower()
	    myStr += re.sub('[.!?]','', tempStr)
	return myStr, Counter(myStr.split())

def createTrainingMatrices(dictionary, corpus):
	allUniqueWords = dictionary.keys()	
	allWords = corpus.split()
	xTrain=[]
	yTrain=[]
	for i in range(len(allWords)):
		wordsAfter = allWords[i+1:i+windowSize+1]
		for word in wordsAfter:
		    xTrain.append(allUniqueWords.index(allWords[i]))
		    yTrain.append(allUniqueWords.index(word))
		wordsBefore = allWords[max(0, i-windowSize):i]
		for word in wordsBefore:
		    xTrain.append(allUniqueWords.index(allWords[i]))
		    yTrain.append(allUniqueWords.index(word))
	return allUniqueWords, xTrain, yTrain

def getTrainingBatch():
	num = randint(0,numTrainingExamples - batchSize)
	arr = xTrain[num:num + batchSize]
	labels = yTrain[num:num + batchSize]
	return arr, labels

if (os.path.isfile('xTrain.npy') and os.path.isfile('yTrain.npy') and os.path.isfile('wordList.txt')):
	xTrain = np.load('xTrain.npy')
	yTrain = np.load('yTrain.npy')
	with open("wordList.txt", "rb") as fp:
		allUniqueWords = pickle.load(fp)

fullCorpus, datasetDictionary = cleanDataset('ConversationData.txt')
wordList, xTrain, yTrain  = createTrainingMatrices(datasetDictionary, fullCorpus)
numTrainingExamples = len(xTrain)
vocabSize = len(wordList)

np.save('xTrain.npy', xTrain)
np.save('yTrain.npy', yTrain)
with open("wordList.txt", "wb") as fp: 
	pickle.dump(allUniqueWords, fp)

sys.exit()

sess = tf.Session()
embeddingMatrix = tf.Variable(tf.random_uniform([vocabSize, wordVecDimensions], -1.0, 1.0))
nceWeights = tf.Variable(tf.truncated_normal([vocabSize, wordVecDimensions], stddev=1.0 / math.sqrt(wordVecDimensions)))
nceBiases = tf.Variable(tf.zeros([vocabSize]))

inputs = tf.placeholder(tf.int32, shape=[batchSize])
outputs = tf.placeholder(tf.int32, shape=[batchSize, 1])

embed = tf.nn.embedding_lookup(embeddingMatrix, inputs)

loss = tf.reduce_mean(
  tf.nn.nce_loss(weights=nceWeights,
                 biases=nceBiases,
                 labels=outputs,
                 inputs=embed,
                 num_sampled=numNegativeSample,
                 num_classes=vocabSize))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0).minimize(loss)

for i in range(numIterations):
	trainInputs, trainLabels = getTrainingBatch()
	_, curLoss = sess.run([optimizer, loss], feed_dict={inputs: trainInputs, outputs: trainLabels})
	if (i % 100 == 0):
		print ('Current loss is:', curLoss)