import tensorflow as tf 
import numpy as np 
import re
from collections import Counter
import sys
import math
from random import randint
import pickle
import os

# This Word2Vec implementation is largely based on this paper
# https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf
# It's a bit old, but Word2Vec is still SOTA and relatively simple, so I'm going with it

wordVecDimensions = 100
batchSize = 128
numNegativeSample = 64
windowSize = 5
numIterations = 10000000

def cleanDataset(filename):
	openedFile = open(filename, 'r')
	allLines = openedFile.readlines()
	myStr = ""
	for line in allLines:
	    tempStr = line.replace('\n',' ').lower()
	    myStr += re.sub('([.!?])','', tempStr)
	finalDict = Counter(myStr.split())
	return myStr, finalDict

def createTrainingMatrices(dictionary, corpus):
	allUniqueWords = dictionary.keys()	
	allWords = corpus.split()
	numTotalWords = len(allWords)
	xTrain=[]
	yTrain=[]
	for i in range(numTotalWords):
		if i % 100000 == 0:
			print 'Finished %d/%d total words' % (i, numTotalWords)
		wordsAfter = allWords[i + 1:i + windowSize + 1]
		wordsBefore = allWords[max(0, i - windowSize):i]
		wordsAdded = wordsAfter + wordsBefore
		for word in wordsAdded:
			xTrain.append(allUniqueWords.index(allWords[i]))
			yTrain.append(allUniqueWords.index(word))
	return allUniqueWords, xTrain, yTrain

def getTrainingBatch():
	num = randint(0,numTrainingExamples - batchSize - 1)
	arr = xTrain[num:num + batchSize]
	labels = yTrain[num:num + batchSize]
	return arr, labels[:,np.newaxis]

if (os.path.isfile('xTrain.npy') and os.path.isfile('yTrain.npy') and os.path.isfile('wordList.txt')):
	xTrain = np.load('xTrain.npy')
	yTrain = np.load('yTrain.npy')
	print 'Finished loading training matrices'
	with open("wordList.txt", "rb") as fp:
		wordList = pickle.load(fp)
	print 'Finished loading word list'

else:
	fullCorpus, datasetDictionary = cleanDataset('ConversationData.txt')
	print 'Finished parsing and cleaning dataset'
	wordList, xTrain, yTrain  = createTrainingMatrices(datasetDictionary, fullCorpus)
	print 'Finished creating training matrices'
	np.save('xTrain.npy', xTrain)
	np.save('yTrain.npy', yTrain)
	with open("wordList.txt", "wb") as fp: 
		pickle.dump(wordList, fp)
	
numTrainingExamples = len(xTrain)
vocabSize = len(wordList)

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

sess.run(tf.global_variables_initializer())
for i in range(numIterations):
	trainInputs, trainLabels = getTrainingBatch()
	_, curLoss = sess.run([optimizer, loss], feed_dict={inputs: trainInputs, outputs: trainLabels})
	if (i % 10000 == 0):
		print ('Current loss is:', curLoss)
	if (i % 1000000 == 0 and i != 0):
		print 'Saving the word embedding matrix'
		embedMatrix = embeddingMatrix.eval(session=sess)
		np.save('embeddingIteration' + str(i) + '.npy', embedMatrix)