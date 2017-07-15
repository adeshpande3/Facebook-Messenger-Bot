import tensorflow as tf 
import numpy as np 
import sys
from random import randint
import pickle
import os

def createTrainingMatrices(conversationFileName, wList, maxLen):
	conversationDictionary = np.load(conversationFileName).item()
	numExamples = len(conversationDictionary)
	xTrain = np.zeros((numExamples, maxLen), dtype='int32')
	yTrain = np.zeros((numExamples, maxLen), dtype='int32')
	for index,(key,value) in enumerate(conversationDictionary.iteritems()):
		# Will store integerized representation of strings here
		encoderMessage = np.zeros((maxLen), dtype='int32')
		decoderMessage = np.zeros((maxLen), dtype='int32')
		# Getting all the individual words in the strings
		keySplit = key.split()
		valueSplit = value.split()
		keyCount = len(keySplit)
		valueCount = len(valueSplit)
		# Throw out sequences that are too long
		if (keyCount > maxLen or valueCount > maxLen):
			continue
		for keyIndex, word in enumerate(keySplit):
			try:
				encoderMessage[keyIndex] = wList.index(word)
			except ValueError:
				# TODO: This isnt really the right way to handle this scenario
				encoderMessage[keyIndex] = 0
		for valueIndex, word in enumerate(keySplit):
			try:
				decoderMessage[keyIndex] = wList.index(word)
			except ValueError:
				decoderMessage[valueIndex] = 0
		xTrain[index] = encoderMessage
		yTrain[index] = decoderMessage
	return numExamples, xTrain, yTrain

def getTrainingBatch(localXTrain, localYTrain, localBatchSize):
	num = randint(0,numTrainingExamples - localBatchSize - 1)
	arr = localXTrain[num:num + localBatchSize]
	labels = localYTrain[num:num + localBatchSize]
	return arr, labels[:,np.newaxis]

# Hyperparamters
batchSize = 1
maxEncoderLength = 30
maxDecoderLength = maxEncoderLength
lstmUnits = 48
numIterations = 0

# Loading in all the data structures
with open("wordList.txt", "rb") as fp:
	wordList = pickle.load(fp)

if (os.path.isfile('Seq2SeqXTrain.npy') and os.path.isfile('Seq2SeqYTrain.npy')):
	xTrain = np.load('Seq2SeqXTrain.npy')
	yTrain = np.load('Seq2SeqYTrain.npy')
	print 'Finished loading training matrices'
	numTrainingExamples = xTrain.shape[0]
else:
	numTrainingExamples, xTrain, yTrain = createTrainingMatrices('conversationDictionary.npy', wordList, maxEncoderLength)
	np.save('Seq2SeqXTrain.npy', xTrain)
	np.save('Seq2SeqYTrain.npy', yTrain)
	print 'Finished creating training matrices'

wordVectors = np.load('embeddingIteration9000000.npy')
numWords = len(wordList)
wordVecDimensions = wordVectors.shape[1]

EOSvector = np.zeros((wordVecDimensions), dtype='int32')
SOSvector = np.ones((wordVecDimensions), dtype='int32')

sess = tf.Session()
tf.reset_default_graph()

# Create the placeholders
encoderInputs = tf.placeholder(shape=(batchSize, maxEncoderLength), dtype=tf.int32)
decoderOutputs = tf.placeholder(shape=(batchSize, maxDecoderLength), dtype=tf.int32)
decoderInputs = tf.placeholder(shape=(batchSize, maxDecoderLength), dtype=tf.int32)

encoderData = tf.Variable(tf.zeros([batchSize, maxEncoderLength, wordVecDimensions]),dtype=tf.float32)
encoderData = tf.nn.embedding_lookup(wordVectors,encoderInputs)
decoderData = tf.Variable(tf.zeros([batchSize, maxDecoderLength, wordVecDimensions]),dtype=tf.float32)
decoderData = tf.nn.embedding_lookup(wordVectors,decoderInputs)

encoderLSTM = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
encoderOutputs, encoderFinalState = tf.nn.dynamic_rnn(encoderLSTM, encoderData, dtype=tf.float32)

decoderLSTM = tf.contrib.rnn.LSTMCell(lstmUnits)
decoderOutputs, decoderFinalState = tf.nn.dynamic_rnn(decoderLSTM, decoderData, 
	initial_state=encoderFinalState,dtype=tf.float32)

# TODO defining a loss function
loss = decoderOutputs
optimizer = tf.train.AdamOptimizer().minimize(loss)

sess.run(tf.global_variables_initializer())

for i in range(numIterations):
	encoderTrain, decoderTrain = getTrainingBatch(xTrain, yTrain, batchSize)
	curLoss,_ = sess.run([loss, optimizer], feed_dict={encoderInputs: encoderTrain, decoderOutputs: decoderTrain})

temp = sess.run(encoderData, feed_dict={encoderInputs: encoderTrain, decoderOutputs: decoderTrain})
print temp.shape