import tensorflow as tf 
import numpy as np 
import sys
from random import randint
import pickle
import os

def createTrainingMatrices(encoderFileName, decoderFileName):
	encoderFile = open(encoderFileName, 'r')
	allInputLines = encoderFile.readlines()
	decoderFile = open(decoderFileName, 'r')
	allOutputLines = decoderFile.readlines()
	print 'Number of input lines:', len(allInputLines)
	print 'Number of output lines:', len(allOutputLines)

def getTrainingBatch():
	num = randint(0,numTrainingExamples - batchSize - 1)
	arr = xTrain[num:num + batchSize]
	labels = yTrain[num:num + batchSize]
	return arr, labels[:,np.newaxis]

createTrainingMatrices('EncoderData.txt', 'DecoderData.txt')
sys.exit()

# Loading in all the data structures
xTrain, yTrain = createTrainingMatrices('EncoderData.txt', 'DecoderData.txt')
wordVectors = np.load('embeddingIteration9000000.npy')
with open("wordList.txt", "rb") as fp:
	wordList = pickle.load(fp)
numWords = len(wordList)
wordVecDimensions = wordVectors.shape[1]

# Hyperparamters
batchSize = 32
maxEncoderLength = 30
maxDecoderLength = maxEncoderLength
lstmUnits = 48

EOSvector = np.zeros((wordVecDimensions))
SOSvector = np.ones((wordVecDimensions))

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
