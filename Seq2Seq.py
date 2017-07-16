import tensorflow as tf 
import numpy as np 
import sys
from random import randint
import datetime
from sklearn.utils import shuffle
import pickle
import os

def createTrainingMatrices(conversationFileName, wList, maxLen):
	conversationDictionary = np.load(conversationFileName).item()
	numExamples = len(conversationDictionary)
	xTrain = np.zeros((numExamples, maxLen), dtype='int32')
	yTrain = np.zeros((numExamples, maxLen), dtype='int32')
	for index,(key,value) in enumerate(conversationDictionary.iteritems()):
		# Will store integerized representation of strings here (initialized as padding)
		encoderMessage = np.full((maxLen), wList.index('<pad>'), dtype='int32')
		decoderMessage = np.full((maxLen), wList.index('<pad>'), dtype='int32')
		# Getting all the individual words in the strings
		keySplit = key.split()
		valueSplit = value.split()
		keyCount = len(keySplit)
		valueCount = len(valueSplit)
		# Throw out sequences that are too long
		if (keyCount > (maxLen - 1) or valueCount > (maxLen - 1)):
			continue
		for keyIndex, word in enumerate(keySplit):
			try:
				encoderMessage[keyIndex] = wList.index(word)
			except ValueError:
				# TODO: This isnt really the right way to handle this scenario
				encoderMessage[keyIndex] = 0
		encoderMessage[keyIndex + 1] = wList.index('<EOS>')
		for valueIndex, word in enumerate(valueSplit):
			try:
				decoderMessage[valueIndex] = wList.index(word)
			except ValueError:
				decoderMessage[valueIndex] = 0
		decoderMessage[valueIndex + 1] = wList.index('<EOS>')
		xTrain[index] = encoderMessage
		yTrain[index] = decoderMessage
	# Remove rows with all zeros
	yTrain = yTrain[~np.all(yTrain == 0, axis=1)]
	xTrain = xTrain[~np.all(xTrain == 0, axis=1)]
	numExamples = xTrain.shape[0]
	return numExamples, xTrain, yTrain

def getTrainingBatch(localXTrain, localYTrain, localBatchSize, maxLen):
	num = randint(0,numTrainingExamples - localBatchSize - 1)
	arr = localXTrain[num:num + localBatchSize]
	labels = localYTrain[num:num + localBatchSize]
	arr, labels = shuffle(arr, labels, random_state=0)

	laggedLabels = []
	EOStokenIndex = wordList.index('<EOS>')
	padTokenIndex = wordList.index('<pad>')
	for example in labels:
		eosFound = np.argwhere(example==EOStokenIndex)[0]
		shiftedExample = np.roll(example,1)
		shiftedExample[0] = EOStokenIndex
		# The EOS token was already at the end, so no need for pad
		if (eosFound != (maxLen - 1)):
			shiftedExample[eosFound+1] = padTokenIndex
		laggedLabels.append(shiftedExample)
	return list(arr), list(labels), laggedLabels

def getTestInput(inputMessage, wList, maxLen):
	encoderMessage = np.full((maxLen), wList.index('<pad>'), dtype='int32')
	inputSplit = inputMessage.lower().split()
	for index,word in enumerate(inputSplit):
		encoderMessage[index] = wList.index(word)
	encoderMessage[index + 1] = wList.index('<EOS>')
	return encoderMessage

def idsToSentence(ids, wList):
	EOStokenIndex = wordList.index('<EOS>')
	padTokenIndex = wordList.index('<pad>')
	myStr = ""
	for num in ids:
		if (num == EOStokenIndex or num == padTokenIndex):
			break
		else:
			myStr = myStr + wList[num] + " "
	return myStr

# Hyperparamters
batchSize = 12
maxEncoderLength = 15
maxDecoderLength = maxEncoderLength
lstmUnits = 48
numIterations = 10000

# Loading in all the data structures
with open("wordList.txt", "rb") as fp:
	wordList = pickle.load(fp)

wordVectors = np.load('embeddingIteration9000000.npy')
vocabSize = len(wordList)
wordVecDimensions = wordVectors.shape[1]

# Add two entries to the word vector matrix. One to represent padding tokens, 
# and one to represent an end of sentence token
padVector = np.zeros((1, wordVecDimensions), dtype='int32')
EOSVector = np.ones((1, wordVecDimensions), dtype='int32')
wordVectors = np.concatenate((wordVectors,padVector), axis=0)
wordVectors = np.concatenate((wordVectors,EOSVector), axis=0)

# Need to modify the word list as well
wordList.append('<pad>')
wordList.append('<EOS>')
vocabSize = vocabSize + 2

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

tf.reset_default_graph()

# Create the placeholders
encoderInputs = [tf.placeholder(tf.int32, shape=(None,)) for i in range(batchSize)]
decoderLabels = [tf.placeholder(tf.int32, shape=(None,)) for i in range(batchSize)]
decoderInputs = [tf.placeholder(tf.int32, shape=(None,)) for i in range(batchSize)]
feedPrevious = tf.placeholder(tf.bool)

encoderLSTM = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
decoderOutputs, decoderFinalState = tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(encoderInputs, decoderInputs, encoderLSTM, 
																		vocabSize, vocabSize, wordVecDimensions, feed_previous=feedPrevious)

#decoderLogits = tf.contrib.layers.linear(decoderOutputs, vocabSize)
decoderPrediction = tf.argmax(decoderOutputs, 2)

lossWeights = [tf.ones_like(l, dtype=tf.float32) for l in decoderLabels]
loss = tf.contrib.legacy_seq2seq.sequence_loss(decoderOutputs, decoderLabels, lossWeights, vocabSize)
optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)

sess = tf.Session()
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())

#tf.summary.scalar('Loss', loss)
#merged = tf.summary.merge_all()
#logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
#writer = tf.summary.FileWriter(logdir, sess.graph)

zeroVector = np.zeros((maxEncoderLength), dtype='int32')

for i in range(numIterations):
	encoderTrain, decoderTargetTrain, decoderInputTrain = getTrainingBatch(xTrain, yTrain, batchSize, maxEncoderLength)

	feedDict = {encoderInputs[t]: encoderTrain[t] for t in range(batchSize)}
	feedDict.update({decoderLabels[t]: decoderTargetTrain[t] for t in range(batchSize)})
	feedDict.update({decoderInputs[t]: decoderInputTrain[t] for t in range(batchSize)})
	feedDict.update({feedPrevious: False})

	curLoss, _, summary, pred = sess.run([loss, optimizer, merged, decoderPrediction], feed_dict=feedDict)
	#writer.add_summary(summary, i)
	if (i % 100 == 0):
		print('Current loss:', curLoss, 'at iteration', i)
	if (i % 10 == 0 and i != 0):
		inputVector = getTestInput("whats up hows it going bro", wordList, maxEncoderLength);
		feedDict = {encoderInputs[t]: inputVector for t in range(batchSize)}
		feedDict.update({decoderLabels[t]: zeroVector for t in range(batchSize)})
		feedDict.update({decoderInputs[t]: zeroVector for t in range(batchSize)})
		feedDict.update({feedPrevious: True})
		ids = (sess.run(decoderPrediction, feed_dict=feedDict))[0]
		print idsToSentence(ids, wordList)

	#if (i % 1000 == 0 and i != 0):
	#	savePath = saver.save(sess, "models/pretrained_seq2seq.ckpt", global_step=i)

#saver.restore(sess, tf.train.latest_checkpoint('models'))