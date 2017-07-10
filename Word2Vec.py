import tensorflow as tf 
import numpy as np 

wordVecDimensions = 100
batchSize = 32
numNegativeSample = 64
numIterations = 10000

def cleanDataset(filename):
	openedFile = open(filename, 'r')
	allLines = openedFile.readlines()

def createTrainingMatrices(dataset):
	allWords = set(dataset.split())
	return allWords, xTrain, yTrain

def getTrainingBatch():
	num = randint(0,numTrainingExamples - batchSize)
	arr = xTrain[num:num + batchSize]
	labels = yTrain[num:num + batchSize]
	return arr, labels

cleanedDataset = cleanDataset('ConversationData.txt')
wordList, xTrain, yTrain  = createTrainingMatrices(cleanedDataset)
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

for i in range(numIterations):
	trainInputs, trainLabels = getTrainingBatch()
	_, curLoss = session.run([optimizer, loss], feed_dict={inputs = trainInputs, outputs = trainLabels})
	if (i % 100 == 0):
		print ('Current loss is:', curLoss)