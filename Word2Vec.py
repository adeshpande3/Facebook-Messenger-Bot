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

# Check out Tensorflow's documentation which is pretty good for Word2Vec
# https://www.tensorflow.org/tutorials/word2vec

wordVecDimensions = 100
batchSize = 128
numNegativeSample = 64
windowSize = 5
numIterations = 100000

# Removes an annoying Tensorflow warning
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# This function just takes in the conversation data and makes it 
# into one huge string, and then uses a Counter to identify words
# and the number of occurences
def processDataset(filename):
    with open(filename, 'r', encoding='utf-8') as openedFile:
        allLines = openedFile.readlines()
    myStr = ""
    for line in allLines:
        myStr += line
    finalDict = Counter(myStr.split())
    return myStr, finalDict

def createTrainingMatrices(dictionary, corpus):
    allUniqueWords = list(dictionary.keys())	
    allWords = corpus.split()
    numTotalWords = len(allWords)
    xTrain=[]
    yTrain=[]
    for i in range(numTotalWords):
        if i % 100000 == 0:
            print('Finished %d/%d total words' % (i, numTotalWords))
        wordsAfter = allWords[i + 1:i + windowSize + 1]
        wordsBefore = allWords[max(0, i - windowSize):i]
        wordsAdded = wordsAfter + wordsBefore
        for word in wordsAdded:
            xTrain.append(allUniqueWords.index(allWords[i]))
            yTrain.append(allUniqueWords.index(word))
    np.save('Word2VecXTrain.npy', xTrain)
    np.save('Word2VecYTrain.npy', yTrain)
    return np.load('Word2VecXTrain.npy'), np.load('Word2VecYTrain.npy')

def getTrainingBatch():
    num = randint(0,numTrainingExamples - batchSize - 1)
    arr = xTrain[num:num + batchSize]
    labels = yTrain[num:num + batchSize]
    return arr, labels[:,np.newaxis]

continueWord2Vec = True
# Loading the data structures if they are present in the directory
if (os.path.isfile('Word2VecXTrain.npy') and os.path.isfile('Word2VecYTrain.npy') and os.path.isfile('wordList.txt')):
    xTrain = np.load('Word2VecXTrain.npy')
    yTrain = np.load('Word2VecYTrain.npy')
    print('Finished loading training matrices')
    with open("wordList.txt", "rb") as fp:
        wordList = pickle.load(fp)
    print('Finished loading word list')

else:
    fullCorpus, datasetDictionary = processDataset('conversationData.txt')
    print('Finished parsing and cleaning dataset')
    wordList = list(datasetDictionary.keys())
    createOwnVectors = input('Do you want to create your own vectors through Word2Vec (y/n)?')
    if (createOwnVectors == 'y'):
        xTrain, yTrain  = createTrainingMatrices(datasetDictionary, fullCorpus)
        print('Finished creating training matrices')
    else:
        continueWord2Vec = False
    with open("wordList.txt", "wb") as fp: 
        pickle.dump(wordList, fp)
    
# If you do not want to create your own word vectors and you'd just like to 
# have Tensorflow's seq2seq take care of that, then you don't need to run 
# anything below this line. 
if (continueWord2Vec == False):
    sys.exit()

numTrainingExamples = len(xTrain)
vocabSize = len(wordList)

tf.compat.v1.disable_eager_execution() # Need to disable eager execution to use placeholder() here in tensorflow 2.0+
sess = tf.compat.v1.Session()
embeddingMatrix = tf.Variable(tf.random.uniform([vocabSize, wordVecDimensions], -1.0, 1.0))
nceWeights = tf.Variable(tf.random.truncated_normal([vocabSize, wordVecDimensions], stddev=1.0 / math.sqrt(wordVecDimensions)))
nceBiases = tf.Variable(tf.zeros([vocabSize]))

inputs = tf.compat.v1.placeholder(tf.int32, shape=[batchSize])
outputs = tf.compat.v1.placeholder(tf.int32, shape=[batchSize, 1])

embed = tf.nn.embedding_lookup(embeddingMatrix, inputs)

loss = tf.reduce_mean(
  tf.nn.nce_loss(weights=nceWeights,
                 biases=nceBiases,
                 labels=outputs,
                 inputs=embed,
                 num_sampled=numNegativeSample,
                 num_classes=vocabSize))

optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1.0).minimize(loss)

sess.run(tf.compat.v1.global_variables_initializer())
for i in range(numIterations):
    trainInputs, trainLabels = getTrainingBatch()
    _, curLoss = sess.run([optimizer, loss], feed_dict={inputs: trainInputs, outputs: trainLabels})
    if (i % 10000 == 0):
        print ('Current loss is:', curLoss)
print('Saving the word embedding matrix')
embedMatrix = embeddingMatrix.eval(session=sess)
np.save('embeddingMatrix.npy', embedMatrix)
