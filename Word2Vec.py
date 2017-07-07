import tensorflow as tf 
import numpy as np 

def cleanDataset(filename):
	openedFile = open(filename, 'r')
	allLines = openedFile.readlines()

cleanDataset('ConversationData.txt')