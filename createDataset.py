import pandas as pd
import os

def getGoogleHangoutsData():
	# Putting all the file names in a list
	allFiles = []
	for filename in os.listdir('GoogleTextForm'):
	    if filename.endswith(".txt"): 
	        allFiles.append('GoogleTextForm/' + filename)

	responseDictionary = dict() # The key is the other person's message, and the value is my response
	# Going through each file, and recording everyone's messages to me, and my responses
	for currentFile in allFiles:
	    otherPersonsMessage = "" 
	    myMessage = ""
	    currentSpeaker = ""
	    openedFile = open(currentFile, 'r') 
	    f = openedFile.readlines()
	    for index,lines in enumerate(f):
	        leftBracket = lines.find('<')
	        rightBracket = lines.find('>')
	        
	        # Find messages that I sent
	        if (lines[leftBracket+1:rightBracket] == "Adit Deshpande"):
	            if not myMessage:
	                # Want to find the first message that I send (if I send multiple in a row)
	                startMessageIndex = index - 1
	            myMessage += lines[rightBracket+1:]
	        
	        elif myMessage:
	            # Now go and see what message the other person sent by looking at previous messages
	            for counter in range(startMessageIndex, 0, -1):
	                currentLine = f[counter]
	                # In case the message above isn't in the right format
	                if (currentLine.find('<') < 0 or currentLine.find('>') < 0):
	                    myMessage, otherPersonsMessage, currentSpeaker = "","",""
	                    break
	                if not currentSpeaker:
	                    # The first speaker not named me
	                    currentSpeaker = currentLine[currentLine.find('<')+1:currentLine.find('>')]
	                elif (currentSpeaker != currentLine[currentLine.find('<')+1:currentLine.find('>')]):
	                    # A different person started speaking, so now I know that the first person's message is done
	                    responseDictionary[otherPersonsMessage] = myMessage
	                    break
	                otherPersonsMessage += currentLine[currentLine.find('>')+1:]
	            myMessage, otherPersonsMessage, currentSpeaker = "","",""
	return responseDictionary

def getFacebookData():
	return responseDictionary

def getLinkedInData():




combinedDictionary = {}
combinedDictionary.update(getGoogleHangoutsData())
combinedDictionary.update(getFacebookData())
combinedDictionary.update(getLinkedInData())