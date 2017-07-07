import pandas as pd
import os
from datetime import datetime

def getGoogleHangoutsData():
	# Putting all the file names in a list
	allFiles = []
	for filename in os.listdir('GoogleTextForm'):
	    if filename.endswith(".txt"): 
	        allFiles.append('GoogleTextForm/' + filename)

	responseDictionary = dict() # The key is the other person's message, and the value is my response
	# Going through each file, and recording everyone's messages to me, and my responses
	for currentFile in allFiles:
		myMessage, otherPersonsMessage, currentSpeaker = "","",""
		openedFile = open(currentFile, 'r') 
		allLines = openedFile.readlines()
	   	for index,lines in enumerate(allLines):
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
	   	            currentLine = allLines[counter]
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
	responseDictionary = dict()
	fbFile = open('fbMessages.txt', 'r') 
	allLines = fbFile.readlines()
	myMessage, otherPersonsMessage, currentSpeaker = "","",""
	for index,lines in enumerate(allLines):
	    rightBracket = lines.find(']') + 2
	    justMessage = lines[rightBracket:]
	    colon = justMessage.find(':')
	    # Find messages that I sent
	    if (justMessage[:colon] == "Adit Deshpande"):
	        if not myMessage:
	            # Want to find the first message that I send (if I send multiple in a row)
	            startMessageIndex = index - 1
	        myMessage += justMessage[colon+2:]
	        
	    elif myMessage:
	        # Now go and see what message the other person sent by looking at previous messages
	        for counter in range(startMessageIndex, 0, -1):
	            currentLine = allLines[counter]
	            rightBracket = currentLine.find(']') + 2
	            justMessage = currentLine[rightBracket:]
	            colon = justMessage.find(':')
	            if not currentSpeaker:
	                # The first speaker not named me
	                currentSpeaker = justMessage[:colon]
	            elif (currentSpeaker != justMessage[:colon] and otherPersonsMessage):
	                # A different person started speaking, so now I know that the first person's message is done
	                responseDictionary[otherPersonsMessage] = myMessage
	                break
	            otherPersonsMessage += justMessage[colon+2:]
	        myMessage, otherPersonsMessage, currentSpeaker = "","",""    
	return responseDictionary

def getLinkedInData():
	df = pd.read_csv('Inbox.csv')
	dateTimeConverter = lambda x: datetime.strptime(x,'%B %d, %Y, %I:%M %p')
	responseDictionary = dict()
	peopleContacted = df['From'].unique().tolist()
	for person in peopleContacted:
	    receivedMessages = df[df['From'] == person]
	    sentMessages = df[df['To'] == person]
	    if (len(sentMessages) == 0 or len(receivedMessages) == 0):
	        # There was no actual conversation
	        continue
	    combined = pd.concat([sentMessages, receivedMessages])
	    combined['Date'] = combined['Date'].apply(dateTimeConverter)
	    combined = combined.sort(['Date'])
	    otherPersonsMessage, myMessage = "",""
	    firstMessage = True
	    for index, row in combined.iterrows():
	        if (row['From'] != 'Adit Deshpande'):
	            if myMessage and otherPersonsMessage:
	                responseDictionary[otherPersonsMessage.rstrip()] = myMessage.rstrip()
	                otherPersonsMessage, myMessage = "",""
	            otherPersonsMessage = otherPersonsMessage + row['Content'] + " "
	        else:
	            if (firstMessage):
	                firstMessage = False
	                # Don't include if I am the person initiating the convo
	                continue
	            myMessage = myMessage + str(row['Content']) + " "
	return responseDictionary

combinedDictionary = {}
print 'Getting Google Hangout Data'
combinedDictionary.update(getGoogleHangoutsData())
print 'Getting Facebook Data'
combinedDictionary.update(getFacebookData())
print 'Getting LinkedIn Data'
combinedDictionary.update(getLinkedInData())
print 'Total len of dictionary', len(combinedDictionary)

encoderFile = open('EncoderData.txt', 'w')
decoderFile = open('DecoderData.txt', 'w')
conversationFile = open('ConversationData.txt', 'w')
for key,value in combinedDictionary.iteritems():
	if (not key.strip() or not value.strip()):
		# If there are empty strings
		continue
   	encoderFile.write(key)
   	decoderFile.write(value)
    # Some formatting for the conversation data file
   	if (key.strip()[-1] != '.'):
   	    formatedKey = key.strip() + '. '
   	else:
   	    formatedKey = key.strip() + ' '
   	if (value.strip()[-1] != '.'):
   	    formatedValue = value.strip() + '. '
   	else:
   	    formatedValue = value.strip() + ' '
   	conversationFile.write(formatedKey + formatedValue)