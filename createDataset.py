import pandas as pd
import numpy as np
import os
import re
from datetime import datetime

fbData = input('Do you have Facebook data to parse through (y/n)?')
googleData = input('Do you have Google Hangouts data to parse through (y/n)?')
linkedInData = input('Do you have LinkedIn data to parse through (y/n)?')
whatsAppData = input('Do you have whatsAppData to parse through (y/n)?')
discordData = input('Do you have discordData to parse through (y/n)?')

def getWhatsAppData():
    personName = input('Enter your full WhatsApp name: ')
    df = pd.read_csv('whatsapp_chats.csv')
    responseDictionary = dict()
    receivedMessages = df[df['From'] != personName]
    sentMessages = df[df['From'] == personName]
    combined = pd.concat([sentMessages, receivedMessages])
    otherPersonsMessage, myMessage = "",""
    firstMessage = True
    for index, row in combined.iterrows():
        if (row['From'] != personName):
            if myMessage and otherPersonsMessage:
                otherPersonsMessage = cleanMessage(otherPersonsMessage)
                myMessage = cleanMessage(myMessage)
                responseDictionary[otherPersonsMessage.rstrip()] = myMessage.rstrip()
                otherPersonsMessage, myMessage = "",""
            otherPersonsMessage = otherPersonsMessage + str(row['Content']) + " "
        else:
            if (firstMessage):
                firstMessage = False
                # Don't include if I am the person initiating the convo
                continue
            myMessage = myMessage + str(row['Content']) + " "
    return responseDictionary

def getGoogleHangoutsData():
    personName = input('Enter your full Hangouts name: ')
    # Putting all the file names in a list
    allFiles = []
    # Edit these file and directory names if you have them saved somewhere else
    for filename in os.listdir('GoogleTextForm'):
        if filename.endswith(".txt"):
            allFiles.append('GoogleTextForm/' + filename)

    responseDictionary = dict() # The key is the other person's message, and the value is my response
    # Going through each file, and recording everyone's messages to me, and my
                                # responses
    for currentFile in allFiles:
        myMessage, otherPersonsMessage, currentSpeaker = "","",""
        with open(currentFile, 'r', encoding='utf-8') as openedFile:
            allLines = openedFile.readlines()
        for index,lines in enumerate(allLines):
            # The sender's name is separated by < and >
            leftBracket = lines.find('<')
            rightBracket = lines.find('>')
            
            # Find messages that I sent
            if (lines[leftBracket + 1:rightBracket] == personName):
                if not myMessage:
                    # Want to find the first message that I send (if I send
                    # multiple in a row)
                    startMessageIndex = index - 1
                myMessage += lines[rightBracket + 1:]

            elif myMessage:
                # Now go and see what message the other person sent by looking at
                # previous messages
                for counter in range(startMessageIndex, 0, -1):
                    currentLine = allLines[counter]
                    # In case the message above isn't in the right format
                    if (currentLine.find('<') < 0 or currentLine.find('>') < 0):
                        myMessage, otherPersonsMessage, currentSpeaker = "","",""
                        break
                    if not currentSpeaker:
                        # The first speaker not named me
                        currentSpeaker = currentLine[currentLine.find('<') + 1:currentLine.find('>')]
                    elif (currentSpeaker != currentLine[currentLine.find('<') + 1:currentLine.find('>')]):
                        # A different person started speaking, so now I know that
                        # the first person's message is done
                        otherPersonsMessage = cleanMessage(otherPersonsMessage)
                        myMessage = cleanMessage(myMessage)
                        responseDictionary[otherPersonsMessage] = myMessage
                        break
                    otherPersonsMessage = currentLine[currentLine.find('>') + 1:] + otherPersonsMessage
                myMessage, otherPersonsMessage, currentSpeaker = "","",""
    return responseDictionary

def getFacebookData():
    personName = input('Enter your full Facebook name: ')
    responseDictionary = dict()
    with open('fbMessages.txt', 'r', encoding='utf-8') as fbFile:
        allLines = fbFile.readlines()
    myMessage, otherPersonsMessage, currentSpeaker = "","",""
    for index,lines in enumerate(allLines):
        rightBracket = lines.find(']') + 2
        justMessage = lines[rightBracket:]
        colon = justMessage.find(':')
        # Find messages that I sent
        if (justMessage[:colon] == personName):
            if not myMessage:
                # Want to find the first message that I send (if I send multiple
                # in a row)
                startMessageIndex = index - 1
            myMessage += justMessage[colon + 2:]

        elif myMessage:
            # Now go and see what message the other person sent by looking at
            # previous messages
            for counter in range(startMessageIndex, 0, -1):
                currentLine = allLines[counter]
                rightBracket = currentLine.find(']') + 2
                justMessage = currentLine[rightBracket:]
                colon = justMessage.find(':')
                if not currentSpeaker:
                    # The first speaker not named me
                    currentSpeaker = justMessage[:colon]
                elif (currentSpeaker != justMessage[:colon] and otherPersonsMessage):
                    # A different person started speaking, so now I know that the
                    # first person's message is done
                    otherPersonsMessage = cleanMessage(otherPersonsMessage)
                    myMessage = cleanMessage(myMessage)
                    responseDictionary[otherPersonsMessage] = myMessage
                    break
                otherPersonsMessage = justMessage[colon + 2:] + otherPersonsMessage
            myMessage, otherPersonsMessage, currentSpeaker = "","",""
    return responseDictionary

def getLinkedInData():
    personName = input('Enter your full LinkedIn name: ')
    df = pd.read_csv('Inbox.csv')
    dateTimeConverter = lambda x: datetime.fromisoformat(x)
    responseDictionary = dict()
    peopleContacted = df['FROM'].unique().tolist()
    for person in peopleContacted:
        receivedMessages = df[df['FROM'] == person]
        sentMessages = df[df['TO'] == person]
        if (len(sentMessages) == 0 or len(receivedMessages) == 0):
            # There was no actual conversation
            continue
        combined = pd.concat([sentMessages, receivedMessages])
        combined['DATE'] = combined['DATE'].apply(dateTimeConverter)
        combined = combined.sort(['DATE'])
        otherPersonsMessage, myMessage = "",""
        firstMessage = True
        for index, row in combined.iterrows():
            if (row['FROM'] != personName):
                if myMessage and otherPersonsMessage:
                    otherPersonsMessage = cleanMessage(otherPersonsMessage)
                    myMessage = cleanMessage(myMessage)
                    responseDictionary[otherPersonsMessage.rstrip()] = myMessage.rstrip()
                    otherPersonsMessage, myMessage = "",""
                otherPersonsMessage = otherPersonsMessage + str(row['CONTENT']) + " "
            else:
                if (firstMessage):
                    firstMessage = False
                    # Don't include if I am the person initiating the convo
                    continue
                myMessage = myMessage + str(row['CONTENT']) + " "
    return responseDictionary

def getDiscordData():
    personName = input('Enter your full Discord name: ')
    # Putting all the file names in a list
    allFiles = []
    # Edit these file and directory names if you have them saved somewhere else
    for filename in os.listdir('DiscordChatLogs'):
        if filename.endswith(".txt"): 
            allFiles.append('DiscordChatLogs/' + filename)
    responseDictionary = dict() # The key is the other person's message, and the value is my response
    # Going through each file, and recording everyone's messages to me, and my
                                # responses
    for currentFile in allFiles:
        with open(currentFile, 'r', encoding='utf-8') as openedFile:
            allLines = openedFile.readlines()
        data = ''.join(allLines)
        otherPersonsMessage, myMessage = "",""
        response_sets = re.findall(r'\[.+\] (?!' + re.escape(personName) + r').+\n(.+)\n{2}(?:\[.+\] ' + re.escape(personName) + r'\n(.+)\n{2})', data)
        for response_set in response_sets:
            otherPersonsMessage = response_set[0]
            myMessage = response_set[1]
            responseDictionary[otherPersonsMessage] = cleanMessage(myMessage)
            otherPersonsMessage, myMessage = "",""
    return responseDictionary

def cleanMessage(message):
    # Remove new lines within message
    cleanedMessage = message.replace('\n',' ').lower()
    # Deal with some weird tokens
    cleanedMessage = cleanedMessage.replace("\xc2\xa0", "")
    # Remove punctuation
    cleanedMessage = re.sub('([.,!?])','', cleanedMessage)
    # Remove multiple spaces in message
    cleanedMessage = re.sub(' +',' ', cleanedMessage)
    return cleanedMessage

combinedDictionary = {}
if (googleData == 'y'):
    print('Getting Google Hangout Data')
    combinedDictionary.update(getGoogleHangoutsData())
if (fbData == 'y'):
    print('Getting Facebook Data')
    combinedDictionary.update(getFacebookData())
if (linkedInData == 'y'):
    print('Getting LinkedIn Data')
    combinedDictionary.update(getLinkedInData())
if (whatsAppData == 'y'):
    print('Getting whatsApp Data')
    combinedDictionary.update(getWhatsAppData())
if (discordData == 'y'):
    print('Getting Discord Data')
    combinedDictionary.update(getDiscordData())
print('Total len of dictionary ' + str(len(combinedDictionary)))

print('Saving conversation data dictionary')
np.save('conversationDictionary.npy', combinedDictionary)

conversationFile = open('conversationData.txt', 'w', encoding='utf-8')
for key, value in combinedDictionary.items():
    if (not key.strip() or not value.strip()):
        # If there are empty strings
        continue
    conversationFile.write(key.strip() + value.strip())
