import pandas as pd
import numpy as np
import os
import re
from datetime import datetime

facebookFlag = input('Do you have Facebook data to parse through (y/n)?')
gHangoutsFlag = input('Do you have Google Hangouts data to parse through (y/n)?')
linkedinFlag = input('Do you have LinkedIn data to parse through (y/n)?')
whatsappFlag = input('Do you have WhatsApp data to parse through (y/n)?')
discordFlag = input('Do you have Discord data to parse through (y/n)?')

def getWhatsAppDataCSV(userName):
    chatFrame = pd.read_csv('whatsapp_chats.csv')
    convoDict = dict()
    receivedMsgs = chatFrame[chatFrame['From'] != userName]
    sentMsgs = chatFrame[chatFrame['From'] == userName]
    mergedMsgs = pd.concat([sentMsgs, receivedMsgs])
    otherMsg, myMsg = "", ""
    firstMsgFlag = True
    for _, row in mergedMsgs.iterrows():
        if (row['From'] != userName):
            if myMsg and otherMsg:
                otherMsg = cleanMessage(otherMsg)
                myMsg = cleanMessage(myMsg)
                convoDict[otherMsg.rstrip()] = myMsg.rstrip()
                otherMsg, myMsg = "", ""
            otherMsg = otherMsg + str(row['Content']) + " "
        else:
            if (firstMsgFlag):
                firstMsgFlag = False
                continue
            myMsg = myMsg + str(row['Content']) + " "
    return convoDict

def getWhatsAppDataTXT(userName):
    fileList = []
    for fname in os.listdir('WhatsAppChatLogs'):
        if fname.endswith(".txt"):
            fileList.append('WhatsAppChatLogs/' + fname)

    convoDict = dict()
    for currentFile in fileList:
        myMsg, otherMsg, currentSpeaker = "", "", ""
        with open(currentFile, 'r', encoding="utf-8") as chatFile:
            lines = chatFile.readlines()
        for idx, line in enumerate(lines):
            leftDelimPattern = re.compile(r'[\]\-]')
            leftDelim = leftDelimPattern.search(line)
            leftDelim = leftDelim.start() if leftDelim else -1
            rightColon = line.find(': ')

            if (line[leftDelim + 1:rightColon].strip() == userName):
                if not myMsg:
                    startMsgIdx = idx - 1
                myMsg += line[rightColon + 1:].strip()

            elif myMsg:
                for counter in range(startMsgIdx, 0, -1):
                    currLine = lines[counter]
                    leftDelim = leftDelimPattern.search(currLine)
                    leftDelim = leftDelim.start() if leftDelim else -1
                    rightColon = line.find(': ')
                    if (leftDelim < 0 or rightColon < 0):
                        myMsg, otherMsg, currentSpeaker = "", "", ""
                        break
                    if not currentSpeaker:
                        currentSpeaker = currLine[leftDelim + 1:rightColon].strip()
                    elif (currentSpeaker != currLine[leftDelim + 1:rightColon].strip()):
                        otherMsg = cleanMessage(otherMsg)
                        myMsg = cleanMessage(myMsg)
                        convoDict[otherMsg] = myMsg
                        break
                    otherMsg = currLine[rightColon + 1:].strip() + otherMsg
                myMsg, otherMsg, currentSpeaker = "", "", ""
    return convoDict

def getWhatsAppData():
    userName = input('Enter your full WhatsApp name: ')
    if os.path.isfile('whatsapp_chats.csv'):
        return getWhatsAppDataCSV(userName)
    else:
        return getWhatsAppDataTXT(userName)

# ... other functions unchanged (Google, Facebook, LinkedIn, Discord) ...

def cleanMessage(message):
    cleanedMsg = message.replace('\n',' ').lower()
    cleanedMsg = cleanedMsg.replace("\xc2\xa0", "")
    cleanedMsg = re.sub('([.,!?])','', cleanedMsg)
    cleanedMsg = re.sub(' +',' ', cleanedMsg)
    return cleanedMsg

finalDictionary = {}
if (gHangoutsFlag == 'y'):
    print('Getting Google Hangout Data')
    finalDictionary.update(getGoogleHangoutsData())
if (facebookFlag == 'y'):
    print('Getting Facebook Data')
    finalDictionary.update(getFacebookData())
if (linkedinFlag == 'y'):
    print('Getting LinkedIn Data')
    finalDictionary.update(getLinkedInData())
if (whatsappFlag == 'y'):
    print('Getting WhatsApp Data')
    finalDictionary.update(getWhatsAppData())
if (discordFlag == 'y'):
    print('Getting Discord Data')
    finalDictionary.update(getDiscordData())

print('Total length of dictionary:', len(finalDictionary))

print('Saving conversation data dictionary')
np.save('conversationDictionary.npy', finalDictionary)

with open('conversationData.txt', 'w', encoding="utf-8") as convFile:
    for key, val in finalDictionary.items():
        if (not key.strip() or not val.strip()):
            continue
        convFile.write(key.strip() + val.strip())
