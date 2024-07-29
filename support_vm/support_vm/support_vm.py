

from pickle import FALSE
import sys
import pandas as pd
import numpy as np
import tarfile
from zipfile import ZipFile
import tarfile
from pathlib import Path
import string
import re
import math
import os
from sklearn.svm import SVC 
from sklearn.metrics import classification_report
import time

sortedBaggie = {}


your_path = 'C:\\Users\\djnvk\\Desktop\\Project2\\createTrainingSets\\createTrainingSets\\20_newsgroups'
#your_path = 'C:/Users/djnvk/Desktop/Project2/bagOfWords/bagOfWords/20Groups'

#listPunct = { '[*]', '[-]', '[.]', '[?]', '[\']', '[\"]', '[)]', '[(]', '[!]','[@]','[#]','[$]','[%]','[^]','[&]','[_]','[+]','[=]','[;]','[:]','[,]','[\\]','[/]','[>]','[<]','[\}]','[\{]','[\[]','[\]]'} 
#listPunct = { '[*]', '[-]', '[.]', '[?]', '[!]','[@]','[#]','[$]','[%]','[^]','[_]','[+]','[=]','[;]','[:]','[,]','[\\]','[/]','[>]','[<]','[\}]','[\{]','[\[]','[\]]', '[\']', '[\"]', '[\)]', '[\&]','[\(]'} 

stop_words = ["0o", "0s", "3a", "3b", "3d", "6b", "6o", "a", "a1", "a2", "a3", "a4", "ab", "able", "about", "above", "abst", "ac", "accordance", "according", "accordingly", "across", "act", "actually", "ad", "added", "adj", "ae", "af", "affected", "affecting", "affects", "after", "afterwards", "ag", "again", "against", "ah", "ain", "ain't", "aj", "al", "all", "allow", "allows", "almost", "alone", "along", "already", "also", "although", "always", "am", "among", "amongst", "amoungst", "amount", "an", "and", "announce", "another", "any", "anybody", "anyhow", "anymore", "anyone", "anything", "anyway", "anyways", "anywhere", "ao", "ap", "apart", "apparently", "appear", "appreciate", "appropriate", "approximately", "ar", "are", "aren", "arent", "aren't", "arise", "around", "as", "a's", "aside", "ask", "asking", "associated", "at", "au", "auth", "av", "available", "aw", "away", "awfully", "ax", "ay", "az", "b", "b1", "b2", "b3", "ba", "back", "bc", "bd", "be", "became", "because", "become", "becomes", "becoming", "been", "before", "beforehand", "begin", "beginning", "beginnings", "begins", "behind", "being", "believe", "below", "beside", "besides", "best", "better", "between", "beyond", "bi", "bill", "biol", "bj", "bk", "bl", "bn", "both", "bottom", "bp", "br", "brief", "briefly", "bs", "bt", "bu", "but", "bx", "by", "c", "c1", "c2", "c3", "ca", "call", "came", "can", "cannot", "cant", "can't", "cause", "causes", "cc", "cd", "ce", "certain", "certainly", "cf", "cg", "ch", "changes", "ci", "cit", "cj", "cl", "clearly", "cm", "c'mon", "cn", "co", "com", "come", "comes", "con", "concerning", "consequently", "consider", "considering", "contain", "containing", "contains", "corresponding", "could", "couldn", "couldnt", "couldn't", "course", "cp", "cq", "cr", "cry", "cs", "c's", "ct", "cu", "currently", "cv", "cx", "cy", "cz", "d", "d2", "da", "date", "dc", "dd", "de", "definitely", "describe", "described", "despite", "detail", "df", "di", "did", "didn", "didn't", "different", "dj", "dk", "dl", "do", "does", "doesn", "doesn't", "doing", "don", "done", "don't", "down", "downwards", "dp", "dr", "ds", "dt", "du", "due", "during", "dx", "dy", "e", "e2", "e3", "ea", "each", "ec", "ed", "edu", "ee", "ef", "effect", "eg", "ei", "eight", "eighty", "either", "ej", "el", "eleven", "else", "elsewhere", "em", "empty", "en", "end", "ending", "enough", "entirely", "eo", "ep", "eq", "er", "es", "especially", "est", "et", "et-al", "etc", "eu", "ev", "even", "ever", "every", "everybody", "everyone", "everything", "everywhere", "ex", "exactly", "example", "except", "ey", "f", "f2", "fa", "far", "fc", "few", "ff", "fi", "fifteen", "fifth", "fify", "fill", "find", "fire", "first", "five", "fix", "fj", "fl", "fn", "fo", "followed", "following", "follows", "for", "former", "formerly", "forth", "forty", "found", "four", "fr", "from", "front", "fs", "ft", "fu", "full", "further", "furthermore", "fy", "g", "ga", "gave", "ge", "get", "gets", "getting", "gi", "give", "given", "gives", "giving", "gj", "gl", "go", "goes", "going", "gone", "got", "gotten", "gr", "greetings", "gs", "gy", "h", "h2", "h3", "had", "hadn", "hadn't", "happens", "hardly", "has", "hasn", "hasnt", "hasn't", "have", "haven", "haven't", "having", "he", "hed", "he'd", "he'll", "hello", "help", "hence", "her", "here", "hereafter", "hereby", "herein", "heres", "here's", "hereupon", "hers", "herself", "hes", "he's", "hh", "hi", "hid", "him", "himself", "his", "hither", "hj", "ho", "home", "hopefully", "how", "howbeit", "however", "how's", "hr", "hs", "http", "hu", "hundred", "hy", "i", "i2", "i3", "i4", "i6", "i7", "i8", "ia", "ib", "ibid", "ic", "id", "i'd", "ie", "if", "ig", "ignored", "ih", "ii", "ij", "il", "i'll", "im", "i'm", "immediate", "immediately", "importance", "important", "in", "inasmuch", "inc", "indeed", "index", "indicate", "indicated", "indicates", "information", "inner", "insofar", "instead", "interest", "into", "invention", "inward", "io", "ip", "iq", "ir", "is", "isn", "isn't", "it", "itd", "it'd", "it'll", "its", "it's", "itself", "iv", "i've", "ix", "iy", "iz", "j", "jj", "jr", "js", "jt", "ju", "just", "k", "ke", "keep", "keeps", "kept", "kg", "kj", "km", "know", "known", "knows", "ko", "l", "l2", "la", "largely", "last", "lately", "later", "latter", "latterly", "lb", "lc", "le", "least", "les", "less", "lest", "let", "lets", "let's", "lf", "like", "liked", "likely", "line", "little", "lj", "ll", "ll", "ln", "lo", "look", "looking", "looks", "los", "lr", "ls", "lt", "ltd", "m", "m2", "ma", "made", "mainly", "make", "makes", "many", "may", "maybe", "me", "mean", "means", "meantime", "meanwhile", "merely", "mg", "might", "mightn", "mightn't", "mill", "million", "mine", "miss", "ml", "mn", "mo", "more", "moreover", "most", "mostly", "move", "mr", "mrs", "ms", "mt", "mu", "much", "mug", "must", "mustn", "mustn't", "my", "myself", "n", "n2", "na", "name", "namely", "nay", "nc", "nd", "ne", "near", "nearly", "necessarily", "necessary", "need", "needn", "needn't", "needs", "neither", "never", "nevertheless", "new", "next", "ng", "ni", "nine", "ninety", "nj", "nl", "nn", "no", "nobody", "non", "none", "nonetheless", "noone", "nor", "normally", "nos", "not", "noted", "nothing", "novel", "now", "nowhere", "nr", "ns", "nt", "ny", "o", "oa", "ob", "obtain", "obtained", "obviously", "oc", "od", "of", "off", "often", "og", "oh", "oi", "oj", "ok", "okay", "ol", "old", "om", "omitted", "on", "once", "one", "ones", "only", "onto", "oo", "op", "oq", "or", "ord", "os", "ot", "other", "others", "otherwise", "ou", "ought", "our", "ours", "ourselves", "out", "outside", "over", "overall", "ow", "owing", "own", "ox", "oz", "p", "p1", "p2", "p3", "page", "pagecount", "pages", "par", "part", "particular", "particularly", "pas", "past", "pc", "pd", "pe", "per", "perhaps", "pf", "ph", "pi", "pj", "pk", "pl", "placed", "please", "plus", "pm", "pn", "po", "poorly", "possible", "possibly", "potentially", "pp", "pq", "pr", "predominantly", "present", "presumably", "previously", "primarily", "probably", "promptly", "proud", "provides", "ps", "pt", "pu", "put", "py", "q", "qj", "qu", "que", "quickly", "quite", "qv", "r", "r2", "ra", "ran", "rather", "rc", "rd", "re", "readily", "really", "reasonably", "recent", "recently", "ref", "refs", "regarding", "regardless", "regards", "related", "relatively", "research", "research-articl", "respectively", "resulted", "resulting", "results", "rf", "rh", "ri", "right", "rj", "rl", "rm", "rn", "ro", "rq", "rr", "rs", "rt", "ru", "run", "rv", "ry", "s", "s2", "sa", "said", "same", "saw", "say", "saying", "says", "sc", "sd", "se", "sec", "second", "secondly", "section", "see", "seeing", "seem", "seemed", "seeming", "seems", "seen", "self", "selves", "sensible", "sent", "serious", "seriously", "seven", "several", "sf", "shall", "shan", "shan't", "she", "shed", "she'd", "she'll", "shes", "she's", "should", "shouldn", "shouldn't", "should've", "show", "showed", "shown", "showns", "shows", "si", "side", "significant", "significantly", "similar", "similarly", "since", "sincere", "six", "sixty", "sj", "sl", "slightly", "sm", "sn", "so", "some", "somebody", "somehow", "someone", "somethan", "something", "sometime", "sometimes", "somewhat", "somewhere", "soon", "sorry", "sp", "specifically", "specified", "specify", "specifying", "sq", "sr", "ss", "st", "still", "stop", "strongly", "sub", "substantially", "successfully", "such", "sufficiently", "suggest", "sup", "sure", "sy", "system", "sz", "t", "t1", "t2", "t3", "take", "taken", "taking", "tb", "tc", "td", "te", "tell", "ten", "tends", "tf", "th", "than", "thank", "thanks", "thanx", "that", "that'll", "thats", "that's", "that've", "the", "their", "theirs", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "thered", "therefore", "therein", "there'll", "thereof", "therere", "theres", "there's", "thereto", "thereupon", "there've", "these", "they", "theyd", "they'd", "they'll", "theyre", "they're", "they've", "thickv", "thin", "think", "third", "this", "thorough", "thoroughly", "those", "thou", "though", "thoughh", "thousand", "three", "throug", "through", "throughout", "thru", "thus", "ti", "til", "tip", "tj", "tl", "tm", "tn", "to", "together", "too", "took", "top", "toward", "towards", "tp", "tq", "tr", "tried", "tries", "truly", "try", "trying", "ts", "t's", "tt", "tv", "twelve", "twenty", "twice", "two", "tx", "u", "u201d", "ue", "ui", "uj", "uk", "um", "un", "under", "unfortunately", "unless", "unlike", "unlikely", "until", "unto", "uo", "up", "upon", "ups", "ur", "us", "use", "used", "useful", "usefully", "usefulness", "uses", "using", "usually", "ut", "v", "va", "value", "various", "vd", "ve", "ve", "very", "via", "viz", "vj", "vo", "vol", "vols", "volumtype", "vq", "vs", "vt", "vu", "w", "wa", "want", "wants", "was", "wasn", "wasnt", "wasn't", "way", "we", "wed", "we'd", "welcome", "well", "we'll", "well-b", "went", "were", "we're", "weren", "werent", "weren't", "we've", "what", "whatever", "what'll", "whats", "what's", "when", "whence", "whenever", "when's", "where", "whereafter", "whereas", "whereby", "wherein", "wheres", "where's", "whereupon", "wherever", "whether", "which", "while", "whim", "whither", "who", "whod", "whoever", "whole", "who'll", "whom", "whomever", "whos", "who's", "whose", "why", "why's", "wi", "widely", "will", "willing", "wish", "with", "within", "without", "wo", "won", "wonder", "wont", "won't", "words", "world", "would", "wouldn", "wouldnt", "wouldn't", "www", "x", "x1", "x2", "x3", "xf", "xi", "xj", "xk", "xl", "xn", "xo", "xs", "xt", "xv", "xx", "y", "y2", "yes", "yet", "yj", "yl", "you", "youd", "you'd", "you'll", "your", "youre", "you're", "yours", "yourself", "yourselves", "you've", "yr", "ys", "yt", "z", "zero", "zi", "zz"]

weightThreshHold = 0





#main.py

import numpy as np


class fileDictionary:
    
    def __init__(self, fileLocation):
        self.name = fileLocation.name
        self.wordFrequency = {}
        self.wordScore = {}
        self.removedItems = {}
        self.totalWordCount =0

        self.populateDictionary(fileLocation)
        self.calWordScores()

        self.sortedList = []

    #Parses words out of the file and creates a dictionary of word counts
    def populateDictionary(self, fileLocation):

        fileContents = self.cleanText(fileLocation.read_text())

        listOfWords = fileContents.split(" ")
        for word in listOfWords:
            word = word.lower()
            self.addToDictionary(word)

    def sort(self, baggie):
        for word, value in baggie.items():
            if(word in self.wordFrequency.keys()):
                self.sortedList.append(self.wordFrequency[word])
            else:
                self.sortedList.append(0)



    #calculates the total number of words in the document
    def numberOfWords(self):
        wordCount = 0
        for key,value in self.wordFrequency.items():
            wordCount += value
        self.totalWordCount = wordCount

    #calculates the document word score for each word
    def calWordScores(self):
        self.numberOfWords()
        for key,value in self.wordFrequency.items():
            self.wordScore[key] = math.log10(self.totalWordCount/value)

    #removes all punctuation
    def cleanText(self, text):

        listPunct = { '[*]','[-]', '[.]', '[?]', '[!]','[@]','[#]','[$]','[\%]','[\^]','[_]','[\+]','[\=]','[\;]','[\:]','[,]','[\\\]','[\/]','[\>]','[\<]','[\}]','[\{]','[\[]','[\]]', '[\']', '[\"]', '[\)]', '[\&]','[\(]', '[|]'} 


        result = re.sub(r'[\n]', ' ', text)
        result = re.sub(r'[\t]', ' ', result)

        for punt in listPunct:
            result = re.sub(rf'{punt}', '', result)

        result = re.sub(r' +', ' ', result)
        return result;

    #adds word or increments up word in dictionary
    def addToDictionary(self, word):
        if(word not in stop_words):
            if(word in self.wordFrequency):
                self.wordFrequency[word] += 1
            else:
                self.wordFrequency[word] = 1
        else:
            if(word in self.removedItems):
                self.removedItems[word]  += 1
            else:
                self.removedItems[word]  = 1
            

class catagoryDictionary:

    def __init__(self, catagoryLocation):
        self.name = catagoryLocation.name
        self.listOfFiles = []
        self.fileCount = 0
        self.wordFileFrequency = {}
        self.wordWeightDictionary = {}
        self.totalWordDictionary = {}
        self.catagoryWordDictionary = {}
        self.wordScoreSum = {}
        self.removedItems = {}

        print(f"Populate: {self.name}")
        self.populateFileDictionaries(catagoryLocation)
        print("Compiling removed")
        self.getRemoved()
        print(f"Weight: {self.name}")
        self.calcWordWeight()
        print(f"Word Dict: {self.name}")
        self.populateWordDictionary()

    def addMoreToCatagory(self, catagoryLocation):
        print(f"Populate: {self.name}")
        self.populateFileDictionaries(catagoryLocation)
        print("Compiling removed")
        self.getRemoved()
        print(f"Weight: {self.name}")
        self.calcWordWeight()
        print(f"Word Dict: {self.name}")
        self.populateWordDictionary()

    def returnCatagoryWordDictionary(self):
        return self.catagoryWordDictionary

    def returnRemovedItems(self):
        return self.removedItems


    def populateWordDictionary(self):

        for word, weight in self.wordWeightDictionary.items():
            if(weight > weightThreshHold):
                if(word in self.totalWordDictionary):
                    self.catagoryWordDictionary[word]= self.totalWordDictionary[word]
            else:
                if(word in self.removedItems):
                    self.removedItems[word] += self.totalWordDictionary[word]
                else:
                    self.removedItems[word] = self.totalWordDictionary[word]

    def populateFileDictionaries(self, catagoryLocation):

        for file in Path(catagoryLocation).iterdir():
            self.listOfFiles.append(fileDictionary(file))
            self.calWordFileFrequency(self.listOfFiles[-1])
            self.fileCount += 1


    def getRemoved(self):
        for document in self.listOfFiles:
            for word, frequency in document.removedItems.items():
                if(word in self.removedItems):
                    self.removedItems[word] += frequency
                else:
                    self.removedItems[word] = frequency

    def calWordFileFrequency(self, fileClass):
        for word,value in fileClass.wordFrequency.items():
                if(word in self.wordFileFrequency):
                    self.wordFileFrequency[word] += 1
                else:
                    self.wordFileFrequency[word] = 1

                if(word in self.totalWordDictionary):
                    self.totalWordDictionary[word] += value
                else:
                    self.totalWordDictionary[word] = value

                



    def calcWordWeight(self):
        
        #w = tf X df
        weight = 0
        idf = 0

        tf = 0

        for word, frequency in self.wordFileFrequency.items():

            idf =  math.log10(self.fileCount / frequency)

            tf = 1 + math.log10(self.totalWordDictionary[word])

            weight = tf * idf

            self.wordWeightDictionary[word] = weight 


        
    def __print__(self):
        print(self.catagoryWordDictionary.items())



    def writeOut(self, writeDictionaryOutFile):

        newFile = open(writeDictionaryOutFile, 'w')
        newFile.write(json.dumps(self.catagoryWordDictionary))
        newFile.close()


def addDicts(dict1, dict2):
    for word, frequency in dict1.items():
        if(word in dict2):
            dict2[word] += frequency
        else:
            dict2[word] = frequency
    return dict2

def writeToFile(fileLocation, dictionary):

    file = open(fileLocation, 'w')

    for word, frequency in dictionary.items():
        file.write(f"{word}: {frequency}\n")

    file.close()

def main():

    #print(f"first: {sys.argv[1]}\n second: {sys.argv[1]}")

    trainingFilePath = "C:/Users/djnvk/Desktop/DataMiningProject2/20_newsgroups_Test"
    testingFilePath = "C:/Users/djnvk/Desktop/DataMiningProject2/20_newsgroups_Train"

   
    listOfCatagories = []
    listOfTesting = []
    listOfTraining = []

    removedItems = {}
    bagOfWords = {}

    catagoryCount = 0
    for catagory in Path(trainingFilePath).iterdir():
        listOfCatagories.append(catagoryDictionary(catagory))
        catagoryCount += 1

        for document in Path(catagory).iterdir():
            listOfTraining.append(document.name)
        #removedItems = addDicts(listOfCatagories[0].returnRemovedItems(), removedItems)

        bagOfWords = addDicts(listOfCatagories[0].returnCatagoryWordDictionary(), bagOfWords)
        

    catagoryCount = 0  
    for catagory in Path(testingFilePath).iterdir():
        listOfCatagories[catagoryCount].addMoreToCatagory(catagory)
        catagoryCount += 1

        for document in Path(catagory).iterdir():
            listOfTesting.append(document.name)
        
        #removedItems = addDicts(listOfCatagories[0].returnRemovedItems(), removedItems)

        bagOfWords = addDicts(listOfCatagories[0].returnCatagoryWordDictionary(), bagOfWords)
        

    #print(f"Removed: {len(removedItems.keys())} words")
    #print(f"There are: {len(bagOfWords.keys())} words in baggie")

    sortedBaggie = dict(sorted(bagOfWords.items(),key= lambda x:x[1]))

    #sortedRemoved = dict(sorted(removedItems.items(),key= lambda x:x[1]))

    #return listOfCatagories
    #writeToFile("BagofWords.txt", sortedBaggie)
    #writeToFile("eliminate.txt", sortedRemoved)

    td_array_train = []
    td_array_test = []
    answer_array_train = []
    answer_array_test = []


    index = 0
    for classCat in listOfCatagories:
        
        for classDocument in classCat.listOfFiles:
            classDocument.sort(bagOfWords)
            if(classDocument.name in listOfTraining):
                td_array_train.append(classDocument.sortedList)
                answer_array_train.append(index)
            if(classDocument.name in listOfTesting):
                td_array_test.append(classDocument.sortedList)
                answer_array_test.append(index)
        index += 1
            #print(len(classDocument.sortedList))

    begin = time.time()

    training_inputs = td_array_train
    training_outputs = answer_array_train

    testing_input =  td_array_test
    testing_outputs = answer_array_test

    model = SVC() 
    model.fit(training_inputs, training_outputs) 
  
    # model prediction results on test data 

    predictions = model.predict(testing_input) 
    print(classification_report(testing_outputs, predictions)) 

    #print(f"Accuracy: %.2f" % (accuracy_score(testing_outputs, predictionResults)))
    end = time.time()

    print(f"\nTime taken: {end-begin}\n")

    print(f"Number of words: {len(sortedBaggie.keys())}")

main()

