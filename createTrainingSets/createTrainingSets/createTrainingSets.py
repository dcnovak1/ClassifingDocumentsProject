

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

import os

your_path = 'C:/Users/djnvk/Desktop/Project2/createTrainingSets/createTrainingSets/20_newsgroups'
#your_path = 'C:/Users/djnvk/Desktop/DataMiningProject2/bagOfWords/bagOfWords/20Groups'

#listPunct = { '[*]', '[-]', '[.]', '[?]', '[\']', '[\"]', '[)]', '[(]', '[!]','[@]','[#]','[$]','[%]','[^]','[&]','[_]','[+]','[=]','[;]','[:]','[,]','[\\]','[/]','[>]','[<]','[\}]','[\{]','[\[]','[\]]'} 
#listPunct = { '[*]', '[-]', '[.]', '[?]', '[!]','[@]','[#]','[$]','[%]','[^]','[_]','[+]','[=]','[;]','[:]','[,]','[\\]','[/]','[>]','[<]','[\}]','[\{]','[\[]','[\]]', '[\']', '[\"]', '[\)]', '[\&]','[\(]'} 


def main():

    dataSetDict = {}

    highest = 0

    for dir in Path(your_path).iterdir():

        listOfFiles = []

        count = 0

        for file in Path(dir).iterdir():

            if file.is_file():
                #print(f"dir: {dir.name} and file: {file.name}\n")

                listOfFiles.append(file.name);
                count += 1

        if(highest < count):
            highest = count

        dataSetDict[dir.name] = listOfFiles
                
    #print(dataSetDict)

    for catagory, filesList in dataSetDict.items():
        #print(f"dir: {catagory} and file: {len(filesList)}\n")
        if(len(filesList) < highest):
            addMore =  highest - len(filesList)
            while(addMore > 0):
                filesList.append('N/A');
                addMore -=1

    allData = pd.DataFrame(dataSetDict)

    #print(allData)



    newsgroups_Train = allData.sample(frac=0.6, random_state=25) #20_newsgroups_Train
    newsgroups_Test = allData.drop(newsgroups_Train.index)#20_newsgroups_Test

    print(newsgroups_Train)
    print(newsgroups_Test)
    newFileDir = "C://Users//djnvk//Desktop//DataMiningProject2//"

    count = 1

    for dir in Path(your_path).iterdir():

        for file in Path(dir).iterdir():

            newFileDir = "C://Users//djnvk//Desktop//DataMiningProject2//"

            #print(f"#{count}:  dir: {dir.name} and file: {file.name}\n")
            count += 1

            if file.is_file():
                if file.name in newsgroups_Test.values:

                    newFileDir += "20_newsgroups_Test//" + dir.name + '//' + file.name + '.txt'
                    newfile = open(newFileDir, 'w')
                    
                    #print(newFileDir)


                    newfile.write(file.read_text())

                    newfile.close()
                elif file.name in newsgroups_Train.values:
                    newFileDir += "20_newsgroups_Train//" + dir.name  + '//' + file.name + '.txt'
                    newfile = open(newFileDir, 'w')
                    
                    #print(newFileDir)

                    newfile.write(file.read_text())

                    newfile.close()
                #else:
                    #print(f"{file.name}: not in")


    #adult_Training_Data = adultData.sample(frac=0.8, random_state=25)
    #adult_Testing_Data = adultData.drop(adult_Training_Data.index)




    return 0;


main()
