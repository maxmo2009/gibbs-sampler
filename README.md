###Last update: 2.2.2015
###Gibbs Sampler For LDA
------
#Dependcies:
1. numpy
2. scipy 
3. nltk -- for tokenization and building dictionary
#Command Line Tool
python Gib.py [corpus path] [iteration times] [number of topics]  
#API sample Codes are given below

from Gib import gibbssampler #Put the project in the same folder of your own work

gib = gibbssampler(path = "test.txt", K = 10, T =100)
gib.init_parameters() # randomize parameters 
gib.iterate() # converging
print gib.produce("phi") # return a 2-D ndarray object for word-topic
print gib.produce("theta") # return a 2-D ndarray object for document-topic