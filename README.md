# Gibbs Sampler For LDA
### Last Update
1.12.2015
### Dependencies
* Numpy
* Scipy
* NLTK
### Installation
Just put Gib.py in the same path as your project.
### Command Line Tool
```sh
python Gib.py [corpus path] [iteration times] [number of topics]
```
### Sampler API

<pre><code>
from Gib import gibbssampler #Put the project in the same folder of your own work
gib = gibbssampler(path = "test.txt", K = 10, T =100)
gib.init_parameters() # randomize parameters 
gib.iterate() # converging
print gib.produce("phi") # return a 2-D ndarray object for word-topic
print gib.produce("theta") # return a 2-D ndarray object for document-topic
<\code><\pre>