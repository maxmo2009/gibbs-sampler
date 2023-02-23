#Gibbs sampler for Latent Dirichlet Allocation (LDA)
#dependcy: nltk, numpy, scipy
#Ref: Parameteres Estimation For Aext Analysis
import os
import numpy as np
import scipy as sp
import nltk
from nltk import word_tokenize
import sys
class gibbssampler(object):


	def __init__(self, K = 10, T = 1,a = 0.1,b=0.1,path = "test_corpus.txt"):
		self.K_topic = int(K)
		self.T_iteration = int(T)
		
		self.path = path
		
		self._alpha = a 
		self._beta = b 
		self.process_raw_text()
		self.indice_generator()
		

	def process_raw_text(self):
		corpus = open(self.path)
		tokens = word_tokenize(corpus.read())
		words = [w.lower() for w in tokens]
		self.vocab = sorted(set(words))
		self.v_sizevocab = len(self.vocab)
		self.n_docs = len(open(self.path).readlines())
		print "# of tokens is " + str(self.v_sizevocab)
		return self.vocab

	def indice_generator(self):
		corpus = open(self.path)
		self.mx = list()
		for d in corpus.readlines():
			d = d.split()
			for i,g in enumerate(d):
				for t, j in enumerate(self.vocab):
					if j == g:
						d[i] = t
			self.mx.append(d)
		return self.mx

	def init_parameters(self):
		self.n_mz = np.zeros((self.n_docs, self.K_topic))
		self.n_zw = np.zeros((self.K_topic, self.v_sizevocab))
		self.n_m = np.zeros((self.n_docs))
		self.n_z = np.zeros((self.K_topic))
		self.topic = {}
		for m in xrange(self.n_docs):
			for i, w in enumerate(self.mx[m]):
				
				z = np.random.randint(self.K_topic)
				self.n_mz[m,z] = self.n_mz[m,z] + 1
				self.n_m[m] = self.n_m[m] + 1
				self.n_zw[z,w] = self.n_zw[z,w] + 1
				self.n_z[z] = self.n_z[z] + 1
				self.topic[(m,i)] = z


	def con_dist_calculator(self, m, w):
		
		eq_l = (self.n_zw[:,w] + self._beta)/(self.n_z + self._beta * self.v_sizevocab)
	
		eq_r = (self.n_mz[m,:] + self._alpha)/(self.n_m[m] + self._alpha * self.K_topic) 
		p_z = eq_r * eq_l
		p_z = p_z / np.sum(p_z)
		return p_z

	def iterate(self):
		for T in xrange(self.T_iteration):
			print "round "+str(T) + " starts"
			for m in xrange(self.n_docs):
				for i, w in enumerate(self.mx[m]):
					z = self.topic[(m,i)]
					self.n_mz[m,z] -= 1
					self.n_m[m] -= 1
					self.n_zw[z,w] -= 1
					self.n_z[z] -= 1
					pz = self.con_dist_calculator(m,w)
					z = np.random.multinomial(1,pz).argmax()
					self.n_mz[m,z] += 1
					self.n_m[m] += 1
					self.n_zw[z,w] += 1
					self.n_z[z] += 1
					self.topic[(m,i)] = z
			#print str(T) +"ends"
		print str(T) + " rounds done!"

	def produce(self,com):
		self.phi =self.n_zw
		self.theta = self.n_mz
		if com == "phi": #topic-word matrix
			for j,x in enumerate(self.n_zw):
				for d,item in enumerate(x):
					p = (item + self._beta)/(self.n_z[j] + self._beta * self.v_sizevocab)
					self.phi[j,d] = p
			return self.phi
		if com == "theta": #doc-topic matrix
			for i,m in enumerate(self.n_m):
				for j,x in enumerate(self.n_mz[i]):
					p = (x+self._alpha)/(m + self._alpha * self.K_topic)
					self.theta[i,j] = p
			#print self.n_mz, self.n_m	
			return self.theta
		else:
			print "invalid command"

	def produce_to_txt(self):
		ftheta = open("theta_output.txt",'w')
		fphi = open("phi_output.txt",'w')
		for m in self.theta:
			for i in m:
				ftheta.write(str(i) + ' ')
			ftheta.write("\n")
		for z in self.phi:
			for j in z:
				fphi.write(str(j) + ' ')
			fphi.write("\n")
		ftheta.close()
		fphi.close()






if __name__ == "__main__":
	pathi = sys.argv[1]
	Ti = sys.argv[2]
	Ki = sys.argv[3]
	gib = gibbssampler(path = pathi,T = Ti, K = Ki)
	gib.init_parameters()
	gib.iterate()
	gib.produce("phi")
	gib.produce("theta")
	gib.produce_to_txt()












