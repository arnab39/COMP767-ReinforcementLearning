import os, sys
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

class KarmedTestbed():
	def __init__(self,k=10):
		self.k = k
		self.Q = np.random.randn(k)

	def actionPerformed(self,a):
		'''
		returns the reward when action a is taken, or arm a is pulled
		arguments:
			a - action
		return:
			reward drawn from the true distribution of arm a
		'''
		assert a in range(0,self.k), "Invalid action chosen. Action must be between 0 and "+str(self.k)
		return self.Q[a]+np.random.randn()

	def plotArmStatistics(self,n=100):
		'''
		plots the reward distribution of all bandit arms as a violin plot
		arguments:
			n - number of samples to be drawn for each arm
		return:
			None
		'''
		rewards = np.zeros((n,self.k))
		for arm in range(self.k):
			for idx in range(n):
				rewards[idx,arm]=self.actionPerformed(arm)
		plt.figure()
		plt.violinplot(rewards,list(range(1,1+self.k)),showmeans=True,showextrema=False)
		plt.xlabel('Action')
		plt.ylabel('Reward Distribution')
		plt.hlines(y=0,xmin=0.5,xmax=10.5,colors='k',linestyles='dashed')
		plt.rc('font',size=30)
		plt.show()