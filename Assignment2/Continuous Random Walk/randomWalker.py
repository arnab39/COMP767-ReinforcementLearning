import os, sys
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

class RandomWalker():
	def __init__(self,state_interval=[0,1],step_interval=[-0.2,0.2],start_state=0.5,gamma=1,seed=13):
		self.dims=1
		self.state_min = state_interval[0]
		self.state_max = state_interval[1]
		self.step_min = step_interval[0]
		self.step_max = step_interval[1]
		self.start_state = start_state
		self.curr_state = self.start_state
		self.gamma = gamma
		self.seed = seed
		np.random.seed(seed)

	def takeAction(self,action=None):
		'''
		Return:
			reward
			True/False flag to indicate if episode terminated (True if episode terminated)
		'''
		if action==None:
			# random policy
			action = np.random.uniform(self.step_min,self.step_max)
		self.curr_state += action
		if self.curr_state<self.state_min or self.curr_state>self.state_max:
			return self.curr_state, True
		else:
			return 0, False

	def reset(self):
		self.curr_state = self.start_state

	def runEpisode_randomPolicy(self,num_steps=10,verbose=0):
		for s in range(num_steps):
			r,is_terminated = self.takeAction()
			if verbose:
				print(r,is_terminated,self.curr_state)
			if is_terminated:
				self.reset()