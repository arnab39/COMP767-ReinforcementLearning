import os, sys
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

class BairdCounterexample():
	def __init__(self,num_states=7,gamma=0.99,seed=13):
		self.num_states=num_states
		self.state_min = 0
		self.state_max = num_states-1
		self.gamma = gamma
		self.seed = seed
		np.random.seed(seed)
		self.curr_state = np.random.randint(self.num_states)
		self.num_actions = 2

	def takeAction(self,action=None,verbose=False):
		'''
		Return:
			reward
			True/False flag to indicate if episode terminated (True if episode terminated)
		'''
		if action==None:
			# random policy
			action = np.random.randint(self.num_actions)
		if verbose:
			print(action)
		if action==1:
			self.curr_state = 6
			return 0, True
		else:
			self.curr_state = np.random.randint(6)
			return 0, False

	def reset(self):
		self.curr_state = np.random.randint(self.num_states)

	def runEpisode_randomPolicy(self,num_steps=10,verbose=False):
		for s in range(num_steps):
			r,is_terminated = self.takeAction(verbose=verbose)
			if verbose:
				print(r,is_terminated,self.curr_state)
			if is_terminated:
				self.reset()

	def get_state_representation(self,state=None):
		if state is None:
			state = self.curr_state
		assert state<self.num_states, "State value should be between 0 and {}".format(self.num_states)
		state_rep = np.zeros((self.num_states+1,))
		if state<=self.num_states-2:
			state_rep[state]=2
			state_rep[self.num_states]=1
		else:
			state_rep[state]=1
			state_rep[self.num_states]=2
		return state_rep