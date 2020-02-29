import os, sys
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import gym

class MountainCar():
	def __init__(self,gamma=1,seed=13):
		self.gamma = gamma
		self.seed = seed
		np.random.seed(seed)
		self.environment = gym.make('MountainCarContinuous-v0')
		self.environment.seed(seed)
		self.environment.reset()
		self.curr_state = self.environment.state
		self.state_min = self.environment.low_state
		self.state_max = self.environment.high_state
		self.actions = [-1,0,1]

	def getState(self):
		self.curr_state = self.environment.state
		return self.curr_state

	def getNormalizedState(self):
		self.curr_state = self.environment.state
		return (self.curr_state-self.state_min)/(self.state_max-self.state_min)

	def takeAction(self,action=None):
		if action==None:
			# random policy
			action = self.actions[np.random.randint(0,len(self.actions))]
		gym_ret = self.environment.step([action])
		self.curr_state = gym_ret[0]
		reward = gym_ret[1]
		is_terminated = gym_ret[2]
		if not is_terminated:
			# deduct reward for spending time, with each passing time when episode is not terminated, -1 reward is obtained
			reward -=1
		return reward, is_terminated

	def reset(self):
		self.environment.reset()
		self.curr_state = self.environment.state

	def runEpisode_randomPolicy(self,num_steps=10,verbose=0):
		for s in range(num_steps):
			r,is_terminated = self.takeAction()
			if verbose:
				print(r,is_terminated,self.curr_state)
			if is_terminated:
				self.reset()