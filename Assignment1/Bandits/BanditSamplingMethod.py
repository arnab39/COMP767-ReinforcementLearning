import os, sys
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from KarmedTestbed import KarmedTestbed

class BanditSamplingMethod():
	def __init__(self,bandit,seed=13):
		self.k = bandit.k
		self.T = bandit
		self.trials = np.zeros((self.k,))
		self.steps = 0
		np.random.seed(seed)

	def explore(self):
		pass

	def exploit(self):
		pass

	def performAction(self,a):
		reward = self.T.actionPerformed(a)
		self.trials[a] += 1
		self.steps+=1
		return reward

	def regret(self,action):
		return np.max(self.T.Q)-self.T.Q[action]

	def isBestArmChosen(self,action):
		if action == np.argmax(self.T.Q):
			return 1
		return 0

	def performance(self,steps=1000,train_steps=10,test_steps=5):
		n_steps = train_steps + test_steps
		train_return = []
		test_return = []
		regret_arr = []
		optimal_action_arr = []
		# breakpoint()
		for s in range(steps):
			if s%n_steps<train_steps:
				action = self.explore()
				reward = self.performAction(action)
				train_return.append(reward)
			else:
				action = self.exploit()
				reward = self.performAction(action)
				test_return.append(reward)
			regret_arr.append(self.regret(action))
			optimal_action_arr.append(self.isBestArmChosen(action))
		train_return = np.array(train_return)
		test_return = np.array(test_return)
		test_return = np.reshape(test_return,(-1,test_steps))
		test_return = np.mean(test_return,axis=1)
		regret_arr = np.array(regret_arr)
		optimal_action_arr = np.array(optimal_action_arr)
		return train_return, test_return, regret_arr, optimal_action_arr

