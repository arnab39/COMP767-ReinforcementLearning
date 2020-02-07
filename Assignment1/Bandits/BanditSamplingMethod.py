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
		'''
		defines the exploration strategy, different for different Sampling algorithms
		'''
		pass

	def exploit(self):
		'''
		defines the exploitation strategy, different for different Sampling algorithms
		'''
		pass

	def performAction(self,a):
		'''
		takes action a, or pulls arm a and updates the count of the arm being pulled
		arguments:
			a - action
		return:
			reward observed by pulling arm a
		'''
		reward = self.T.actionPerformed(a)
		self.trials[a] += 1
		self.steps+=1
		return reward

	def regret(self,action):
		'''
		returns the theoretical regret when action is taken at some step
		arguments:
			action - chosen action or arm pulled
		return:
			theoretical regret value of choosing action
		'''
		return np.max(self.T.Q)-self.T.Q[action]

	def isBestArmChosen(self,action):
		'''
		returns a binary 0/1 value if best action or optimal arm was chosen at some step
		arguments:
			action - chosen action or arm pulled
		return:
			1 if action is optimal, 0 otherwise
		'''
		if action == np.argmax(self.T.Q):
			return 1
		return 0

	def performance(self,steps=1000,train_steps=10,test_steps=5):
		'''
		returns the train/test rewards, regrets and optimal action chosen metrics for an experiment
		arguments:
			steps - total number of steps to run the experiment
			train_steps - number of steps to train/update beliefs about arm distributions
			test_steps - number of steps to test the return for the best estimated arm/action
		return:
			train_return - numpy array containing returns obtained on training steps in the experiment
			test_return - numpy array containing returns obtained on testing steps in the experiment
			regret_arr - numpy array containing regrets obtained at each step in the experiment
			optimal_action_arr - numpy array containing if the optimal action was chosen at each step in the experiment
		'''
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

