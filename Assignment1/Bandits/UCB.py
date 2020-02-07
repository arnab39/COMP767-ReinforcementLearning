import os, sys
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d
from KarmedTestbed import KarmedTestbed
from BanditSamplingMethod import BanditSamplingMethod

class UCBSampling(BanditSamplingMethod):
	def __init__(self,bandit,c=0.1,seed=13):
		super().__init__(bandit=bandit,seed=seed)
		self.Q = np.zeros((self.k,))
		self.c = c

	def explore(self):
		'''
		defines the UCB exploration strategy. The arm with the highest upper confidence bound is chosen
		arguments:
			
		return:
			action - chosen action after sampling from the prob distribution
		'''
		H = self.Q + self.c*np.sqrt(np.log(1.+self.steps)/(0.1+self.trials))
		action = np.argmax(H)
		return action

	def exploit(self):
		'''
		defines the UCB exploitation strategy. The arm with the highest estimate of mean is chosen (greedy)
		arguments:
			
		return:
			action - greedily chosen action
		'''
		return np.argmax(self.Q)

	def performAction(self,a):
		'''
		Performs action a (or pulls arm a) and updates the estimates of arm's returns
		Estimates of Q are updated using the running-mean strategy
		arguments:
			a - action chosen by the algorithm
		return:
			reward - observed reward on performing action a on the k-armed bandit
		'''
		reward = super().performAction(a)
		self.Q[a] = (self.trials[a]-1)*self.Q[a]/self.trials[a] + reward/self.trials[a]
		return reward


def evaluate_UCB(bandit,c,repeats=10,total_steps=1000,train_steps=10,test_steps=5):
	'''
	Evaluates the UCB exploration strategy for different seeds on the same bandit problem and returns performance metrics
	arguments:
		bandit - a KarmedTestbed class instance, defining the bandit problem
		c - value of c for the UCB sampling algorithm
		repeats - number of seeds to repeat the experiment for
		total_steps - total number of steps to run the experiment
		train_steps - number of steps to train/update beliefs about arm distributions
		test_steps - number of steps to test the return for the best estimated arm/action
	return:
		train_return_arr - numpy array containing returns obtained on training steps in the experiment
		test_return_arr - numpy array containing returns obtained on testing steps in the experiment
		regret_arr - numpy array containing regrets obtained at each step in the experiment
		optimal_action_arr - numpy array containing if the optimal action was chosen at each step in the experiment
	'''
	train_return_arr = []
	test_return_arr = []
	regret_arr = []
	optimal_action_arr = []
	for r in tqdm(range(repeats)):
		B = UCBSampling(bandit=bandit,c=c,seed=r)
		train_return, test_return, regret, optimal_action_chosen = B.performance(steps=total_steps, train_steps=train_steps, test_steps=test_steps)
		train_return_arr.append(train_return)
		test_return_arr.append(test_return)
		regret_arr.append(regret)
		optimal_action_arr.append(optimal_action_chosen)
	train_return_arr = np.array(train_return_arr)
	test_return_arr = np.array(test_return_arr)
	regret_arr = np.array(regret_arr)
	optimal_action_arr = np.array(optimal_action_arr)
	return train_return_arr, test_return_arr, regret_arr, optimal_action_arr

def plot_UCB_hyperparam(repeats,train_steps,test_steps,total_steps,c_range,smooth=True):
	'''
	Plots the performance metrics of UCB exploration strategy for different c values
	arguments:
		repeats - number of seeds to repeat the experiment for
		train_steps - number of steps to train/update beliefs about arm distributions
		test_steps - number of steps to test the return for the best estimated arm/action
		total_steps - total number of steps to run the experiment
		c_range - list of values for c to evaluate the UCB sampling algorithm
		smooth - True/False flag to indicate smoothing the plots using gaussian_filter1d
	return:

	'''
	bandit = KarmedTestbed(k=10)
	for c in c_range:
		train_return_arr,test_return_arr,regret_arr, optimal_action_arr = evaluate_UCB(bandit=bandit,c=c,repeats=repeats,total_steps=total_steps,train_steps=train_steps,test_steps=test_steps)
		avg_train_return = np.mean(train_return_arr,axis=0)
		sterr_train_return = np.std(train_return_arr,axis=0)/np.sqrt(np.size(train_return_arr,axis=0))
		avg_test_return = np.mean(test_return_arr,axis=0)
		sterr_test_return = np.std(test_return_arr,axis=0)/np.sqrt(np.size(test_return_arr,axis=0))
		avg_regret = np.mean(regret_arr,axis=0)
		sterr_regret = np.std(regret_arr,axis=0)/np.sqrt(np.size(regret_arr,axis=0))
		avg_optimal_action_percent = np.mean(optimal_action_arr,axis=0)
		sterr_optimal_action_percent = np.std(optimal_action_arr,axis=0)/np.sqrt(np.size(optimal_action_arr,axis=0))
		if smooth:
			avg_train_return = gaussian_filter1d(avg_train_return,sigma=2)
			sterr_train_return = gaussian_filter1d(sterr_train_return,sigma=2)
			avg_test_return = gaussian_filter1d(avg_test_return,sigma=2)
			sterr_test_return = gaussian_filter1d(sterr_test_return,sigma=2)
			avg_regret = gaussian_filter1d(avg_regret,sigma=2)
			sterr_regret = gaussian_filter1d(sterr_regret,sigma=2)
			avg_optimal_action_percent = gaussian_filter1d(avg_optimal_action_percent,sigma=2)
			sterr_optimal_action_percent = gaussian_filter1d(sterr_optimal_action_percent,sigma=2)
		plt.figure(1)
		plt.plot(avg_train_return,label='UCB C='+str(c))
		plt.fill_between(np.linspace(1,avg_train_return.shape[0],avg_train_return.shape[0]),avg_train_return-sterr_train_return,avg_train_return+sterr_train_return,alpha=0.4)
		plt.figure(2)
		plt.plot(np.linspace(train_steps,train_steps*(total_steps//(train_steps+test_steps)),total_steps//(train_steps+test_steps)),avg_test_return,label='UCB C='+str(c))
		plt.fill_between(np.linspace(train_steps,train_steps*(total_steps//(train_steps+test_steps)),total_steps//(train_steps+test_steps)),avg_test_return-sterr_test_return,avg_test_return+sterr_test_return,alpha=0.4)
		plt.figure(3)
		# breakpoint()
		all_idx = np.linspace(0,total_steps-1,total_steps).astype(int)
		testing_idx = np.array([i for i in range(total_steps) if i%(train_steps+test_steps)>=train_steps])
		training_idx = np.setdiff1d(all_idx,testing_idx)
		p = plt.plot(training_idx,avg_regret[training_idx],label='UCB C='+str(c)+'(Train)')
		plt.fill_between(training_idx,avg_regret[training_idx]-sterr_regret[training_idx],avg_regret[training_idx]+sterr_regret[training_idx],alpha=0.4)
		plt.errorbar(testing_idx,avg_regret[testing_idx],yerr=sterr_regret[testing_idx],linestyle='None',marker='*',markersize=10,color=p[0].get_color(),label='UCB C='+str(c)+'(Test)')
		plt.figure(4)
		p = plt.plot(training_idx,avg_optimal_action_percent[training_idx],label='UCB C='+str(c)+'(Train)')
		plt.fill_between(training_idx,avg_optimal_action_percent[training_idx]-sterr_optimal_action_percent[training_idx],avg_optimal_action_percent[training_idx]+sterr_optimal_action_percent[training_idx],alpha=0.4)
		plt.errorbar(testing_idx,avg_optimal_action_percent[testing_idx],yerr=sterr_optimal_action_percent[testing_idx],linestyle='None',marker='*',markersize=10,color=p[0].get_color(),label='UCB C='+str(c)+'(Test)')
	plt.figure(1)
	plt.xticks(fontsize=12)
	plt.yticks(fontsize=12)
	plt.title('Average Training return',size=16)
	plt.xlabel('Steps',size=16)
	plt.ylabel('Average reward',size=16)
	plt.legend()
	plt.figure(2)
	plt.xticks(fontsize=12)
	plt.yticks(fontsize=12)
	plt.title('Average Testing return',size=16)
	plt.xlabel('Steps',size=16)
	plt.ylabel('Average reward',size=16)
	plt.legend()
	plt.figure(3)
	plt.xticks(fontsize=12)
	plt.yticks(fontsize=12)
	plt.title('Average Regret',size=16)
	plt.xlabel('Steps',size=16)
	plt.ylabel('Average regret',size=16)
	plt.legend()
	plt.figure(4)
	plt.xticks(fontsize=12)
	plt.yticks(fontsize=12)
	plt.title('Percentage of optimal action choice',size=18)
	plt.xlabel('Steps',size=16)
	plt.ylabel('% Optimal Action',size=16)
	plt.legend()
	plt.show()

if __name__ =='__main__':
	plot_UCB_hyperparam(repeats=50,train_steps=10,test_steps=5,total_steps=1000,c_range=[0.2,1,5],smooth=True)