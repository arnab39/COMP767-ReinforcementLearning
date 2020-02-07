import os, sys
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d
from KarmedTestbed import KarmedTestbed
from BanditSamplingMethod import BanditSamplingMethod

class BoltzmannExploration(BanditSamplingMethod):
	def __init__(self,bandit,temp=100,seed=13):
		super().__init__(bandit=bandit,seed=seed)
		self.Q = np.zeros((self.k,))
		self.temp = temp

	def explore(self):
		'''
		defines the Boltzmann exploration strategy. The probability of choosing arm i is proportional to exp(Q_i/T). 
		A random number is sampled and used to select the arm from the prob distribution
		arguments:
			
		return:
			action - chosen action after sampling from the prob distribution
		'''
		curr_temp = self.temp/(1.+self.steps)
		H = np.exp(self.Q/curr_temp)
		prob = H/np.sum(H)
		prob_cumsum = np.cumsum(prob)
		p = np.random.uniform(0,1)
		action = np.where(prob_cumsum>p)[0][0]
		return action

	def exploit(self):
		'''
		defines the Boltzmann exploitation strategy. The arm with the highest estimate of mean is chosen (greedy)
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
		

def evaluate_BoltzmannExploration(bandit,temp,repeats=10,total_steps=1000,train_steps=10,test_steps=5):
	'''
	Evaluates the Boltzmann exploration strategy for different seeds on the same bandit problem and returns performance metrics
	arguments:
		bandit - a KarmedTestbed class instance, defining the bandit problem
		temp - starting temperature for the Boltzmann sampling algorithm
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
		B = BoltzmannExploration(bandit=bandit,temp=temp,seed=r)
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

def plot_Boltzmann_hyperparam(repeats,train_steps,test_steps,total_steps,temp_range,smooth=True):
	'''
	Plots the performance metrics of Boltzmann exploration strategy for different initial Temperature values
	arguments:
		repeats - number of seeds to repeat the experiment for
		train_steps - number of steps to train/update beliefs about arm distributions
		test_steps - number of steps to test the return for the best estimated arm/action
		total_steps - total number of steps to run the experiment
		temp_range - list of values for starting temperature to evaluate the Boltzmann sampling algorithm
		smooth - True/False flag to indicate smoothing the plots using gaussian_filter1d
	return:

	'''
	bandit = KarmedTestbed(k=10)
	for t in temp_range:
		train_return_arr,test_return_arr,regret_arr, optimal_action_arr = evaluate_BoltzmannExploration(bandit=bandit,temp=t,repeats=repeats,total_steps=total_steps,train_steps=train_steps,test_steps=test_steps)
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
		plt.plot(avg_train_return,label='Boltzmann T='+str(t))
		plt.fill_between(np.linspace(1,avg_train_return.shape[0],avg_train_return.shape[0]),avg_train_return-sterr_train_return,avg_train_return+sterr_train_return,alpha=0.4)
		plt.figure(2)
		plt.plot(np.linspace(train_steps,train_steps*(total_steps//(train_steps+test_steps)),total_steps//(train_steps+test_steps)),avg_test_return,label='Boltzmann T='+str(t))
		plt.fill_between(np.linspace(train_steps,train_steps*(total_steps//(train_steps+test_steps)),total_steps//(train_steps+test_steps)),avg_test_return-sterr_test_return,avg_test_return+sterr_test_return,alpha=0.4)
		plt.figure(3)
		# breakpoint()
		all_idx = np.linspace(0,total_steps-1,total_steps).astype(int)
		testing_idx = np.array([i for i in range(total_steps) if i%(train_steps+test_steps)>=train_steps])
		training_idx = np.setdiff1d(all_idx,testing_idx)
		p = plt.plot(training_idx,avg_regret[training_idx],label='Boltzmann T='+str(t)+'(Train)')
		plt.fill_between(training_idx,avg_regret[training_idx]-sterr_regret[training_idx],avg_regret[training_idx]+sterr_regret[training_idx],alpha=0.4)
		plt.errorbar(testing_idx,avg_regret[testing_idx],yerr=sterr_regret[testing_idx],linestyle='None',marker='*',markersize=10,color=p[0].get_color(),label='Boltzmann T='+str(t)+'(Test)')
		plt.figure(4)
		p = plt.plot(training_idx,avg_optimal_action_percent[training_idx],label='Boltzmann T='+str(t)+'(Train)')
		plt.fill_between(training_idx,avg_optimal_action_percent[training_idx]-sterr_optimal_action_percent[training_idx],avg_optimal_action_percent[training_idx]+sterr_optimal_action_percent[training_idx],alpha=0.4)
		plt.errorbar(testing_idx,avg_optimal_action_percent[testing_idx],yerr=sterr_optimal_action_percent[testing_idx],linestyle='None',marker='*',markersize=10,color=p[0].get_color(),label='Boltzmann T='+str(t)+'(Test)')
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
	plot_Boltzmann_hyperparam(repeats=50,train_steps=10,test_steps=5,total_steps=1000,temp_range=[10,50,100,500,1000],smooth=True)