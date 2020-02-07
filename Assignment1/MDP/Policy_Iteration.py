import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from FrozenLake import FrozenLakeClass
from Grid import ExampleGridClass

###########################################
##### ALL FUCTIONS FOR POLICY ITERATION ###
###########################################

def greedify_policy(Q_vals):
	'''
	chooses the optimal action at each step such that the agent moves to the state with the maximum value
	arguments:
		Q_vals - vector that stores the values of next states for each action
	return:
		greedy_policy - the greedified policy, a matrix containing prob of actions
	'''
	tolerance = 1e-9
	max_Q = np.max(Q_vals)
	greedy_policy = np.zeros(len(Q_vals))
	for action in range(len(Q_vals)):
		if np.abs(Q_vals[action]-max_Q)<=tolerance:
			greedy_policy[action] = 1.
	greedy_policy /= np.sum(greedy_policy)
	return greedy_policy


def policy_evaluation(environment, policy, theta=1e-8):
	'''
	Computes the value function for a given policy
	arguments:
		environment - class instance of an MDP environment
		policy - the policy for which value function is to be evaluated
		theta - tolerance level to decide convergence
	return:
		Value_fn - the estimated value of each state, evaluated for given policy
	'''
	Value_fn = np.zeros(environment.num_states)
	while True:
		delta = 0
		for state in range(environment.num_states):
			Vs = 0
			for action in range(environment.num_actions):
				
				for next_state in range(environment.num_states):
					if type(environment).__name__ == 'ExampleGridClass':
						Vs += policy[state][action]*environment.transition[state][action][next_state]*(environment.rewards[state][action]+environment.gamma*Value_fn[next_state])
					else:
						Vs += policy[state][action]*environment.transition[state][action][next_state]*(environment.rewards[state][action][next_state]+environment.gamma*Value_fn[next_state])
			delta = max(delta, np.abs(Vs-Value_fn[state]))
			Value_fn[state] = Vs
		if delta<theta:
			break
	return Value_fn

def policy_improvement(environment, Value_fn, curr_policy):
	'''
	Decides a new policy based on given value function
	arguments:
		environment - class instance of an MDP environment
		Value_fn - estimated value of each state
		curr_policy - present policy
	return:
		curr_policy - updated policy
		policy_stable - True/False flag indicating if policy was changed
	'''
	policy_stable = True
	for state in range(environment.num_states):
		temp = curr_policy[state].copy()
		q_vals = np.zeros(environment.num_actions)
		for action in range(environment.num_actions):
			for next_state in range(environment.num_states):
				if type(environment).__name__ == 'ExampleGridClass':
					q_vals[action] += environment.transition[state][action][next_state]*(environment.rewards[state][action]+environment.gamma*Value_fn[next_state])
				else:
					q_vals[action] += environment.transition[state][action][next_state]*(environment.rewards[state][action][next_state]+environment.gamma*Value_fn[next_state]) 
		curr_policy[state] = greedify_policy(q_vals)
		if not np.array_equal(curr_policy[state], temp):
			policy_stable = False
	return curr_policy, policy_stable

def policy_iteration(environment, theta=1e-8):
	'''
	Runs the entire tabular policy iteration algorithm till convergence
	arguments:
		environment - class instance of an MDP environment
		theta - tolerance level to decide convergence
	return:
		V - value function that the algorithm converged to
		pi - optimal policy that the algorithm converged to
	'''
	V = np.zeros(environment.num_states) 
	pi = np.random.rand(environment.num_states,environment.num_actions)
	pi = pi/pi.sum(axis=1,keepdims=True)
	policy_stable = False
	while not policy_stable:
		V = policy_evaluation(environment,pi,theta)
		pi, policy_stable = policy_improvement(environment,V,pi)
		print(pi,policy_stable)
		if type(environment).__name__ == 'ExampleGridClass':
			plt.imshow(V[:-1].reshape(int(np.sqrt(environment.num_states)),int(np.sqrt(environment.num_states)))); plt.colorbar(); plt.show()
		else:
			plt.imshow(V.reshape(int(np.sqrt(environment.num_states)),int(np.sqrt(environment.num_states)))); plt.colorbar(); plt.show()
	return V, pi


def policy_iteration_with_performance(environment, theta=1e-8, iterations=90, train_steps=1, test_steps=5):
	'''
	Runs the tabular policy iteration algorithm for given iterations and returns cumulative reward and total steps at each iteration
	arguments:
		environment - class instance of an MDP environment
		theta - tolerance level to decide convergence
		iterations - number of iterations to run the algorithm
		train_steps - number of consecutive iterations to update the value function and policy
		test_steps - number of consecutive iterations to evaluate the current policy in the environment
	return:
		cumulative_reward_array - array of cumulative reward at each step (both train and test included)
		total_steps_array - array of total number of steps at each step (both train and test included)
	'''
	n_steps = train_steps+test_steps
	V = np.zeros(environment.num_states)
	pi = np.random.rand(environment.num_states,environment.num_actions)
	pi = pi/pi.sum(axis=1,keepdims=True)
	cumulative_reward_array = np.zeros(iterations)
	total_steps_array = np.zeros(iterations)
	for iter in tqdm(range(iterations)):
		cumulative_reward_array[iter], total_steps_array[iter] = environment.run_episode(pi)
		if (iter%n_steps)>=test_steps:
			V = policy_evaluation(environment,pi,theta)
			pi, policy_stable = policy_improvement(environment,V,pi)
	return cumulative_reward_array, total_steps_array

def plot_performance_policy_iteration(env_class,name='FrozenLake-v0', iterations=90, train_steps=1, test_steps=5):
	'''
	Plots the performance metrics of tabular policy iteration algorithm
	arguments:
		env_class - MDP environment class
		name - name of environment (for FrozenLakeClass)
		iterations - number of iterations to run the algorithm
		train_steps - number of consecutive iterations to update the value function and policy
		test_steps - number of consecutive iterations to evaluate the current policy in the environment
	return:
		
	'''
	cumulative_reward_array_train_plot = []
	cumulative_reward_array_test_plot = []
	total_steps_array_train_plot = []
	total_steps_array_test_plot = []
	for seed in range(5):
		np.random.seed(seed)
		if env_class.__name__ == 'ExampleGridClass':
			env = env_class(seed=seed)
		else:
			env = env_class(seed=seed,name=name)
		cumulative_reward_array, total_steps_array = policy_iteration_with_performance(env,theta=1e-9,iterations=iterations,train_steps=train_steps,test_steps=test_steps)
		all_idx = np.array(list(range(iterations)))
		train_idx = np.array([i for i in range(iterations) if i%(train_steps+test_steps)>=test_steps])
		test_idx = np.setdiff1d(all_idx,train_idx)
		cumulative_reward_array_train = cumulative_reward_array[train_idx]
		cumulative_reward_array_test = cumulative_reward_array[test_idx].reshape(-1,test_steps).mean(axis=1)
		total_steps_array_train = total_steps_array[train_idx]
		total_steps_array_test = total_steps_array[test_idx].reshape(-1,test_steps).mean(axis=1)
		cumulative_reward_array_train_plot.append(cumulative_reward_array_train)
		cumulative_reward_array_test_plot.append(cumulative_reward_array_test)
		total_steps_array_train_plot.append(total_steps_array_train)
		total_steps_array_test_plot.append(total_steps_array_test)
	cumulative_reward_array_train_plot = np.array(cumulative_reward_array_train_plot)
	cumulative_reward_array_test_plot = np.array(cumulative_reward_array_test_plot)
	cumulative_reward_array_train_mean = np.mean(cumulative_reward_array_train_plot,axis=0)
	cumulative_reward_array_train_std = np.std(cumulative_reward_array_train_plot,axis=0)
	cumulative_reward_array_test_mean = np.mean(cumulative_reward_array_test_plot,axis=0)
	cumulative_reward_array_test_std = np.std(cumulative_reward_array_test_plot,axis=0)
	total_steps_array_train_plot = np.array(total_steps_array_train_plot)
	total_steps_array_test_plot = np.array(total_steps_array_test_plot)
	total_steps_array_train_mean = np.mean(total_steps_array_train_plot,axis=0)
	total_steps_array_train_std = np.std(total_steps_array_train_plot,axis=0)
	total_steps_array_test_mean = np.mean(total_steps_array_test_plot,axis=0)
	total_steps_array_test_std = np.std(total_steps_array_test_plot,axis=0)
	plt.figure(1);
	plt.xticks(fontsize=11)
	plt.yticks(fontsize=11)
	plt.plot(train_idx,cumulative_reward_array_train_mean,label='Mean cumulative reward of an episode')
	plt.fill_between(train_idx,cumulative_reward_array_train_mean-cumulative_reward_array_train_std,cumulative_reward_array_train_mean+cumulative_reward_array_train_std,alpha=0.4)
	optimal_reward_train = np.max(cumulative_reward_array_train_plot,axis=1)
	plt.axhline(y=np.mean(optimal_reward_train),color='k',linestyle='--',label='Optimal performance')
	plt.axhspan(np.mean(optimal_reward_train)-np.std(optimal_reward_train),np.mean(optimal_reward_train)+np.std(optimal_reward_train),alpha=0.2,color='k')
	plt.title('Cumulative Reward during training vs episodes with Policy iteration',size=12)
	plt.ylabel('Cumulative reward',size=12)
	plt.xlabel('Episodes',size=12)
	plt.xscale('log')
	plt.legend()
	plt.figure(2);
	plt.xticks(fontsize=11)
	plt.yticks(fontsize=11)
	plt.plot(np.linspace(1,len(cumulative_reward_array_test_mean),len(cumulative_reward_array_test_mean)),cumulative_reward_array_test_mean,label='Mean cumulative reward of an episode')
	plt.fill_between(np.linspace(1,len(cumulative_reward_array_test_mean),len(cumulative_reward_array_test_mean)),cumulative_reward_array_test_mean-cumulative_reward_array_test_std,cumulative_reward_array_test_mean+cumulative_reward_array_test_std,alpha=0.4)
	optimal_reward_test = np.max(cumulative_reward_array_test_plot,axis=1)
	plt.axhline(y=np.mean(optimal_reward_test),color='k',linestyle='--',label='Optimal performance')
	plt.axhspan(np.mean(optimal_reward_test)-np.std(optimal_reward_test),np.mean(optimal_reward_test)+np.std(optimal_reward_test),alpha=0.2,color='k')
	plt.title('Cumulative Reward during testing vs episodes with Policy iteration',size=12)
	plt.ylabel('Cumulative reward',size=12)
	plt.xlabel('Episodes',size=12)
	plt.xscale('log')
	plt.legend()
	plt.figure(3);
	plt.xticks(fontsize=11)
	plt.yticks(fontsize=11)
	plt.plot(train_idx,total_steps_array_train_mean,label='Mean steps of an episode')
	plt.fill_between(train_idx,total_steps_array_train_mean-total_steps_array_train_std,total_steps_array_train_mean+total_steps_array_train_std,alpha=0.4)
	optimal_steps_train = np.min(total_steps_array_train_plot,axis=1)
	plt.axhline(y=np.mean(optimal_steps_train),color='k',linestyle='--',label='Optimal performance')
	plt.axhspan(np.mean(optimal_steps_train)-np.std(optimal_steps_train),np.mean(optimal_steps_train)+np.std(optimal_steps_train),alpha=0.2,color='k')
	plt.title('Total steps per episode during training vs episodes with Policy iteration',size=12)
	plt.ylabel('Total steps/episode',size=12)
	plt.xlabel('Episodes',size=12)
	plt.xscale('log')
	plt.legend()
	plt.figure(4);
	plt.xticks(fontsize=11)
	plt.yticks(fontsize=11)
	plt.plot(np.linspace(1,len(total_steps_array_test_mean),len(total_steps_array_test_mean)),total_steps_array_test_mean,label='Mean steps of an episode')
	plt.fill_between(np.linspace(1,len(total_steps_array_test_mean),len(total_steps_array_test_mean)),total_steps_array_test_mean-total_steps_array_test_std,total_steps_array_test_mean+total_steps_array_test_std,alpha=0.4)
	optimal_steps_test = np.min(total_steps_array_test_plot,axis=1)
	plt.axhline(y=np.mean(optimal_steps_test),color='k',linestyle='--',label='Optimal performance')
	plt.axhspan(np.mean(optimal_steps_test)-np.std(optimal_steps_test),np.mean(optimal_steps_test)+np.std(optimal_steps_test),alpha=0.2,color='k')
	plt.title('Total steps per episode during testing vs episodes with Policy iteration',size=12)
	plt.ylabel('Total steps/episode',size=12)
	plt.xlabel('Episodes',size=12)
	plt.xscale('log')
	plt.legend()
	plt.show()