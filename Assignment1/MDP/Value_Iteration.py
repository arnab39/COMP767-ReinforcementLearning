import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from FrozenLake import FrozenLakeClass
from Grid import ExampleGridClass

###########################################
##### ALL FUCTIONS FOR VALUE ITERATION ###
###########################################

def greedify_policy(Q_vals):
	tolerance = 1e-9
	max_Q = np.max(Q_vals)
	greedy_policy = np.zeros(len(Q_vals))
	for action in range(len(Q_vals)):
		if np.abs(Q_vals[action]-max_Q)<=tolerance:
			greedy_policy[action] = 1.
	greedy_policy /= np.sum(greedy_policy)
	return greedy_policy

def value_iteration(environment, theta=1e-8):
	V = np.zeros(environment.num_states)
	pi = np.random.rand(environment.num_states,environment.num_actions)
	pi = pi/pi.sum(axis=1,keepdims=True)
	while True:
		# plt.imshow(V.reshape(4,4)); plt.colorbar(); plt.show()
		delta = 0
		for state in range(environment.num_states):
			Vs = V[state]
			q_vals = np.zeros(environment.num_actions)
			for action in range(environment.num_actions):
				for next_state in range(environment.num_states):
					if type(environment).__name__ == 'ExampleGridClass':
						q_vals[action] += environment.transition[state][action][next_state]*(environment.rewards[state][action]+environment.gamma*V[next_state])
					else:
						q_vals[action] += environment.transition[state][action][next_state]*(environment.rewards[state][action][next_state]+environment.gamma*V[next_state])
			V[state] = np.max(q_vals)
			delta = max(delta, np.abs(Vs-V[state]))
			pi[state] = greedify_policy(q_vals)
		if delta<theta:
			break
	if type(environment).__name__ == 'ExampleGridClass':
		plt.imshow(V[:-1].reshape(int(np.sqrt(environment.num_states)),int(np.sqrt(environment.num_states)))); plt.colorbar(); plt.show()
	else:
		plt.imshow(V.reshape(int(np.sqrt(environment.num_states)),int(np.sqrt(environment.num_states)))); plt.colorbar(); plt.show()
	return V, pi

def value_iteration_with_performance(environment, theta=1e-8, iterations=90, train_steps=1, test_steps=5):
	n_steps = train_steps+test_steps
	V = np.zeros(environment.num_states)
	pi = np.random.rand(environment.num_states,environment.num_actions)
	pi = pi/pi.sum(axis=1,keepdims=True)
	cumulative_reward_array = np.zeros(iterations)
	total_steps_array = np.zeros(iterations)
	for iter in tqdm(range(iterations)):
		cumulative_reward_array[iter], total_steps_array[iter] = environment.run_episode(pi)
		if (iter%n_steps)>=test_steps:
			for state in range(environment.num_states):
				Vs = V[state]
				q_vals = np.zeros(environment.num_actions)
				for action in range(environment.num_actions):
					for next_state in range(environment.num_states):
						if type(environment).__name__ == 'ExampleGridClass': 
							q_vals[action] += environment.transition[state][action][next_state]*(environment.rewards[state][action]+environment.gamma*V[next_state])
						else:
							q_vals[action] += environment.transition[state][action][next_state]*(environment.rewards[state][action][next_state]+environment.gamma*V[next_state])
				V[state] = np.max(q_vals)
				pi[state] = greedify_policy(q_vals)
	return cumulative_reward_array, total_steps_array

def plot_performance_value_iteration(env_class,name='FrozenLake-v0', iterations=90, train_steps=1, test_steps=5):
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
		cumulative_reward_array, total_steps_array = value_iteration_with_performance(env,theta=1e-9)
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
	plt.plot(train_idx,cumulative_reward_array_train_mean,label='Mean cumulative reward of an episode')
	plt.fill_between(train_idx,cumulative_reward_array_train_mean-cumulative_reward_array_train_std,cumulative_reward_array_train_mean+cumulative_reward_array_train_std,alpha=0.4)
	plt.axhline(y=np.max(cumulative_reward_array_train_plot),color='k',linestyle='--',label='Optimal performance')
	plt.title('Cumulative Reward during training vs episodes with Value iteration')
	plt.ylabel('Cumulative reward')
	plt.xlabel('Episodes')
	plt.xscale('log')
	plt.legend()
	plt.figure(2);
	plt.plot(np.linspace(1,len(cumulative_reward_array_test_mean),len(cumulative_reward_array_test_mean)),cumulative_reward_array_test_mean,label='Mean cumulative reward of an episode')
	plt.fill_between(np.linspace(1,len(cumulative_reward_array_test_mean),len(cumulative_reward_array_test_mean)),cumulative_reward_array_test_mean-cumulative_reward_array_test_std,cumulative_reward_array_test_mean+cumulative_reward_array_test_std,alpha=0.4)
	plt.axhline(y=np.max(cumulative_reward_array_test_plot),color='k',linestyle='--',label='Optimal performance')
	plt.title('Cumulative Reward during testing vs episodes with Value iteration')
	plt.ylabel('Cumulative reward')
	plt.xlabel('Episodes')
	plt.xscale('log')
	plt.legend()
	plt.figure(3);
	plt.plot(train_idx,total_steps_array_train_mean,label='Mean steps of an episode')
	plt.fill_between(train_idx,total_steps_array_train_mean-total_steps_array_train_std,total_steps_array_train_mean+total_steps_array_train_std,alpha=0.4)
	plt.axhline(y=np.min(total_steps_array_train_plot),color='k',linestyle='--',label='Optimal performance')
	plt.title('Total steps per episode during training vs episodes with Value iteration')
	plt.ylabel('Total steps/episode')
	plt.xlabel('Episodes')
	plt.xscale('log')
	plt.legend()
	plt.figure(4);
	plt.plot(np.linspace(1,len(total_steps_array_test_mean),len(total_steps_array_test_mean)),total_steps_array_test_mean,label='Mean steps of an episode')
	plt.fill_between(np.linspace(1,len(total_steps_array_test_mean),len(total_steps_array_test_mean)),total_steps_array_test_mean-total_steps_array_test_std,total_steps_array_test_mean+total_steps_array_test_std,alpha=0.4)
	plt.axhline(y=np.min(total_steps_array_test_plot),color='k',linestyle='--',label='Optimal performance')
	plt.title('Total steps per episode during testing vs episodes with Value iteration')
	plt.ylabel('Total steps/episode')
	plt.xlabel('Episodes')
	plt.xscale('log')
	plt.legend()
	plt.show()
