import os, sys
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import gym

class FrozenLakeClass():
	def __init__(self,name='FrozenLake-v0',seed=13):
		self.env = gym.make(name)
		self.env.reset()
		self.obs = self.env.observation_space
		self.actions = self.env.action_space
		self.transition, self.rewards = self.transition_reward_matrix()
		np.random.seed(seed)
		self.max_T = self.obs.n*3

	def valid_action(self, a):
		assert self.actions.contains(a), "Invalid action chosen!!"

	def valid_state(self, s):
		assert self.obs.contains(s), "Invalid state reached!!"

	def transition_reward_matrix(self):
		P = np.zeros((self.obs.n,self.actions.n,self.obs.n))
		R = np.zeros((self.obs.n,self.actions.n,self.obs.n))
		for s in range(self.obs.n):
			for a in range(self.actions.n):
				for entry in self.env.P[s][a]:
					P[s][a][entry[1]] = entry[0]
					R[s][a][entry[1]] = entry[2]

		return P,R

	def run_episode(self, policy, gamma=0.9):
		attempts = 1
		cumulative_reward = 0
		total_steps = 0
		observation = self.env.reset()
		for t in range(self.max_T):
			prob_action = np.cumsum(policy[observation])
			p = np.random.uniform(0,1)
			action = np.where(prob_action>p)[0][0]
			observation, reward, done, info = self.env.step(action)
			cumulative_reward += reward
			total_steps += 1
			if done and reward==1:
				break
			elif done and reward==0:
				attempts += 1
				cumulative_reward -= 1
				total_steps += self.max_T
				observation = self.env.reset()

		return cumulative_reward, total_steps

def greedify_policy(Q_vals):
	tolerance = 1e-9
	max_Q = np.max(Q_vals)
	greedy_policy = np.zeros(len(Q_vals))
	for action in range(len(Q_vals)):
		if np.abs(Q_vals[action]-max_Q)<=tolerance:
			greedy_policy[action] = 1.
	greedy_policy /= np.sum(greedy_policy)
	return greedy_policy

def policy_evaluation(environment, Value_fn, policy, gamma=0.9, theta=1e-8):
	while True:
		delta = 0
		for state in range(environment.obs.n):
			Vs = 0
			for action in range(environment.actions.n):
				for next_state in range(environment.obs.n):
					Vs += policy[state][action]*environment.transition[state][action][next_state]*(environment.rewards[state][action][next_state]+gamma*Value_fn[next_state])
			delta = max(delta, np.abs(Vs-Value_fn[state]))
			Value_fn[state] = Vs
		if delta<theta:
			break
	return Value_fn

def policy_improvement(environment, Value_fn, curr_policy, gamma=0.9):
	policy_stable = True
	for state in range(environment.obs.n):
		temp = curr_policy[state].copy()
		q_vals = np.zeros(environment.actions.n)
		for action in range(environment.actions.n):
			for next_state in range(environment.obs.n):
				q_vals[action] += environment.transition[state][action][next_state]*(environment.rewards[state][action][next_state]+gamma*Value_fn[next_state])
		curr_policy[state] = greedify_policy(q_vals)
		if not np.array_equal(curr_policy[state], temp):
			policy_stable = False
	return curr_policy, policy_stable

def policy_iteration(environment, gamma=0.9, theta=1e-8):
	V = np.zeros(environment.obs.n)
	pi = np.ones((environment.obs.n,environment.actions.n))/environment.actions.n 
	policy_stable = False
	while not policy_stable:
		V = policy_evaluation(environment,V,pi,gamma,theta)
		pi, policy_stable = policy_improvement(environment,V,pi,gamma)
		print(pi,policy_stable)
		plt.imshow(V.reshape(4,4)); plt.colorbar(); plt.show()
	return V, pi

def policy_iteration_with_performance(environment, gamma=0.9, theta=1e-8, iterations=100):
	V = np.zeros(environment.obs.n)
	pi = np.ones((environment.obs.n,environment.actions.n))/environment.actions.n 
	cumulative_reward_array = np.zeros(iterations)
	total_steps_array = np.zeros(iterations)
	for iter in tqdm(range(iterations)):
		cumulative_reward_array[iter], total_steps_array[iter] = environment.run_episode(pi,gamma=gamma)
		V = policy_evaluation(environment,V,pi,gamma,theta)
		pi, policy_stable = policy_improvement(environment,V,pi,gamma)
	return cumulative_reward_array, total_steps_array

def plot_performance_policy_iteration(env_class,name):
	cumulative_reward_array_plot = []
	total_steps_array_plot = []
	for seed in range(5):
		np.random.seed(seed)
		F = env_class(name=name,seed=seed)
		# F.env.render()
		cumulative_reward_array, total_steps_array = policy_iteration_with_performance(F,gamma=0.9,theta=1e-9)
		cumulative_reward_array_plot.append(cumulative_reward_array)
		total_steps_array_plot.append(total_steps_array)
	cumulative_reward_array_plot = np.array(cumulative_reward_array_plot)
	cumulative_reward_array_mean = np.mean(cumulative_reward_array_plot,axis=0)
	cumulative_reward_array_std = np.std(cumulative_reward_array_plot,axis=0)
	total_steps_array_plot = np.array(total_steps_array_plot)
	total_steps_array_mean = np.mean(total_steps_array_plot,axis=0)
	total_steps_array_std = np.std(total_steps_array_plot,axis=0)
	plt.figure(1);
	plt.plot(np.linspace(1,len(cumulative_reward_array_mean),len(cumulative_reward_array_mean)),cumulative_reward_array_mean)
	plt.fill_between(np.linspace(1,len(cumulative_reward_array_mean),len(cumulative_reward_array_mean)),cumulative_reward_array_mean-cumulative_reward_array_std,cumulative_reward_array_mean+cumulative_reward_array_std,alpha=0.4)
	plt.title('Cumulative Reward vs episodes with Policy iteration')
	plt.ylabel('Cumulative reward')
	plt.xlabel('Episodes')
	plt.figure(2);
	plt.plot(np.linspace(1,len(total_steps_array_mean),len(total_steps_array_mean)),total_steps_array_mean)
	plt.fill_between(np.linspace(1,len(total_steps_array_mean),len(total_steps_array_mean)),total_steps_array_mean-total_steps_array_std,total_steps_array_mean+total_steps_array_std,alpha=0.4)
	plt.title('Total steps per episode vs episodes with Policy iteration')
	plt.ylabel('Total steps/episode')
	plt.xlabel('Episodes')
	plt.show()

if __name__ == '__main__':
	plot_performance_policy_iteration(env_class=FrozenLakeClass, name='FrozenLake-v0')