import numpy as np
import gym

class FrozenLakeClass():
	def __init__(self,name='FrozenLake-v0',slippery=False,seed=13,gamma=0.9):
		self.env = gym.make(name,is_slippery=slippery)
		self.env.seed(seed)
		self.env.reset()
		self.obs = self.env.observation_space
		self.actions = self.env.action_space
		self.num_states = self.obs.n
		self.num_actions = self.actions.n
		self.transition, self.rewards = self.transition_reward_matrix()
		self.gamma = gamma
		np.random.seed(seed)
		self.max_T = self.num_states*3

	def valid_action(self, a):
		assert self.actions.contains(a), "Invalid action chosen!!"

	def valid_state(self, s):
		assert self.obs.contains(s), "Invalid state reached!!"

	def transition_reward_matrix(self):
		P = np.zeros((self.num_states,self.num_actions,self.num_states))
		R = np.zeros((self.num_states,self.num_actions,self.num_states))
		for s in range(self.num_states):
			for a in range(self.num_actions):
				for entry in self.env.P[s][a]:
					P[s][a][entry[1]] = entry[0]
					R[s][a][entry[1]] = entry[2]

		return P,R

	def run_episode(self, policy):
		attempts = 1
		cumulative_reward = 0
		total_steps = 0
		observation = self.env.reset()
		for t in range(self.max_T):
			prob_action = np.cumsum(policy[observation])
			p = np.random.uniform(0,1)
			action = np.where(prob_action>p)[0][0]
			observation, reward, done, info = self.env.step(action)
			cumulative_reward += (self.gamma**t)*reward
			total_steps += 1
			if done and reward==1:
				break
			elif done and reward==0:
				attempts += 1
				total_steps += self.max_T
				observation = self.env.reset()

		return cumulative_reward, total_steps
