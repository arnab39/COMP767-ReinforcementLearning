import numpy as np
import emdp.gridworld as gw

class ExampleGridClass():
	def __init__(self, seed, grid_size=5, gamma=0.9):
		self.grid_size = grid_size
		self.seed = seed
		np.random.seed(self.seed)
		self.terminal_states = [(self.grid_size-1,self.grid_size-1)]
		self.transition = gw.build_simple_grid(self.grid_size,self.terminal_states, p_success=1)
		self.rewards = np.zeros((self.transition.shape[0], self.transition.shape[1]))
		self.start_prob = np.zeros(self.transition.shape[0]) 
		self.gamma = gamma
		self.num_states = self.transition.shape[0]
		self.num_actions = self.transition.shape[1]
		self.set_transition_and_reward()
		self.grid = gw.GridWorldMDP(self.transition, self.rewards, self.gamma, self.start_prob, self.terminal_states, self.grid_size, self.seed)
		self.max_T = self.num_states*3
 	
	def set_transition_and_reward(self):
		self.start_prob[np.random.randint(low=0,high=self.num_states-1)] = 1		
		self.transition[6,:,:] = 0
		self.transition[6,:,21] = 1
		self.rewards[6, 3] = +2
		self.rewards[(self.grid_size-1)*self.grid_size-1, 3] = +10
		self.rewards[self.grid_size*self.grid_size-2, 1] = +1
		return 
	
	def run_episode(self, policy):
		self.grid.reset()
		attempts = 1
		cumulative_reward = 0
		total_steps = 0
		state = self.grid.current_state.argmax()
		for t in range(self.max_T):
			prob_action = np.cumsum(policy[state])
			p = np.random.uniform(0,1)
			action = np.where(prob_action>p)[0][0]
			state, reward, done, _ = self.grid.step(int(action))
			state = state.argmax()
			cumulative_reward += (self.gamma**t)*reward
			total_steps += 1
			if done:
				break

		return cumulative_reward, total_steps
