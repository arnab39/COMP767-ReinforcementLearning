import os, sys
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from bairdCounterexample import BairdCounterexample

class OffPolicyTD():
	def __init__(self,environment,behaviorPolicy=None,targetPolicy=None,functionApprox='linear'):
		self.environment = environment
		self.gamma = environment.gamma
		if behaviorPolicy is not None:
			self.behaviorPolicy = behaviorPolicy
		else:
			self.behaviorPolicy = np.ones((environment.num_states,environment.num_actions))/environment.num_actions
		if targetPolicy is not None:
			self.targetPolicy = targetPolicy
		else:
			self.targetPolicy = np.ones((environment.num_states,environment.num_actions))/environment.num_actions
		self.functionApprox = functionApprox

	def get_action_from_policy(self,policy,curr_state):
		prob_cumsum = np.cumsum(policy[curr_state])
		p = np.random.uniform(0,1)
		action = np.where(prob_cumsum>p)[0][0]
		return action

	def estimate_value_function_approx(self,alpha,lamda,init_weights=None,episodes=25,return_trajectory=False,verbose=False):
		if self.functionApprox == 'linear':
			if init_weights is None:
				init_weights = np.zeros(self.environment.get_state_representation().shape)
			assert init_weights.shape==self.environment.get_state_representation().shape, "Initial weight array should be of shape {}".format(self.environment.get_state_representation().shape)
		else:
			raise NotImplementedError
		if return_trajectory:
			weights, weights_trajectory = self.TD_lambda(init_weights,alpha,lamda,episodes,return_trajectory,verbose)
			return weights, weights_trajectory
		else:
			weights = self.TD_lambda(init_weights,alpha,lamda,episodes,return_trajectory,verbose)
			return weights

	def apply_function_approx(self,func,state_rep):
		if self.functionApprox=='linear':
			return np.sum(func*state_rep)
		else:
			raise NotImplementedError

	def TD_lambda(self,func,alpha,lamda,episodes=25,return_trajectory=False,verbose=False):
		if return_trajectory:
			func_trajectory = [func.copy()]
		for e in range(episodes):
			if verbose:
					print("Starting episode {}".format(e+1))
			self.environment.reset()
			max_T = 10		# max time steps to run an episode
			eligibility_trace = np.zeros(func.shape)
			for iter in range(max_T):
				curr_state_rep = self.environment.get_state_representation()
				if self.behaviorPolicy is None:
					action = None
				else:
					action = self.get_action_from_policy(self.behaviorPolicy,self.environment.curr_state)
				imp_sampling_ratio = self.targetPolicy[self.environment.curr_state][action]/self.behaviorPolicy[self.environment.curr_state][action]
				reward,is_terminated = self.environment.takeAction(action)
				next_state_rep = self.environment.get_state_representation()
				if self.functionApprox=='linear':
					weight_derivative = curr_state_rep
				else:
					raise NotImplementedError
				eligibility_trace = lamda*self.environment.gamma*eligibility_trace + weight_derivative
				TD_error = reward + self.environment.gamma*self.apply_function_approx(func,next_state_rep) - self.apply_function_approx(func,curr_state_rep)
				if verbose:
					print(TD_error,np.max(eligibility_trace))
					plt.subplot(412);plt.plot(func); # plt.show()
					plt.subplot(413);plt.plot(eligibility_trace); # plt.show()
					plt.subplot(411);plt.plot(curr_state_rep); # plt.show()
					plt.subplot(414);plt.plot(next_state_rep); plt.show()
				func += alpha*imp_sampling_ratio*TD_error*eligibility_trace
				if return_trajectory:
					func_trajectory.append(func.copy())
				if is_terminated:
					break
		if return_trajectory:
			return func, func_trajectory
		else:
			return func

	def plot_estimated_value_func(self,func):
		lower_limit = self.environment.state_min
		upper_limit = self.environment.state_max
		# sampled_states = np.random.uniform(lower_limit,upper_limit,(50,))
		sampled_states = np.linspace(lower_limit,upper_limit,self.environment.num_states)
		estimated_val = [self.apply_function_approx(func,self.environment.get_state_representation(s)) for s in sampled_states]
		plt.plot(sampled_states,estimated_val,'*')
		plt.show()

	def calculate_value_func_MSE(self,func):
		lower_limit = self.environment.state_min
		upper_limit = self.environment.state_max
		# sampled_states = np.random.uniform(lower_limit,upper_limit,(50,))
		sampled_states = np.linspace(lower_limit,upper_limit,self.environment.num_states)
		estimated_val = np.array([self.apply_function_approx(func,self.environment.get_state_representation(s)) for s in sampled_states])
		mse = np.mean((0-estimated_val)**2)
		return mse

if __name__=='__main__':
	alpha = 0.01
	lamda = 0.
	# colors = ['darkviolet','blue','green','gold','darkorange','red','gray','black']
	colors = ['magenta','pink','blue','cyan','black','purple','darkorange','brown']
	seeds = 20
	num_states = 7
	behaviorPolicy = np.tile(np.array([6/7,1/7]).reshape((1,2)),(num_states,1))
	targetPolicy = np.tile(np.array([0,1]).reshape((1,2)),(num_states,1))
	init_weights = np.array([1,1,1,1,1,1,10,1]).astype('float64')
	all_weights_trajectory = []
	min_steps = None
	for seed in tqdm(range(seeds)):
		B = BairdCounterexample(num_states=num_states,gamma=0.99,seed=seed)
		offPolicy_TD = OffPolicyTD(environment=B,behaviorPolicy=behaviorPolicy,targetPolicy=targetPolicy,functionApprox='linear')
		val_func_estimator,weights_array_trajectory = offPolicy_TD.estimate_value_function_approx(alpha=alpha,lamda=lamda,init_weights=init_weights.copy(),episodes=200,return_trajectory=True,verbose=False)	
		# offPolicy_TD.plot_estimated_value_func(SCC,val_func_estimator)
		weights_array_trajectory = np.array(weights_array_trajectory)
		# tqdm.write("{} {} {:.4f} {:.4f} {}".format(weights_array_trajectory.shape,weights_array_trajectory.shape[0],weights_array_trajectory.min(),weights_array_trajectory.max(),min_steps))
		if min_steps is None or weights_array_trajectory.shape[0]<min_steps:
			min_steps = weights_array_trajectory.shape[0]
		all_weights_trajectory.append(weights_array_trajectory)
	for seed in range(seeds):
		all_weights_trajectory[seed] = all_weights_trajectory[seed][:min_steps]
	all_weights_trajectory = np.array(all_weights_trajectory)
	# print(all_weights_trajectory.shape)
	for w in range(all_weights_trajectory.shape[2]):
		mean_weight_trajectory = np.mean(all_weights_trajectory[:,:,w],axis=0)
		std_weight_trajectory = np.std(all_weights_trajectory[:,:,w],axis=0)
		plt.plot(np.linspace(0,min_steps-1,min_steps),mean_weight_trajectory,label='w{}'.format(w+1),color=colors[w])
		plt.fill_between(np.linspace(0,min_steps-1,min_steps),mean_weight_trajectory-std_weight_trajectory,mean_weight_trajectory+std_weight_trajectory,color=colors[w],alpha=0.4)
	plt.legend(fontsize=14)
	plt.xticks(fontsize=13)
	plt.yticks(fontsize=13)
	plt.xlabel('Steps',size=14)
	plt.ylabel('Weight values',size=14)
	plt.title('Evolution of weights in Semi-gradient Off-policy TD($\lambda$={})'.format(lamda),fontsize=16)
	plt.show()

