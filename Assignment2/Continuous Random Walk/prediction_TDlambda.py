import os, sys
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from randomWalker import RandomWalker
from sparseCoarseCoding import SparseCoarseCoding

class Prediction_TDLambda():
	def __init__(self,environment,policy=None,functionApprox='linear'):
		self.environment = environment
		self.policy = None
		self.functionApprox = functionApprox

	def flatten_linear_states(self,state_representation):
		return np.ravel(state_representation)

	def estimate_value_function_approx(self,alpha,lamda,episodes=25,seed=13,verbose=False):
		SCC = SparseCoarseCoding(dims=self.environment.dims,bins=10,tilings=10,state_interval=[self.environment.state_min,self.environment.state_max],seed=seed)
		if self.functionApprox == 'linear':
			weights = np.zeros(self.flatten_linear_states(SCC.getCodedState(self.environment.curr_state)).shape)
		else:
			raise NotImplementedError
		weights = self.TD_lambda(SCC,weights,alpha,lamda,episodes,verbose)
		return SCC,weights

	def apply_function_approx(self,func,sparse_state_rep):
		if self.functionApprox=='linear':
			return np.mean(func[self.flatten_linear_states(sparse_state_rep)==1])
		else:
			raise NotImplementedError

	def TD_lambda(self,sparseCoding,func,alpha,lamda,episodes=25,verbose=False):
		for e in range(episodes):
			if verbose:
					print("Starting episode {}".format(e+1))
			self.environment.reset()
			max_T = 100		# max time steps to run an episode
			eligibility_trace = np.zeros(func.shape)
			for iter in range(max_T):
				curr_state_rep = sparseCoding.getCodedState(self.environment.curr_state)
				if self.policy is None:
					action = None
				else:
					action = self.policy[curr_state_rep]
				reward,is_terminated = self.environment.takeAction(action)
				next_state_rep = sparseCoding.getCodedState(self.environment.curr_state)
				if self.functionApprox=='linear':
					weight_derivative = self.flatten_linear_states(curr_state_rep)
				else:
					raise NotImplementedError
				eligibility_trace = lamda*self.environment.gamma*eligibility_trace + weight_derivative
				TD_error = reward + self.environment.gamma*self.apply_function_approx(func,next_state_rep) - self.apply_function_approx(func,curr_state_rep)
				if verbose:
					print(TD_error,np.max(eligibility_trace))
					plt.subplot(312);plt.imshow(func.reshape(10,11)); plt.colorbar();# plt.show()
					plt.subplot(311);plt.imshow(curr_state_rep); plt.colorbar();# plt.show()
					plt.subplot(313);plt.imshow(next_state_rep); plt.colorbar(); plt.show()
					# self.plot_estimated_value_func(sparseCoding,func)
				func += (alpha/sparseCoding.tilings)*TD_error*eligibility_trace
				if is_terminated:
					break
		return func

	def plot_estimated_value_func(self,sparseCoding,func):
		lower_limit = self.environment.state_min
		upper_limit = self.environment.state_max
		sampled_states = np.random.uniform(lower_limit,upper_limit,(50,))
		estimated_val = [self.apply_function_approx(func,sparseCoding.getCodedState(s)) for s in sampled_states]
		plt.plot(sampled_states,estimated_val,'*')
		plt.show()

	def calculate_value_func_MSE(self,sparseCoding,func):
		lower_limit = self.environment.state_min
		upper_limit = self.environment.state_max
		sampled_states = np.linspace(lower_limit,upper_limit,21)
		estimated_val = np.array([self.apply_function_approx(func,sparseCoding.getCodedState(s)) for s in sampled_states])
		mse = np.mean((sampled_states-estimated_val)**2)
		return mse

if __name__=='__main__':
	num_alpha = 25
	num_lamda = 6
	colors = ['darkviolet','blue','green','gold','darkorange','red']
	lamda_range = np.linspace(0,1,num_lamda)
	alpha_range = np.linspace(0,1,num_alpha)
	seeds = 50
	R = RandomWalker()
	pred_TD = Prediction_TDLambda(environment=R,policy=None,functionApprox='linear')
	mean_mse_arr = np.zeros((num_lamda,num_alpha))
	std_mse_arr = np.zeros((num_lamda,num_alpha))
	for l_idx,lamda in tqdm(enumerate(lamda_range)):
		for a_idx,alpha in enumerate(alpha_range):
			mse_arr = np.zeros((seeds,))
			for seed in range(seeds):
				SCC,val_func_estimator = pred_TD.estimate_value_function_approx(alpha=alpha,lamda=lamda,episodes=80,seed=seed,verbose=False)
				mse_arr[seed] = pred_TD.calculate_value_func_MSE(SCC,val_func_estimator)
			mean_mse_arr[l_idx,a_idx] = np.mean(mse_arr)
			std_mse_arr[l_idx,a_idx] = np.std(mse_arr)/np.sqrt(seeds)
	# pred_TD.plot_estimated_value_func(SCC,val_func_estimator)
		plt.plot(alpha_range,mean_mse_arr[l_idx],color=colors[l_idx],label="$\lambda$={:.3f}".format(lamda))
		plt.fill_between(alpha_range,mean_mse_arr[l_idx]-std_mse_arr[l_idx],mean_mse_arr[l_idx]+std_mse_arr[l_idx],color=colors[l_idx],alpha=0.4)
	plt.ylim(top=0.6)
	plt.legend(fontsize=14)
	plt.xticks(fontsize=13)
	plt.yticks(fontsize=13)
	plt.xlabel('$\\alpha$',size=14)
	plt.ylabel('Mean Squared Error',size=14)
	plt.title('Performance of TD($\lambda$) with accumulating traces',fontsize=16)
	plt.show()