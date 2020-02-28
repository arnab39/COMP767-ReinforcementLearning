import os, sys
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from randomWalker import RandomWalker

class SparseCoarseCoding():
	def __init__(self,dims=1,bins=10,tilings=10,state_interval=[0,1],seed=13):
		self.dims = dims
		self.bins = bins
		self.tilings = tilings
		self.state_min = state_interval[0]
		self.state_max = state_interval[1]
		self.seed = seed
		np.random.seed(seed)
		if self.dims>1:
			raise NotImplementedError
		self.interval = (self.state_max-self.state_min)/(self.bins)
		self.offsets = np.random.rand(self.tilings,)*self.interval
		self.state_space = np.zeros((self.tilings,self.bins+1))

	def getCodedState(self,state):
		state_dim = 1 if np.shape(state)==() else np.shape(state)[0]
		assert state_dim==self.dims, "State to be coded ({}) is of different dimension than the Sparse Coarse Coding object ({})!".format(state_dim,self.dims)
		tile_idx = np.ceil((state-self.offsets)/self.interval).astype(int)
		tile_idx[tile_idx>self.bins]=self.bins	# ensuring any index is not greater than bins+1 (max state idx)
		tile_idx[tile_idx<0]=0						# ensuring any index is not less than 0 (min state idx)
		self.state_space = np.zeros((self.tilings,self.bins+1))
		self.state_space[np.array(list(range(self.tilings))),tile_idx] = 1
		return self.state_space

if __name__=='__main__':
	SCC = SparseCoarseCoding(dims=1,bins=10,tilings=10,state_interval=[0,1])
	print(SCC.offsets)
	R = RandomWalker()
	for steps in range(10):
		r,is_terminated = R.takeAction()
		print(r,is_terminated,R.curr_state)
		sparseState = SCC.getCodedState(R.curr_state)
		print(sparseState)
		if is_terminated:
			R.reset()
	print(SCC.offsets)