import os, sys
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
from mountainCar import MountainCar

class AsymmetricTileCoding():
	def __init__(self,dims=2,bins=8,tilings=8,state_limits=[[0,0],[1,1]],seed=13):
		self.dims = dims
		self.bins = bins
		self.tilings = tilings
		self.states_min = state_limits[0]
		self.states_max = state_limits[1]
		self.seed = seed
		np.random.seed(seed)
		if self.dims!=2:
			raise NotImplementedError
		if self.tilings<4*self.dims:
			warnings.warn("Number of tilings should ideally be higher for proper asymmetric coding",RuntimeWarning)
		self.interval = [(self.states_max[dim]-self.states_min[dim])/(self.bins) for dim in range(self.dims)]
		self.offsets = []
		for dim in range(self.dims):
			dim_displacement = 2*dim+1	# select sequential odd numbers starting from 1
			dim_offsets = self.states_min[dim]+np.linspace(0,dim_displacement*self.interval[dim],self.tilings+1)[:-1]
			dim_offsets = dim_offsets - np.mean(dim_offsets) 	# centering the offsets to have roughly symmetric spanning of the state space
			self.offsets.append(dim_offsets)
		self.offsets = np.array(self.offsets)

	def getCodedState(self,state):
		state_dim = 1 if np.shape(state)==() else np.shape(state)[0]
		assert state_dim==self.dims, "State to be coded ({}) is of different dimension than the Asymmetric Tile Coding object ({})!".format(state_dim,self.dims)
		tile_idxs = []
		for dim in range(self.dims):
			tile_idx = np.floor((state[dim]-self.offsets[dim])/self.interval[dim]).astype(int)
			tile_idx[tile_idx>self.bins-1]=self.bins-1	# ensuring any index is not greater than bins+1 (max state idx)
			tile_idx[tile_idx<0]=0						# ensuring any index is not less than 0 (min state idx)
			tile_idxs.append(tile_idx)
		tile_idxs = np.array(tile_idxs)
		# state_space = np.zeros((self.tilings,)+self.dims*(self.bins,))
		return tile_idxs.T 		# returned numpy array will be of shape (num_tilings,dims) --> each row stores the indices of the bin that should be 1

if __name__=='__main__':
	ATC = AsymmetricTileCoding(dims=2,bins=4,tilings=4,state_limits=[[0,0],[1,1]])
	print(ATC.offsets)
	M = MountainCar()
	for steps in range(10):
		r,is_terminated = M.takeAction(-1)
		print(r,is_terminated,M.curr_state,M.getNormalizedState())
		tileCodes = ATC.getCodedState(M.getNormalizedState())
		print(tileCodes.shape,tileCodes)
		if is_terminated:
			R.reset()
	print(ATC.offsets)