import Value_Iteration as VI
import Policy_Iteration as PI
from FrozenLake import FrozenLakeClass
from Grid import ExampleGridClass

if __name__ == '__main__':
	# PI.policy_iteration(environment=FrozenLakeClass(name='FrozenLake8x8-v0',seed=1))
	# PI.plot_performance_policy_iteration(env_class=FrozenLakeClass, name = "FrozenLake8x8-v0")
	# VI.value_iteration(FrozenLakeClass('FrozenLake-v0',seed=13))
	VI.plot_performance_value_iteration(env_class=FrozenLakeClass, name = "FrozenLake-v0")
	# PI.policy_iteration(environment=ExampleGridClass(seed=1))
	# PI.plot_performance_policy_iteration(env_class=ExampleGridClass)
	# VI.value_iteration(ExampleGridClass(seed=1))
	# VI.plot_performance_value_iteration(env_class=ExampleGridClass)
