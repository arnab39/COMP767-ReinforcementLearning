# Description of files

* [KarmedTestbed.py](Bandits/KarmedTestbed.py) --> Defines the K-armed Bandit problem class, used by all sampling algorithms
* [BanditSamplingMethod.py](Bandits/BanditSamplingMethod.py) --> Parent class for all sampling algorithms
* [BoltzmannExploration.py](Bandits/BoltzmannExploration.py) --> Boltzmann sampling algorithm class with functions for hyperparameter search
* [UCB.py](Bandits/UCB.py) --> Upper Confidence Bound (UCB) sampling algorithm class with functions for hyperparameter search
* [ThompsonSampling.py](Bandits/ThompsonSampling.py) --> Thompson sampling algorithm class with functions for hyperparameter search

-----------------------------
All codes can be run on the [Colab notebook](https://colab.research.google.com/drive/1luPiGZjlXYUABRbHyc0vgUnWloFQafgu#scrollTo=SWc199jpJbGj) to observe/reproduce the results.

-----------------------------
## Key observations & notes
### General notes
* **Regret** is a theoretical bound, calculated using the difference between the true means of the optimal arm and the chosen arm
* Each sampling algorithms explores and updates its estimates for train_steps iterations
* The performance of the algorithm is evaluated by choosing the arm with the best reward estimate in subsequent test_steps iterations

### Boltzmann Sampling
* Initial estimate of reward for each arm = 0
* Probability of choosing an arm is proportional to exp(Q<sub>i</sub>/T)
* Hyperparameter search done for temprature variable, T
* Simulated annealing procedure used to decay T with sampling steps
* Initially all arms are explored with roughly equal probability (high T), eventually the algorithm chooses the arm with the best estimate (low T)
* Experiments are repeated with 
