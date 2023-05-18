from evolutional.differentialevolution import DifferentialEvolution
from evolutional.geneticalgorithm import GeneticAlgorithm
import matplotlib.pyplot as plt
import numpy as np

class Optimization:
    """
    A class for performing optimization.
    """

    def __init__(self, algorithm, function, **kwargs):
        """
        Initialize an optimization.

        Parameters
        ----------
        algorithm : str
            The name of the algorithm to use.
        function : Function
            The function to optimize.
        kwargs : dict
            Keyword arguments to pass to the algorithm.
        """
        self.algorithm = algorithm
        self.function = function
        self.kwargs = kwargs

    def optimize(self):
        """
        Optimize a function using a given algorithm.

        Returns
        -------
        result : list
            Optimal parameters.
        """
        match self.algorithm:
            case 'genetic-algorithm':
                return _optimize_genetic_algorithm(self.function, **self.kwargs)
            case 'differential-evolution':
                return _optimize_differential_evolution(self.function, **self.kwargs)


def _optimize_genetic_algorithm(function, **kwargs):
    ga = GeneticAlgorithm(function, **kwargs)
    result = ga.run()
    plt.title(function.__name__)
    plt.ylim([np.min(result[2]), np.max(result[2])])
    plt.plot(range(0, result[1]), result[2])
    plt.show()
    return result[0]


def _optimize_differential_evolution(function, **kwargs):
    de = DifferentialEvolution(function, **kwargs)
    result = de.run()
    plt.title(function.__name__)
    plt.ylim([np.min(result[2]) - 0.01, np.max(result[2]) + 0.01])
    plt.plot(range(0, result[1]), result[2])
    plt.show()
    return result[0]
