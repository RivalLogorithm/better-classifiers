from evolutional.geneticalgorithm import GeneticAlgorithm


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

# def optimize(algorithm, function, **kwargs):
#     """
#     Optimize a function using a given algorithm.
#
#     Parameters
#     ----------
#     algorithm : str
#         The name of the algorithm to use.
#     function : Function
#         The function to optimize.
#     kwargs : dict
#         Keyword arguments to pass to the algorithm.
#
#     Returns
#     -------
#     result : dict
#         The result of the optimization.
#     """
#     match algorithm:
#         case 'genetic-algorithm':
#             return _optimize_genetic_algorithm(function, **kwargs)


def _optimize_genetic_algorithm(function, **kwargs):
    ga = GeneticAlgorithm(function, **kwargs)
    return ga.run()
