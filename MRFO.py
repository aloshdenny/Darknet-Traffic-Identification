import numpy as np
from optimizer import Optimizer


class OriginalMRFO(Optimizer):
    """
    The original version of: Manta Ray Foraging Optimization (MRFO)

    Links:
        1. https://doi.org/10.1016/j.engappai.2019.103300

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + somersault_range (float): [1.5, 3], somersault factor that decides the somersault range of manta rays, default=2

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.swarm_based.MRFO import OriginalMRFO
    >>>
    >>> def fitness_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict1 = {
    >>>     "fit_func": fitness_function,
    >>>     "lb": [-10, -15, -4, -2, -8],
    >>>     "ub": [10, 15, 12, 8, 20],
    >>>     "minmax": "min",
    >>> }
    >>>
    >>> epoch = 1000
    >>> pop_size = 50
    >>> somersault_range = 2.0
    >>> model = OriginalMRFO(epoch, pop_size, somersault_range)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Zhao, W., Zhang, Z. and Wang, L., 2020. Manta ray foraging optimization: An effective bio-inspired
    optimizer for engineering applications. Engineering Applications of Artificial Intelligence, 87, p.103300.
    """

    def __init__(self, epoch=10000, pop_size=40, somersault_range=2.5, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            somersault_range (float): somersault factor that decides the somersault range of manta rays, default=2
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.somersault_range = self.validator.check_int("somersault_range", somersault_range, [1.0, 5.0])
        self.set_parameters(["epoch", "pop_size", "somersault_range"])
        self.nfe_per_epoch = 2 * self.pop_size
        self.sort_flag = False

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_new = []
        for idx in range(0, self.pop_size):
            # Cyclone foraging (Eq. 5, 6, 7)
            if np.random.rand() < 0.5:
                r1 = np.random.uniform()
                beta = 2 * np.exp(r1 * (self.epoch - epoch) / self.epoch) * np.sin(2 * np.pi * r1)

                if (epoch + 1) / self.epoch < np.random.rand():
                    x_rand = np.random.uniform(self.problem.lb, self.problem.ub)
                    if idx == 0:
                        x_t1 = x_rand + np.random.uniform() * (x_rand - self.pop[idx][self.ID_POS]) + \
                               beta * (x_rand - self.pop[idx][self.ID_POS])
                    else:
                        x_t1 = x_rand + np.random.uniform() * (self.pop[idx - 1][self.ID_POS] - self.pop[idx][self.ID_POS]) + \
                               beta * (x_rand - self.pop[idx][self.ID_POS])
                else:
                    if idx == 0:
                        x_t1 = self.g_best[self.ID_POS] + np.random.uniform() * (self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS]) + \
                               beta * (self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS])
                    else:
                        x_t1 = self.g_best[self.ID_POS] + np.random.uniform() * (self.pop[idx - 1][self.ID_POS] - self.pop[idx][self.ID_POS]) + \
                               beta * (self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS])
            # Chain foraging (Eq. 1,2)
            else:
                r = np.random.uniform()
                alpha = 2 * r * np.sqrt(np.abs(np.log(r)))
                if idx == 0:
                    x_t1 = self.pop[idx][self.ID_POS] + r * (self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS]) + \
                           alpha * (self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS])
                else:
                    x_t1 = self.pop[idx][self.ID_POS] + r * (self.pop[idx - 1][self.ID_POS] - self.pop[idx][self.ID_POS]) + \
                           alpha * (self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS])
            pos_new = self.amend_position(x_t1, self.problem.lb, self.problem.ub)
            pop_new.append([pos_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                target = self.get_target_wrapper(pos_new)
                self.pop[idx] = self.get_better_solution(self.pop[idx], [pos_new, target])
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_wrapper_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new)
        _, g_best = self.update_global_best_solution(self.pop, save=False)
        pop_child = []
        for idx in range(0, self.pop_size):
            # Somersault foraging   (Eq. 8)
            x_t1 = self.pop[idx][self.ID_POS] + self.somersault_range * \
                   (np.random.uniform() * g_best[self.ID_POS] - np.random.uniform() * self.pop[idx][self.ID_POS])
            pos_new = self.amend_position(x_t1, self.problem.lb, self.problem.ub)
            pop_child.append([pos_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                target = self.get_target_wrapper(pos_new)
                self.pop[idx] = self.get_better_solution(self.pop[idx], [pos_new, target])
        if self.mode in self.AVAILABLE_MODES:
            pop_child = self.update_target_wrapper_population(pop_child)
            self.pop = self.greedy_selection_population(self.pop, pop_child)


class IMRFO(Optimizer):
    """
    The original version of: Manta Ray Foraging Optimization (MRFO)

    Links:
        1. https://doi.org/10.1016/j.engappai.2019.103300

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + somersault_range (float): [1.5, 3], somersault factor that decides the somersault range of manta rays, default=2



    References
    ~~~~~~~~~~
    [1] Zhao, W., Zhang, Z. and Wang, L., 2020. Manta ray foraging optimization: An effective bio-inspired
    optimizer for engineering applications. Engineering Applications of Artificial Intelligence, 87, p.103300.
    """

    def __init__(self, epoch=10000, pop_size=100, somersault_range=2.0, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            somersault_range (float): somersault factor that decides the somersault range of manta rays, default=2
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.somersault_range = self.validator.check_int("somersault_range", somersault_range, [1.0, 5.0])
        self.set_parameters(["epoch", "pop_size", "somersault_range"])
        self.nfe_per_epoch = 2 * self.pop_size
        self.sort_flag = False

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_new = []
        for idx in range(0, self.pop_size):
            # Cyclone foraging (Eq. 5, 6, 7)
            if np.random.rand() < 0.5:
                r1 = np.random.uniform()
                beta = 2 * np.exp(r1 * (self.epoch - epoch) / self.epoch) * np.sin(2 * np.pi * r1)

                if (epoch + 1) / self.epoch < np.random.rand():
                    x_rand = np.random.uniform(self.problem.lb, self.problem.ub)
                    if idx == 0:
                        x_t1 = x_rand + np.random.uniform() * (x_rand - self.pop[idx][self.ID_POS]) + \
                               beta * (x_rand - self.pop[idx][self.ID_POS])
                    else:
                        x_t1 = x_rand + np.random.uniform() * (self.pop[idx - 1][self.ID_POS] - self.pop[idx][self.ID_POS]) + \
                               beta * (x_rand - self.pop[idx][self.ID_POS])
                else:
                    if idx == 0:
                        x_t1 = self.g_best[self.ID_POS] + np.random.uniform() * (self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS]) + \
                               beta * (self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS])
                    else:
                        x_t1 = self.g_best[self.ID_POS] + np.random.uniform() * (self.pop[idx - 1][self.ID_POS] - self.pop[idx][self.ID_POS]) + \
                               beta * (self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS])
            # Chain foraging (Eq. 1,2)
            else:
                r = np.random.uniform()
                alpha = 2 * r * np.sqrt(np.abs(np.log(r)))
                if idx == 0:
                    x_t1 = self.pop[idx][self.ID_POS] + r * (self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS]) + \
                           alpha * (self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS])
                else:
                    x_t1 = self.pop[idx][self.ID_POS] + r * (self.pop[idx - 1][self.ID_POS] - self.pop[idx][self.ID_POS]) + \
                           alpha * (self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS])
            pos_new = self.amend_position(x_t1, self.problem.lb, self.problem.ub)
            pop_new.append([pos_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                target = self.get_target_wrapper(pos_new)
                self.pop[idx] = self.get_better_solution(self.pop[idx], [pos_new, target])
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_wrapper_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new)
        _, g_best = self.update_global_best_solution(self.pop, save=False)
        pop_child = []
        for idx in range(0, self.pop_size):
            # Somersault foraging   (Eq. 8)
            x_t1 = self.pop[idx][self.ID_POS] + self.somersault_range * \
                   (np.random.uniform() * g_best[self.ID_POS] - np.random.uniform() * self.pop[idx][self.ID_POS])
            pos_new = self.amend_position(x_t1, self.problem.lb, self.problem.ub)
            pop_child.append([pos_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                target = self.get_target_wrapper(pos_new)
                self.pop[idx] = self.get_better_solution(self.pop[idx], [pos_new, target])
        if self.mode in self.AVAILABLE_MODES:
            pop_child = self.update_target_wrapper_population(pop_child)
            self.pop = self.greedy_selection_population(self.pop, pop_child)