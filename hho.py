import numpy as np

class HarrisHawkOptimization:
    def __init__(self, fitness_function, n_hawks, dim, max_iter, lb, ub):
        """
        Harris Hawk Optimization (HHO) Algorithm

        Parameters:
            fitness_function (callable): 要优化的目标函数，返回 (score, params)
            n_hawks (int): 鹰群数量
            dim (int): 参数维度
            max_iter (int): 最大迭代次数
            lb (list or np.ndarray): 每个参数的下界
            ub (list or np.ndarray): 每个参数的上界
        """
        self.fitness_function = fitness_function
        self.n_hawks = n_hawks
        self.dim = dim
        self.max_iter = max_iter
        self.lb = np.array(lb)
        self.ub = np.array(ub)
        self.stagnation_count = 0

    def optimize(self, verbose=True):
        hawks = np.random.uniform(self.lb, self.ub, (self.n_hawks, self.dim))
        fitness = np.zeros(self.n_hawks)
        params_list = [None] * self.n_hawks

        for i in range(self.n_hawks):
            fitness[i], params_list[i] = self.fitness_function(hawks[i])

        best_idx = np.argmin(fitness)
        rabbit_pos = hawks[best_idx].copy()
        rabbit_energy = fitness[best_idx]
        best_params = params_list[best_idx]

        global_best_pos = rabbit_pos.copy()
        global_best_energy = rabbit_energy
        global_best_params = best_params

        last_rabbit_energy = rabbit_energy

        for t in range(self.max_iter):
            E0 = 2 * np.random.random() - 1
            E = 2 * E0 * (1 - (t / self.max_iter) ** 2)

            stagnation_limit = int(2 + 3 * (t / self.max_iter))

            for i in range(self.n_hawks):
                if abs(E) >= 1:
                    if np.random.random() < 0.5:
                        r_idx = np.random.randint(self.n_hawks)
                        hawks[i] = hawks[r_idx] - np.random.random() * abs(
                            hawks[r_idx] - 2 * np.random.random() * hawks[i])
                    else:
                        hawks[i] = (rabbit_pos - hawks.mean(0)) * np.random.random() + self.lb + \
                                   np.random.random() * (self.ub - self.lb)
                else:
                    r = np.random.random()
                    J = 2 * (1 - np.random.random())
                    if r >= 0.5 and abs(E) >= 0.5:
                        hawks[i] = rabbit_pos - E * abs(rabbit_pos - hawks[i])
                    elif r >= 0.5 and abs(E) < 0.5:
                        hawks[i] = rabbit_pos - E * abs(2 * rabbit_pos - hawks[i])
                    elif r < 0.5 and abs(E) >= 0.5:
                        hawks[i] = rabbit_pos - E * abs(J * rabbit_pos - hawks[i])
                    else:
                        hawks[i] = rabbit_pos - E * abs(J * rabbit_pos - hawks[i])

                hawks[i] = np.clip(hawks[i], self.lb, self.ub)

                curr_fitness, curr_params = self.fitness_function(hawks[i])
                if curr_fitness < fitness[i]:
                    fitness[i] = curr_fitness
                    params_list[i] = curr_params
                if curr_fitness < rabbit_energy:
                    rabbit_energy = curr_fitness
                    rabbit_pos = hawks[i].copy()
                    best_params = curr_params

                if curr_fitness < global_best_energy:
                    global_best_energy = curr_fitness
                    global_best_pos = hawks[i].copy()
                    global_best_params = curr_params

            improvement = abs(last_rabbit_energy - rabbit_energy)
            if improvement < 1e-6:
                self.stagnation_count += 1
                if self.stagnation_count >= stagnation_limit:
                    perturbation_scale = 0.1 * (1 - t / self.max_iter)
                    rabbit_pos += np.random.uniform(-perturbation_scale, perturbation_scale, self.dim) * (
                        self.ub - self.lb)
                    rabbit_pos = np.clip(rabbit_pos, self.lb, self.ub)
                    rabbit_energy, best_params = self.fitness_function(rabbit_pos)

                    if rabbit_energy < global_best_energy:
                        global_best_energy = rabbit_energy
                        global_best_pos = rabbit_pos.copy()
                        global_best_params = best_params
                    self.stagnation_count = 0
            else:
                self.stagnation_count = 0

            last_rabbit_energy = rabbit_energy

            if verbose:
                print(f"\n迭代次数{t + 1}/{self.max_iter}")
                print(f"当前准确率: {-rabbit_energy:.4f}")
                print(f"全局最佳准确率: {-global_best_energy:.4f}")
                print(f"最优参数: {best_params}")

        return global_best_pos, -global_best_energy, global_best_params
