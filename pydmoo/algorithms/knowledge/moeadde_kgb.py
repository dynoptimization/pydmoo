import random
import time

import numpy as np
from pymoo.core.population import Population
from pymoo.operators.survival.rank_and_crowding import RankAndCrowding
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from sklearn.naive_bayes import GaussianNB

from pydmoo.algorithms.base.dmoo.dmoeadde import DMOEADDE


def euclidean_distance(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.sqrt(np.sum((a - b) ** 2))


class MOEADDEKGB(DMOEADDE):
    """Knowledge Guided Bayesian (KGB).

    References
    ----------
    Ye, Y., Li, L., Lin, Q., Wong, K.-C., Li, J., and Ming, Z. (2022).
    Knowledge guided bayesian classification for dynamic multi-objective optimization.
    Knowledge-Based Systems, 250, 109173.
    https://doi.org/10.1016/j.knosys.2022.109173
    """
    def __init__(
        self,
        c_size=13,
        perturb_dev=0.1,
        ps={},
        save_ps=False,
        **kwargs,
    ):

        super().__init__(**kwargs)

        self.C_SIZE = c_size
        self.PERTURB_DEV = perturb_dev

        self.ps = ps
        self.save_ps = save_ps

        self.nr_rand_solutions = 50 * self.pop_size
        self.t = 0

    def _next_static_dynamic(self):
        # for dynamic environment
        pop = self.pop

        if self.state is None:

            # archive the current POS
            self.add_to_ps()

            change_detected = self._detect_change_sample_part_population()

            if change_detected:

                start_time = time.time()

                pop = self._response_mechanism()

                # reevaluate because we know there was a change
                self.evaluator.eval(self.problem, self.pop, skip_already_evaluated=False, algorithm=self)

                # reevaluate because we know there was a change
                self.evaluator.eval(self.problem, pop)

                if len(pop) > self.pop_size:
                    # do a survival to recreate rank and crowding of all individuals
                    # Modified by DynOpt on Dec 21, 2025
                    # n_survive=len(pop) -> n_survive=self.pop_size
                    pop = RankAndCrowding().do(self.problem, pop, n_survive=self.pop_size, random_state=self.random_state)  # len(pop)

                self.data["response_duration"] = time.time() - start_time

        return pop

    def _response_mechanism(self):
        """Response mechanism."""
        # increase t counter for unique key of PS
        self.t += 1

        # conduct knowledge reconstruction examination
        pop_useful, pop_useless, c = self.knowledge_reconstruction_examination()

        # Train a naive bayesian classifier
        model = self.naive_bayesian_classifier(pop_useful, pop_useless)

        # generate a lot of random solutions with the dimensions of problem decision space
        X_test = self.random_strategy(self.nr_rand_solutions)

        # introduce noise to vary previously useful solutions
        noise = self.random_state.normal(0, self.PERTURB_DEV, self.problem.n_var)
        noisy_useful_history = np.asarray(pop_useful) + noise

        # check whether solutions are within bounds
        noisy_useful_history = self.check_boundaries(noisy_useful_history)

        # add noisy useful history to randomly generated solutions
        X_test = np.vstack((X_test, noisy_useful_history))

        # predict whether random solutions are useful or useless
        Y_test = model.predict(X_test)

        # create list of useful predicted solutions
        predicted_pop = self.predicted_population(X_test, Y_test)

        # ------ POPULATION GENERATION --------
        # take a random sample from predicted pop and known useful pop
        if len(predicted_pop) >= self.pop_size - self.C_SIZE:
            init_pop = []
            predicted_pop = random.sample(
                predicted_pop, self.pop_size - self.C_SIZE
            )

            # add sampled solutions to init_pop
            for solution in predicted_pop:
                init_pop.append(solution)

            # add cluster centroids to init_pop
            for solution in c:
                init_pop.append(np.asarray(solution))

        else:

            # if not enough predicted solutions are available, add all predicted solutions to init_pop
            init_pop = []

            for solution in predicted_pop:
                init_pop.append(solution)

            # add cluster centroids to init_pop
            for solution in c:
                init_pop.append(np.asarray(solution))

        # if there are still not enough solutions in init_pop randomly sample previously useful solutions directly without noise to init_pop
        if len(init_pop) < self.pop_size:

            # fill up init_pop with randomly sampled solutions from pop_useful
            if len(pop_useful) >= self.pop_size - len(init_pop):

                init_pop = np.vstack(
                    (
                        init_pop,
                        random.sample(pop_useful, self.pop_size - len(init_pop)),
                    )
                )
            else:
                # if not enough solutions are available, add all previously known useful solutions without noise to init_pop
                for solution in pop_useful:
                    init_pop.append(solution)

        # if there are still not enough solutions in init_pop generate random solutions with the dimensions of problem decision space
        if len(init_pop) < self.pop_size:

            # fill up with random solutions
            init_pop = np.vstack(
                (init_pop, self.random_strategy(self.pop_size - len(init_pop)))
            )

        # recreate the current population without being evaluated
        pop = Population.new(X=init_pop)

        # reevaluate because we know there was a change
        self.evaluator.eval(self.problem, pop)

        # do a survival to recreate rank and crowding of all individuals
        pop = RankAndCrowding().do(self.problem, pop, n_survive=len(pop), random_state=self.random_state)  # self.pop_size

        return pop

    def knowledge_reconstruction_examination(self):
        """
        Perform the knowledge reconstruction examination.
        :return: Tuple containing the useful population, useless population, and cluster centroids
        """
        clusters = self.ps  # set historical PS set as clusters
        Nc = self.C_SIZE  # set final nr of clusters
        size = len(self.ps)  # set size iteration to length of cluster
        run_counter = 0  # counter variable to give unique key

        # while there are still clusters to be condensed
        while size > Nc:

            counter = 0
            min_distance = None
            min_distance_index = []

            # get clusters that are closest to each other by calculating the euclidean distance
            for keys_i in clusters.keys():
                for keys_j in clusters.keys():
                    if (
                        clusters[keys_i]["solutions"]
                        is not clusters[keys_j]["solutions"]
                    ):

                        dst = euclidean_distance(
                            clusters[keys_i]["centroid"],
                            clusters[keys_j]["centroid"],
                        )

                        if min_distance == None:
                            min_distance = dst
                            min_distance_index = [keys_i, keys_j]
                        elif dst < min_distance:
                            min_distance = dst

                            min_distance_index = [keys_i, keys_j]

                        counter += 1

            # merge closest clusters
            for solution in clusters[min_distance_index[1]]["solutions"]:
                clusters[min_distance_index[0]]["solutions"].append(solution)

            # calculate new centroid for merged cluster
            clusters[min_distance_index[0]][
                "centroid"
            ] = self.calculate_cluster_centroid(
                clusters[min_distance_index[0]]["solutions"]
            )

            # remove cluster that was merged
            del clusters[min_distance_index[1]]

            size -= 1
            run_counter += 1

        c = []  # list of centroids
        pop_useful = []
        pop_useless = []

        # get centroids of clusters
        for key in clusters.keys():
            c.append(clusters[key]["centroid"])

        # create pymoo population objected to evaluate centroid solutions
        centroid_pop = Population.new("X", c)

        # evaluate centroids
        self.evaluator.eval(self.problem, centroid_pop)

        # do non-dominated sorting on centroid solutions
        ranking = NonDominatedSorting().do(centroid_pop.get("F"), return_rank=True)[-1]

        # add the individuals from the clusters with the best objective values to the useful population the rest is useless :(

        for idx, rank in enumerate(ranking):
            if rank == 0:
                for key in clusters.keys():
                    if centroid_pop[idx].X == clusters[key]["centroid"]:
                        for cluster_individual in clusters[key]["solutions"]:
                            pop_useful.append(cluster_individual)
            else:
                for key in clusters.keys():
                    if centroid_pop[idx].X == clusters[key]["centroid"]:
                        for cluster_individual in clusters[key]["solutions"]:
                            pop_useless.append(cluster_individual)

        # return useful and useless population and the centroid solutions
        return pop_useful, pop_useless, c

    def naive_bayesian_classifier(self, pop_useful, pop_useless):
        """
        Train a naive Bayesian classifier using the useful and useless populations.
        :param pop_useful: Useful population
        :param pop_useless: Useless population
        :return: Trained GaussianNB classifier
        """
        labeled_useful_solutions = []
        labeled_useless_solutions = []

        # add labels to solutions
        for individual in pop_useful:
            labeled_useful_solutions.append((individual, +1))

        for individual in pop_useless:
            labeled_useless_solutions.append((individual, -1))

        x_train = []
        y_train = []

        for i in range(len(labeled_useful_solutions)):
            x_train.append(labeled_useful_solutions[i][0])
            y_train.append(labeled_useful_solutions[i][1])

        for i in range(len(labeled_useless_solutions)):
            x_train.append(labeled_useless_solutions[i][0])
            y_train.append(labeled_useless_solutions[i][1])

        x_train = np.asarray(x_train)
        y_train = np.asarray(y_train)

        # fit the naive bayesian classifier with the training data
        model = GaussianNB()
        model.fit(x_train, y_train)

        return model

    def add_to_ps(self):
        """
        Add the current Pareto optimal set (POS) to the Pareto set (PS) with individual keys.
        """

        PS_counter = 0

        for individual in self.opt:

            if isinstance(individual.X, list):
                individual.X = np.asarray(individual.X)

            centroid = self.calculate_cluster_centroid(individual.X)

            self.ps[str(PS_counter) + "-" + str(self.t)] = {
                "solutions": [individual.X.tolist()],
                "centroid": centroid,
            }

            PS_counter += 1

    def predicted_population(self, X_test, Y_test):
        """
        Create a predicted population from the test set with positive labels.
        :param X_test: Test set of features
        :param Y_test: Test set of labels
        :return: Predicted population
        """
        predicted_pop = []
        for i in range(len(Y_test)):
            if Y_test[i] == 1:
                predicted_pop.append(X_test[i])
        return predicted_pop

    def calculate_cluster_centroid(self, solution_cluster):
        """
        Calculate the centroid for a given cluster of solutions.
        :param solution_cluster: List of solutions in the cluster
        :return: Cluster centroid
        """
        # Get number of variable shape
        try:
            n_vars = len(solution_cluster[0])
        except TypeError:
            solution_cluster = np.array(solution_cluster)
            return solution_cluster.tolist()

        # TODO: this is lazy garbage fix whats coming in
        cluster = []
        for i in range(len(solution_cluster)):
            # cluster.append(solution_cluster[i].tolist())
            cluster.append(solution_cluster[i])
        solution_cluster = np.asarray(cluster)

        # Get number of solutions
        length = solution_cluster.shape[0]

        centroid_points = []

        # calculate centroid for each variable, by taking mean of every variable of cluster
        for i in range(n_vars):
            # calculate sum over cluster
            centroid_points.append(np.sum(solution_cluster[:, i]))

        return [x / length for x in centroid_points]

    def check_boundaries(self, pop):
        """
        Check and fix the boundaries of the given population.
        :param pop: Population to check and fix boundaries
        :return: Population with corrected boundaries
        """
        # check whether numpy array or pymoo population is given
        if isinstance(pop, Population):
            pop = pop.get("X")

        # check if any solution is outside the bounds
        for individual in pop:
            for i in range(len(individual)):
                if individual[i] > self.problem.xu[i]:
                    individual[i] = self.problem.xu[i]
                elif individual[i] < self.problem.xl[i]:
                    individual[i] = self.problem.xl[i]
        return pop

    def random_strategy(self, N_r):
        """
        Generate a random population within the problem boundaries.
        :param N_r: Number of random solutions to generate
        :return: Randomly generated population
        """
        # generate a random population of size N_r
        # TODO: Check boundaries
        random_pop = self.random_state.random((N_r, self.problem.n_var))

        # check if any solution is outside the bounds
        for individual in random_pop:
            for i in range(len(individual)):
                if individual[i] > self.problem.xu[i]:
                    individual[i] = self.problem.xu[i]
                elif individual[i] < self.problem.xl[i]:
                    individual[i] = self.problem.xl[i]

        return random_pop
