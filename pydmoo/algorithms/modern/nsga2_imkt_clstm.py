import numpy as np
from pymoo.core.population import Population
from pymoo.operators.survival.rank_and_crowding import RankAndCrowding

from pydmoo.algorithms.modern.nsga2_imkt import NSGA2IMKT
from pydmoo.algorithms.modern.nsga2_imkt_lstm import prepare_data_means_std
from pydmoo.core.bounds import clip_and_randomize
from pydmoo.core.inverse import closed_form_solution
from pydmoo.core.lstm.lstm import LSTMpredictor
from pydmoo.core.sample_gaussian import univariate_gaussian_sample


class NSGA2IMcLSTM(NSGA2IMKT):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._n_timesteps = 10
        self._sequence_length = 5  # Use 5 historical time steps to predict next step
        self._incremental_learning = False

    def _setup(self, problem, **kwargs):
        super()._setup(problem, **kwargs)

        # Must be here
        self._lstm = LSTMpredictor(
            self._sequence_length,
            hidden_dim=64,
            num_layers=1,
            epochs=50,
            batch_size=32,
            lr=0.001,
            device="cpu",  # for fair comparison
            patience=5,
            seed=self.seed,
            model_type="lstm",
            incremental_learning=self._incremental_learning,
        )

    def _response_mechanism(self):
        """Response mechanism."""
        pop = self.pop
        X = pop.get("X")

        # recreate the current population without being evaluated
        pop = Population.new(X=X)

        # sample self.pop_size individuals in decision space
        samples_old = self.sampling_new_pop()

        # select self.pop_size/2 individuals with better convergence and diversity
        samples = samples_old[:int(len(samples_old)/2)]

        # knowledge in objective space
        means_stds, mean, std = self._in_decision_or_objective_space_1d(samples, "objective_space")

        # Check if sufficient historical data is available for LSTM prediction
        if len(means_stds) > self._n_timesteps:
            # Update pool
            self.data["means_stds"] = means_stds[self._n_timesteps:]

            # Prepare time series data from historical means and standard deviations
            time_series_data = prepare_data_means_std(self._n_timesteps, means_stds)

            # Initialize predictor and generate prediction for next time step
            next_prediction = self._lstm.convert_train_predict(time_series_data)

            # Convert prediction tensor to numpy array for further processing
            next_prediction = next_prediction.numpy()

            # Split prediction into mean and standard deviation components
            # First n_obj elements represent mean values, remaining elements represent standard deviations
            mean_new, std_new = next_prediction[:self.problem.n_obj], next_prediction[self.problem.n_obj:]
            std_new = np.abs(std_new)

        else:
            mean_new, std_new = self._select_means_stds(means_stds, mean, std)

        # sample self.pop_size individuals in objective space
        F = univariate_gaussian_sample(mean_new, std_new, self.pop_size, random_state=self.random_state)

        # TODO
        # inverse mapping
        # X = FB
        B = closed_form_solution(samples.get("X"), samples.get("F"))

        # X = FB
        X = np.dot(F, B)

        # bounds
        if self.problem.has_bounds():
            xl, xu = self.problem.bounds()
            X = clip_and_randomize(X, xl, xu, random_state=self.random_state)

        # merge
        pop = Population.merge(samples_old, Population.new(X=X))

        return pop

    def sampling_new_pop(self):
        ps = self.opt.get("X")
        X = self.pop.get("X")

        if not self.problem.has_constraints():

            last_ps = self.data.get("last_ps", [])
            if len(last_ps) == 0:
                last_ps = ps
            self.data["last_ps"] = ps

            d = np.mean(ps, axis=0) - np.mean(last_ps, axis=0)

            radius = max(np.linalg.norm(d) / self.problem.n_obj, 0.1)

            X = X + d + self.random_state.uniform(low=-radius, high=radius, size=X.shape)

        # bounds
        if self.problem.has_bounds():
            xl, xu = self.problem.bounds()
            X = clip_and_randomize(X, xl, xu, random_state=self.random_state)

        samples = Population.new(X=X)
        samples = self.evaluator.eval(self.problem, samples)

        # do a survival to recreate rank and crowding of all individuals
        samples = RankAndCrowding().do(self.problem, samples, n_survive=len(samples))
        return samples

    def _select_means_stds(self, means_stds, mean_new, std_new):
        # Unpack means and stds
        means = np.array([m[0] for m in means_stds])
        stds = np.array([m[1] for m in means_stds])

        # Weighted combination
        mean_new = 0.5 * mean_new + 0.5 * means[-1]
        std_new = 0.5 * std_new + 0.5 * stds[-1]
        return mean_new, std_new


class NSGA2IMicLSTM(NSGA2IMcLSTM):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._n_timesteps = 10
        self._sequence_length = 5  # Use 5 historical time steps to predict next step
        self._incremental_learning = True


class NSGA2IMicLSTM1003(NSGA2IMcLSTM):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._n_timesteps = 10
        self._sequence_length = 3
        self._incremental_learning = True


class NSGA2IMicLSTM1005(NSGA2IMcLSTM):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._n_timesteps = 10
        self._sequence_length = 5
        self._incremental_learning = True


class NSGA2IMicLSTM1007(NSGA2IMcLSTM):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._n_timesteps = 10
        self._sequence_length = 7
        self._incremental_learning = True


class NSGA2IMicLSTM1009(NSGA2IMcLSTM):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._n_timesteps = 10
        self._sequence_length = 9
        self._incremental_learning = True


class NSGA2IMicLSTM1503(NSGA2IMcLSTM):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._n_timesteps = 15
        self._sequence_length = 3
        self._incremental_learning = True


class NSGA2IMicLSTM1505(NSGA2IMcLSTM):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._n_timesteps = 15
        self._sequence_length = 5
        self._incremental_learning = True


class NSGA2IMicLSTM1507(NSGA2IMcLSTM):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._n_timesteps = 15
        self._sequence_length = 7
        self._incremental_learning = True


class NSGA2IMicLSTM1509(NSGA2IMcLSTM):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._n_timesteps = 15
        self._sequence_length = 9
        self._incremental_learning = True


class NSGA2IMicLSTM1511(NSGA2IMcLSTM):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._n_timesteps = 15
        self._sequence_length = 11
        self._incremental_learning = True


class NSGA2IMicLSTM1513(NSGA2IMcLSTM):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._n_timesteps = 15
        self._sequence_length = 13
        self._incremental_learning = True
