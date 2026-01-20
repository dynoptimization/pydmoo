import numpy as np
from pymoo.core.population import Population

from pydmoo.algorithms.modern.nsga2_imkt import NSGA2IMKT
from pydmoo.core.bounds import clip_and_randomize
from pydmoo.core.inverse import closed_form_solution
from pydmoo.core.lstm.lstm import LSTMpredictor
from pydmoo.core.sample_gaussian import univariate_gaussian_sample


class NSGA2IMLSTM(NSGA2IMKT):
    """Inverse Modeling with LSTM (IMLSTM).

    Inverse Modeling for Dynamic Multiobjective Optimization with LSTM prediction In objective Space.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.size_pool = 10
        self.denominator = 0.5

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


def prepare_data_means_std(n_timesteps, means_stds):
    """Prepare time series data from means and standard deviations.

    This function converts a sequence of mean vectors and standard deviation vectors
    into a time series format suitable for machine learning models. It concatenates
    mean values and standard deviation values to create feature vectors for each time step.

    Parameters
    ----------
    means_stds : list of tuples
        List containing (mean, std, n_iter) pairs for each time step, where:
        - mean: 1D numpy array of mean values
        - std: 1D numpy array of standard deviation values
        - n_iter: number of iterations (not used in feature extraction)

    Returns
    -------
    time_series_data : list
        Combined feature data with shape (n_timesteps, n_features)
        Each row represents a time step containing:
        [mean_1, mean_2, ..., mean_n, std_1, std_2, ..., std_n]
    """
    # Create time series data
    time_series_data = []  # shape: (n_timesteps, n_features)

    # Process only the most recent n_timesteps
    for m, s, _ in means_stds[-n_timesteps:]:
        # Combine mean vector and standard deviation vector
        # [*m] unpacks all mean values
        # [*s] unpacks all standard deviation values
        feature_vector = [*m, *s]

        time_series_data.append(feature_vector)

    return time_series_data


class NSGA2IMiLSTM(NSGA2IMLSTM):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.size_pool = 10
        self.denominator = 0.5
        self._n_timesteps = 10
        self._sequence_length = 5
        self._incremental_learning = True


class NSGA2IMiLSTM1003(NSGA2IMLSTM):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.size_pool = 10
        self.denominator = 0.5
        self._n_timesteps = 10
        self._sequence_length = 3
        self._incremental_learning = True


class NSGA2IMiLSTM1005(NSGA2IMLSTM):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.size_pool = 10
        self.denominator = 0.5
        self._n_timesteps = 10
        self._sequence_length = 5
        self._incremental_learning = True


class NSGA2IMiLSTM1007(NSGA2IMLSTM):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.size_pool = 10
        self.denominator = 0.5
        self._n_timesteps = 10
        self._sequence_length = 7
        self._incremental_learning = True


class NSGA2IMiLSTM1009(NSGA2IMLSTM):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.size_pool = 10
        self.denominator = 0.5
        self._n_timesteps = 10
        self._sequence_length = 9
        self._incremental_learning = True


class NSGA2IMiLSTM1503(NSGA2IMLSTM):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.size_pool = 15
        self.denominator = 0.5
        self._n_timesteps = 15
        self._sequence_length = 3
        self._incremental_learning = True


class NSGA2IMiLSTM1505(NSGA2IMLSTM):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.size_pool = 15
        self.denominator = 0.5
        self._n_timesteps = 15
        self._sequence_length = 5
        self._incremental_learning = True


class NSGA2IMiLSTM1507(NSGA2IMLSTM):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.size_pool = 15
        self.denominator = 0.5
        self._n_timesteps = 15
        self._sequence_length = 7
        self._incremental_learning = True


class NSGA2IMiLSTM1509(NSGA2IMLSTM):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.size_pool = 15
        self.denominator = 0.5
        self._n_timesteps = 15
        self._sequence_length = 9
        self._incremental_learning = True


class NSGA2IMiLSTM1511(NSGA2IMLSTM):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.size_pool = 15
        self.denominator = 0.5
        self._n_timesteps = 15
        self._sequence_length = 11
        self._incremental_learning = True


class NSGA2IMiLSTM1513(NSGA2IMLSTM):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.size_pool = 15
        self.denominator = 0.5
        self._n_timesteps = 15
        self._sequence_length = 13
        self._incremental_learning = True
