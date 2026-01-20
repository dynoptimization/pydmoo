import numpy as np
from pymoo.core.population import Population

from pydmoo.algorithms.modern.moeadde_imkt_n import MOEADDEIMKTN
from pydmoo.algorithms.modern.nsga2_imkt_n_lstm import prepare_data_mean_cov
from pydmoo.algorithms.utils.utils import make_semidefinite, reconstruct_covariance_from_triu
from pydmoo.core.bounds import clip_and_randomize
from pydmoo.core.inverse import closed_form_solution
from pydmoo.core.lstm.lstm import LSTMpredictor
from pydmoo.core.sample_gaussian import multivariate_gaussian_sample


class MOEADDEIMNLSTM(MOEADDEIMKTN):
    """Inverse Modeling with LSTM (IMNLSTM).

    Inverse Modeling for Dynamic Multiobjective Optimization with Knowledge Transfer In objective Space.
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
        means_covs, mean, cov = self._in_decision_or_objective_space_nd(samples, "objective_space")

        # Check if sufficient historical data is available for LSTM prediction
        if len(means_covs) > self._n_timesteps:
            # Update pool
            self.data["means_covs"] = means_covs[self._n_timesteps:]

            # Prepare time series data from historical means and covariance matrices
            time_series_data = prepare_data_mean_cov(self._n_timesteps, means_covs)

            # Initialize predictor and generate prediction for next time step
            next_prediction = self._lstm.convert_train_predict(time_series_data)

            # Convert prediction tensor to numpy array for further processing
            next_prediction = next_prediction.numpy()

            # Split prediction into mean and covariance components
            # First n_obj elements represent the mean vector, Remaining elements represent the flattened covariance matrix values
            mean_new, cov_new_ = next_prediction[:self.problem.n_obj], next_prediction[self.problem.n_obj:]
            cov_new = reconstruct_covariance_from_triu(cov_new_, len(mean_new))
            cov_new = make_semidefinite(cov_new)

        else:
            mean_new, cov_new = self._select_means_covs(means_covs, mean, cov)

        # sample self.pop_size individuals in objective space
        F = multivariate_gaussian_sample(mean_new, cov_new, self.pop_size, random_state=self.random_state)

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


class MOEADDEIMNiLSTM(MOEADDEIMNLSTM):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.size_pool = 10
        self.denominator = 0.5
        self._n_timesteps = 10
        self._sequence_length = 5  # Use 5 historical time steps to predict next step
        self._incremental_learning = True
