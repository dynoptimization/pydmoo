import os
import random

import numpy as np
import torch


class TimeSeriesBase:
    """Base class for time series forecasting models.

    This class provides common functionality for time series data processing,
    including data conversion, training data preparation, and prediction methods.

    Attributes
    ----------
    sequence_length : int
        Number of historical time steps used for each prediction.
    device : torch.device
        Computation device (CPU or GPU) for model inference.
    model_type : str
        Type of model architecture ('lstm' or 'transformer').
    """

    def __init__(self, sequence_length: int, device: torch.device, model_type: str = "lstm") -> None:
        """Initialize the time series base class.

        Parameters
        ----------
        sequence_length : int
            Number of historical time steps used for each prediction.
        device : torch.device
            Computation device (CPU or GPU) for model inference.
        model_type : str, optional
            Type of model architecture, by default "lstm".
            Supported values: 'lstm', 'transformer'

        Raises
        ------
        ValueError
            If model_type is not supported.
        """
        if model_type not in ["lstm", "transformer"]:
            raise ValueError(f"Unsupported model_type: {model_type}. Use 'lstm' or 'transformer'.")

        self.sequence_length = sequence_length
        self.device = device
        self.model_type = model_type

    def _set_random_seed(self, seed, deterministic=True):
        os.environ['PYTHONHASHSEED'] = str(seed)

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if self.device == "cpu":
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            torch.set_default_device('cpu')
            torch.set_default_dtype(torch.float32)

        elif torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            if deterministic:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                if hasattr(torch, 'use_deterministic_algorithms'):
                    torch.use_deterministic_algorithms(True)

    def set_seed(self, seed: int) -> None:
        """Set random seed after initialization.

        Parameters
        ----------
        seed : int
            Random seed value.
        """
        self.seed = seed
        self._set_random_seed(seed)

    def convert_to_tensor(self, time_series_data: list[list[float]]) -> torch.Tensor:
        """Convert input time series data to PyTorch tensor.

        Parameters
        ----------
        time_series_data : list[list[float]]
            Input time series as list of feature vectors.
            Shape: (n_timesteps, n_features)

        Returns
        -------
        series_data : torch.Tensor
            Converted tensor of shape (n_timesteps, n_features)

        Raises
        ------
        ValueError
            If input data is invalid, not 2D, or insufficient for sequence length.

        Notes
        -----
        Performs comprehensive validation including:
        - Data type checking
        - Array dimensionality validation
        - Sequence length sufficiency check
        """
        # Input validation
        if not time_series_data or not all(isinstance(x, (int, float)) for row in time_series_data for x in row):
            raise ValueError("Invalid time series data")

        # Convert to numpy array first for efficient processing
        np_array = np.array(time_series_data, dtype=np.float32)

        # Validate array shape and dimensions
        if np_array.ndim != 2:
            raise ValueError(f"Expected 2D array, got {np_array.ndim}D array")

        n_timesteps, _ = np_array.shape  # (n_timesteps, n_features)

        # Validate sequence length requirement
        if self.sequence_length >= n_timesteps:
            raise ValueError(
                f"Sequence length {self.sequence_length} must be less than number of timesteps {n_timesteps}"
            )

        # Convert to PyTorch tensor
        series_data = torch.FloatTensor(np_array)
        return series_data

    def prepare_training_data(self, series_data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare training sequences and targets using sliding window approach.

        Parameters
        ----------
        series_data : torch.Tensor
            Input time series tensor of shape (n_timesteps, n_features)

        Returns
        -------
        sequences_tensor : torch.Tensor
            Training sequences of shape (n_samples, sequence_length, n_features)
        targets_tensor : torch.Tensor
            Target values of shape (n_samples, n_features)

        Notes
        -----
        Uses sliding window method to create input-target pairs.
        Number of training samples = n_timesteps - sequence_length.

        The sliding window approach ensures temporal continuity in training data
        by creating overlapping sequences from the original time series.
        """
        n_timesteps, _ = series_data.shape

        # Calculate number of training samples
        n_samples = n_timesteps - self.sequence_length

        # Create sequence-target pairs using sliding window
        indices = torch.arange(n_samples).unsqueeze(1) + torch.arange(self.sequence_length)
        sequences_tensor = series_data[indices]  # shape: (n_samples, sequence_length, n_features)
        targets_tensor = series_data[self.sequence_length:]  # shape: (n_samples, n_features)

        return sequences_tensor, targets_tensor

    @staticmethod
    def create_training_dataloader(
        sequences_tensor: torch.Tensor, targets_tensor: torch.Tensor, batch_size: int = 32
    ) -> torch.utils.data.DataLoader:
        """Create DataLoader for time series training data.

        Parameters
        ----------
        sequences_tensor : torch.Tensor
            Input sequences of shape (n_samples, sequence_length, n_features)
        targets_tensor : torch.Tensor
            Target values of shape (n_samples, n_features)
        batch_size : int, optional
            Number of samples per batch, by default 32

        Returns
        -------
        dataloader : torch.utils.data.DataLoader
            Configured DataLoader for training

        Notes
        -----
        Key configurations:
        - shuffle=False: Maintains temporal order for time series data
        - pin_memory=False: Optimizes GPU data transfer
        - drop_last=False: Uses all available samples
        - num_workers=0: Avoids multiprocessing issues

        Temporal ordering is critical for time series data to preserve
        the sequential dependencies between data points.
        """
        dataset = torch.utils.data.TensorDataset(sequences_tensor, targets_tensor)

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,  # Critical for time series data
            num_workers=0,
            pin_memory=False,
            drop_last=False,
        )

        return dataloader

    def predict_future(
        self, model, historical_data: torch.Tensor | list[list[float]], n_steps: int = 1
    ) -> torch.Tensor:
        """Generate multiple future predictions using iterative forecasting.

        Parameters
        ----------
        model : torch.nn.Module
            Trained model for time series prediction.
        historical_data : torch.Tensor | list[list[float]]
            Historical time series data of shape (n_timesteps, n_features).
        n_steps : int, optional
            Number of future time steps to predict, by default 1.

        Returns
        -------
        predictions : torch.Tensor
            Predicted values for future time steps of shape (n_steps, n_features).

        Raises
        ------
        ValueError
            If model has not been trained before prediction or if model_type is unsupported.

        Notes
        -----
        Uses recursive prediction strategy with model-specific handling:

        - For LSTM and Transformer models:
            1. Extract the most recent `sequence_length` points as the initial context.
            2. Predict the next time step.
            3. Slide the window: drop the oldest point, append the new prediction.
            4. Repeat for `n_steps`.

        - Critical design choice for LSTM:
            Hidden state is **reset to `None` at every step** to match the training protocol,
            where each batch sample was processed independently with zero-initialized hidden states.
            This ensures consistency between training and inference, improving stability and reproducibility.

        - Output predictions are moved to CPU to avoid GPU memory bloat during long continual loops.
        """
        if not isinstance(historical_data, torch.Tensor):
            historical_data = self.convert_to_tensor(historical_data)

        if model is None:
            raise ValueError("Model must be trained first")

        if len(historical_data) < self.sequence_length:
            raise ValueError(
                f"historical_data has only {len(historical_data)} timesteps, "
                f"but sequence_length = {self.sequence_length} is required."
            )

        # Initialize with historical data (sequence_length, n_features)
        current_sequence = historical_data[-self.sequence_length:].clone().to(self.device)
        predictions = []

        model.eval()

        with torch.no_grad():
            for _ in range(n_steps):
                # Prepare input by adding batch dimension
                input_seq = current_sequence.unsqueeze(0)  # (1, sequence_length, n_features); already on device

                # Generate prediction using trained model with model-specific handling
                if self.model_type == "lstm":
                    # CRITICAL: Use `hidden=None` to match training protocol.
                    # Training used independent sequences with zero-initialized states.
                    # Passing a carried-over hidden would create state-input misalignment.
                    pred, _ = model(input_seq, hidden=None)  # (1, n_features); stateless sliding window
                elif self.model_type == "transformer":
                    # Transformer has no recurrent state — naturally stateless.
                    pred = model(input_seq)  # (1, n_features)
                else:
                    raise ValueError(f"Unsupported model_type: {self.model_type}. Use 'lstm' or 'transformer'.")

                pred = pred.squeeze(0)  # (n_features,), still on device
                predictions.append(pred.cpu())  # only detach & move to CPU for storage, leaves original pred unchanged

                # Update sequence: remove oldest point, append new prediction
                current_sequence = torch.cat([current_sequence[1:], pred.unsqueeze(0)])  # stays on device

        return torch.stack(predictions)  # Stack into (n_steps, n_features)
