import torch
import torch.nn as nn
import torch.optim as optim

from .base import TimeSeriesBase


class PredictorLSTM(nn.Module):
    """Lightweight LSTM network for time-series forecasting.

    Attributes
    ----------
    input_dim : int
        Number of input features per time step.
    hidden_dim : int
        Number of hidden units in LSTM layers.
    num_layers : int
        Number of LSTM layers.
    lstm : nn.LSTM
        Core LSTM network module.
    output_layer : nn.Linear
        Final output projection layer.
    """

    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout):
        """Initialize the LSTM predictor model.

        Parameters
        ----------
        input_dim : int
            Number of input features per time step.
        hidden_dim : int
            Number of hidden units in LSTM layers.
        num_layers : int
            Number of LSTM layers.
        output_dim : int
            Number of output features to predict.
        dropout : float
            Dropout probability between LSTM layers (applied only when num_layers > 1).
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # LSTM architecture with cuDNN optimization
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Output projection layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, hidden=None):
        """Forward pass through the LSTM network.

        Parameters
        ----------
        x : torch.Tensor
            Input sequence tensor of shape (batch_size, sequence_length, input_dim).
        hidden : tuple, optional
            Initial hidden state tuple (h_0, c_0) where:
            - h_0 : torch.Tensor, shape (num_layers, batch_size, hidden_dim)
            - c_0 : torch.Tensor, shape (num_layers, batch_size, hidden_dim)

        Returns
        -------
        pred : torch.Tensor
            Prediction for next time step with shape (batch_size, output_dim).
        hidden_out : tuple
            Final hidden state tuple (h_n, c_n) with same shapes as input hidden.
        """
        # LSTM forward pass
        output, hidden_out = self.lstm(x, hidden)

        # Extract hidden state from the last time step
        last_hidden = output[:, -1, :]  # shape: (batch_size, hidden_dim)

        # Generate prediction for next time step
        pred = self.output_layer(last_hidden)  # shape: (batch_size, output_dim)

        return pred, hidden_out


class PredictorMultiLayerLSTM(nn.Module):
    """Multi-layer LSTM model for time series prediction.

    Uses sequential LSTM layers to encode temporal patterns and predict next time step.

    Attributes
    ----------
    input_dim : int
        Number of input features per time step.
    hidden_dim : int
        Number of hidden units in each LSTM layer.
    lstm_layers : nn.ModuleList
        Stack of LSTM layers.
    output_layer : nn.Sequential
        Output network to generate predictions.
    """

    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        """Initialize the Multi-layer LSTM predictor model.

        Parameters
        ----------
        input_dim : int
            Number of input features per time step.
        hidden_dim : int
            Number of hidden units in each LSTM layer.
        num_layers : int
            Number of LSTM layers to stack.
        output_dim : int
            Number of output features to predict.
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Create stack of LSTM layers
        # First layer: input_dim -> hidden_dim, subsequent layers: hidden_dim -> hidden_dim
        self.lstm_layers = nn.ModuleList(
            [
                nn.LSTM(
                    input_size=input_dim if i == 0 else hidden_dim,
                    hidden_size=hidden_dim,
                    batch_first=True,  # Input shape: (batch_size, sequence_length, input_dim)
                    num_layers=1,  # Each layer is separate for individual hidden state control
                    dropout=0,
                )
                for i in range(num_layers)
            ]
        )

        # Output network to transform LSTM encoding to prediction
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),  # Reduce dimension for computational efficiency
            nn.ReLU(),  # Introduce non-linearity
            nn.Dropout(0.1),  # Prevent overfitting
            nn.Linear(hidden_dim // 2, output_dim),  # Final prediction layer
        )

    def forward(self, x: torch.Tensor, hidden_states=None):
        """Forward pass through the LSTM network.

        Parameters
        ----------
        x : torch.Tensor
            Input sequence tensor of shape (batch_size, sequence_length, input_dim).
        hidden_states : list of tuples, optional
            Previous hidden states for each LSTM layer as (hidden_state, cell_state) tuples.

        Returns
        -------
        prediction : torch.Tensor
            Predicted values for next time step with shape (batch_size, output_dim).
        new_hidden_states : list of tuples
            Updated hidden states for each LSTM layer.
        """
        # Initialize hidden states if not provided
        if hidden_states is None:
            hidden_states = [None] * len(self.lstm_layers)

        # Process input through each LSTM layer sequentially
        output = x  # Start with raw input: shape (batch_size, sequence_length, input_dim)
        new_hidden_states = []  # Store updated hidden states for all layers

        # Pass data through each LSTM layer
        for i, lstm_layer in enumerate(self.lstm_layers):
            # Each LSTM layer processes the sequence
            # output shape: (batch_size, sequence_length, hidden_dim)
            output, (h_n, c_n) = lstm_layer(output, hidden_states[i] if hidden_states[i] is not None else None)
            new_hidden_states.append((h_n, c_n))

        # Extract final time step encoding for prediction
        # last_output shape: (batch_size, hidden_dim)
        last_output = output[:, -1, :]

        # Generate prediction for next time step
        # prediction shape: (batch_size, output_dim)
        prediction = self.output_layer(last_output)

        return prediction, new_hidden_states


class LSTMpredictor(TimeSeriesBase):
    """LSTM-based time series predictor with sliding window approach."""

    def __init__(
        self,
        sequence_length,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
        epochs: int = 50,
        batch_size: int = 32,
        lr: float = 0.001,
        device: str = "cpu",
        patience: int = 5,
        seed: int | None = None,
        model_type: str = "lstm",
        incremental_learning: bool = False,
    ):
        """Initialize the LSTM predictor with training configuration.

        Parameters
        ----------
        sequence_length : int
            Number of historical time steps used for prediction.
        hidden_dim : int
            Number of hidden units in LSTM layers, by default 64.
        num_layers : int
            Number of LSTM layers, by default 1.
        dropout : float
            Dropout probability between layers, by default 0.0.
        epochs : int
            Maximum number of training epochs, by default 50.
        batch_size : int
            Number of samples per training batch, by default 32.
        lr : float
            Learning rate for optimizer, by default 0.001.
        device : str
            Computation device ('cpu' or 'cuda'), by default "cpu".
        patience : int
            Early stopping patience (epochs without improvement), by default 5.
        seed : int, optional
            Random seed for reproducibility, by default None.
        model_type : str
            Type of model architecture, by default "lstm".
            Supported values: 'lstm', 'transformer'
        incremental_learning : bool
            Whether to use incremental learning mode, by default False.
            If True, model will be reused and updated with new data.
        """
        self.sequence_length = sequence_length
        if self.sequence_length < 1:
            raise ValueError("The sequence length should be greater than 0.")

        # Model hyperparameters
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout if num_layers > 1 else 0.0

        # Training parameters
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr

        # Device configuration
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Training parameters
        self.patience = patience
        self._improvement_threshold = 0.001

        # Repeatability
        self.seed = seed
        if self.seed is not None:
            self._set_random_seed(self.seed)

        # Initialize parent class
        super().__init__(self.sequence_length, self.device, model_type=model_type)

        # Incremental learning setup
        self.model = None
        self.incremental_learning = incremental_learning

    def convert_train_predict(self, time_series_data: list[list[float]]) -> torch.Tensor:
        """Complete pipeline: convert data, train model, and generate prediction.

        Parameters
        ----------
        time_series_data : list[list[float]]
            Input time series data as list of feature vectors.
            Shape: (n_timesteps, n_features)
            Example: [[1.0, 2.0], [1.1, 2.1], [1.2, 2.2]] for 3 timesteps, 2 features

        Returns
        -------
        next_prediction : torch.Tensor
            Predicted values for the next time step.
            Shape: (n_features,)
            Example: tensor([1.3, 2.3]) for 2 features

        Notes
        -----
        This method provides an end-to-end workflow:
        1. Convert raw data to tensor format
        2. Train LSTM model on the entire dataset
        3. Generate prediction for the next time step
        The trained model is stored in self.model for future predictions.
        """
        # Convert input data to tensor format
        tensor_data = self.convert_to_tensor(time_series_data)

        # Train model on the prepared data
        trained_model, _ = self.train_model(tensor_data)

        # Generate prediction using trained model
        next_prediction = self.predict_future(trained_model, tensor_data, n_steps=1)

        return next_prediction[0]

    def train_model(self, series_data: torch.Tensor):
        """Train LSTM model on provided time series data.

        Parameters
        ----------
        series_data : torch.Tensor
            Input time series data of shape (n_timesteps, n_features)

        Returns
        -------
        tuple[nn.Module, list[float]]
            Trained model and list of training losses

        Notes
        -----
        Training workflow:
        1. Prepare training sequences and targets using sliding window
        2. Create DataLoader for efficient batch processing
        3. Train model using configured optimization parameters
        """
        # Prepare training sequences and targets
        sequences_tensor, targets_tensor = self.prepare_training_data(series_data)

        # Create DataLoader with appropriate batch size
        dataloader = self.create_training_dataloader(
            sequences_tensor, targets_tensor, batch_size=min(self.batch_size, sequences_tensor.shape[0])
        )

        # Train LSTM model
        trained_model, losses = self._train_model(dataloader)

        return trained_model, losses

    def _construct_model(self, dataloader: torch.utils.data.DataLoader) -> nn.Module:
        """Construct LSTM model using prepared sequential data.

        Parameters
        ----------
        dataloader : torch.utils.data.DataLoader
            DataLoader containing training sequences and corresponding targets.

        Returns
        -------
        model : nn.Module
            Configured LSTM model instance.
        """
        # Extract model dimensions from dataset
        input_dim = dataloader.dataset[0][0].shape[-1]
        output_dim = dataloader.dataset[0][1].shape[-1]

        # Initialize model based on continual learning configuration
        if (self.model is None) or (not self.incremental_learning):
            # Create new model for initial training or non-continual learning scenarios
            model = PredictorLSTM(
                input_dim=input_dim,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                output_dim=output_dim,
                dropout=self.dropout,
            ).to(self.device)
            if self.incremental_learning:
                self.model = model  # Store model reference for future continual learning
        else:
            # Utilize existing model for continual learning
            model = self.model

        return model

    def _train_model(self, dataloader: torch.utils.data.DataLoader) -> tuple[nn.Module, list[float]]:
        """Train LSTM model on prepared sequence data.

        Parameters
        ----------
        dataloader : torch.utils.data.DataLoader
            DataLoader containing training sequences and targets

        Returns
        -------
        model : nn.Module
            Trained LSTM model instance
        losses : list[float]
            List of average losses per epoch

        Notes
        -----
        Training configuration:
        - Loss: Mean Squared Error (MSE)
        - Optimizer: Adam with L2 regularization
        - Learning rate scheduler: StepLR with decay
        - Gradient clipping: Prevents gradient explosion
        - Early stopping: Monitors validation loss improvement

        The training process includes:
        1. Model initialization with appropriate dimensions
        2. Batch-wise forward and backward passes
        3. Learning rate scheduling
        4. Early stopping based on loss improvement
        5. Progress logging at regular intervals
        """
        # Construct model
        model = self._construct_model(dataloader)

        # Training configuration
        criterion, optimizer, scheduler = self._optimizer_scheduler(model)

        # Early stopping initialization
        best_loss = float("inf")  # positive infinity
        patience_counter = 0

        # Training loop
        losses = []
        for epoch in range(self.epochs):
            model.train()
            epoch_loss = 0.0
            batch_count = 0

            for batch_idx, (X_batch, y_batch) in enumerate(dataloader):
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                # Forward pass
                optimizer.zero_grad()
                predictions, _ = model(X_batch)  # stateless; independent-sequence
                loss = criterion(predictions, y_batch)

                # Backward pass with gradient clipping
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                # Accumulate loss
                current_loss = loss.item()
                epoch_loss += current_loss
                batch_count += 1

                # Print batch progress
                if (batch_idx + 1) % 100 == 0:
                    print(f"Batch {batch_idx}, Loss: {current_loss:.4f}")

            # Update learning rate
            scheduler.step()

            # Calculate average epoch loss
            avg_epoch_loss = epoch_loss / batch_count if batch_count > 0 else 0.0
            losses.append(avg_epoch_loss)

            # Early stopping check
            if avg_epoch_loss < best_loss * (1 - self._improvement_threshold):  # relative improvement
                best_loss = avg_epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    # print(f"Early stopping at epoch {epoch + 1}")
                    break

            # Print epoch progress
            if (epoch + 1) % 100 == 0:
                current_lr = optimizer.param_groups[0]["lr"]
                print(f"Epoch [{epoch + 1}/{self.epochs}], Average Loss: {avg_epoch_loss:.6f}, LR: {current_lr:.6f}")

        # Store model for future continual learning
        if self.incremental_learning:
            self.model = model
        return model, losses

    def _optimizer_scheduler(self, model: nn.Module):
        """Configure optimizer and learning rate scheduler.

        Parameters
        ----------
        model : torch.nn.Module
            The model to optimize

        Returns
        -------
        tuple
            criterion: loss function
            optimizer: configured optimizer
            scheduler: learning rate scheduler
        """
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
        return criterion, optimizer, scheduler
