"""JMP-like neural fitting module."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from functools import partial
from typing import Annotated, Any, Literal, TypeAlias

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
from jaxtyping import Array
from typing_extensions import TypedDict

__all__ = [
    "ActivationFunction",
    "History",
    "Hyperparameters",
    "JohnsonParams",
    "Metadata",
    "ModelComponents",
    "NeuralFitResult",
    "NeuralFitResult",
    "PRNGKey",
    "Params",
    "PenaltyMethod",
    "PerformanceMetrics",
    "TourResult",
    "TrainingInfo",
    "TransformMethod",
    "TransformParams",
    "neural_fit",
    "predict_neural_model",
]


def neural_fit(  # noqa: C901
    data: pd.DataFrame,
    target_column: str,
    feature_columns: list[str] | None = None,
    hidden_dims: tuple[int, ...] = (10,),
    activation: ActivationFunction = "tanh",
    transform_method: TransformMethod = "johnson_su",
    penalty_method: PenaltyMethod = "squared",
    penalty_lambda: float = 0.01,
    num_tours: int = 5,
    learning_rate: float = 0.01,
    num_epochs: int = 1000,
    validation_split: float = 0.2,
    random_seed: int = 42,
    *,
    transform_covariates: bool = True,
    robust_fit: bool = False,
    progress_bar: bool = True,
) -> NeuralFitResult:
    """Neural fitting function with JMP-like capabilities.

    Args:
        data: Input data with features and target.
        target_column: Name of the target column.
        feature_columns: List of feature column names. If None, uses all numeric
            columns except target.
        hidden_dims: Hidden layer dimensions, e.g., (10, 5) for two hidden layers.
        activation: Activation function to use.
        transform_method: Transformation method to use.
        robust_fit: Whether to use least absolute deviations instead of least squares.
        penalty_method: Penalty method to apply.
        penalty_lambda: Penalty parameter λ.
        num_tours: Number of restart tours with different random starting points.
        learning_rate: Learning rate for optimization.
        num_epochs: Number of training epochs per tour.
        validation_split: Fraction of data to use for validation.
        random_seed: Random seed for reproducibility.
        transform_covariates: Whether to transform continuous variables to normality.
        progress_bar: Whether to show a progress bar during training.

    Returns:
        Dictionary containing trained model, parameters, and training history.

    Raises:
        RuntimeError: If all training tours fail.
    """
    if feature_columns is None:
        feature_columns = [
            col
            for col in data.select_dtypes(include=[np.number]).columns
            if col != target_column
        ]

    def transform_fn(data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        if transform_covariates:
            return _transform_covariates(data, target_column, transform_method)
        return data

    df_work = transform_fn(data)

    # Prepare arrays
    X = jnp.array(df_work[feature_columns].values, dtype=jnp.float32)
    y = jnp.array(df_work[target_column].values, dtype=jnp.float32)

    # Standardize features
    X_mean = jnp.mean(X, axis=0)
    X_std = jnp.std(X, axis=0) + 1e-8  # Add small epsilon to avoid division by zero
    X_normalized = (X - X_mean) / X_std

    # Network architecture
    input_dim = X_normalized.shape[1]
    output_dim = 1 if len(y.shape) == 1 else y.shape[1]

    init_params_fn, forward_fn = _create_network(
        input_dim, hidden_dims, output_dim, activation
    )

    # Multiple tours (restarts)
    best_params = None
    best_val_loss = float("inf")
    best_history: History = {"train_loss": [], "val_loss": []}
    tour_results = []

    key = jax.random.PRNGKey(random_seed)

    for tour in range(num_tours):
        key, tour_key = jax.random.split(key)

        try:
            params, history = _train_network(
                X_normalized,
                y,
                init_params_fn,
                forward_fn,
                tour_key,
                learning_rate=learning_rate,
                num_epochs=num_epochs,
                robust_fit=robust_fit,
                penalty_method=penalty_method,
                penalty_lambda=penalty_lambda,
                validation_split=validation_split,
                progress_bar=progress_bar,
            )

            final_val_loss = min(history["val_loss"])
            tour_results.append(
                TourResult(tour=tour, val_loss=final_val_loss, history=history)
            )

            if final_val_loss < best_val_loss:
                best_val_loss = final_val_loss
                best_params = params
                best_history = history

        except Exception as e:  # noqa: BLE001
            warnings.warn(f"Tour {tour} failed: {e!s}", stacklevel=2)
            continue

    if best_params is None:
        msg = "All tours failed. Try adjusting hyperparameters."
        raise RuntimeError(msg)

    # Create prediction function
    def predict_fn(X_new: Array) -> Array:
        """Make predictions on new data.

        Args:
            X_new: New input features.

        Returns:
            Predictions.
        """
        X_new_norm = (X_new - X_mean) / X_std
        return forward_fn(best_params, X_new_norm).squeeze()

    # Model performance on full dataset
    y_pred = predict_fn(X)

    if robust_fit:
        final_loss = jnp.mean(jnp.abs(y - y_pred))
    else:
        final_loss = jnp.mean((y - y_pred) ** 2)

    r_squared = 1 - jnp.var(y - y_pred) / jnp.var(y)

    # Build performance metrics
    performance = PerformanceMetrics(
        mse=float(jnp.mean((y - y_pred) ** 2)) if not robust_fit else 0.0,
        mae=float(jnp.mean(jnp.abs(y - y_pred))) if robust_fit else 0.0,
        r_squared=float(r_squared),
        best_validation_loss=float(best_val_loss),
    )

    # Ensure the correct metric is set
    if robust_fit:
        performance["mae"] = float(final_loss)
    else:
        performance["mse"] = float(final_loss)

    return NeuralFitResult(
        model=ModelComponents(
            params=best_params,
            forward_fn=forward_fn,
            predict_fn=predict_fn,
            transform_fn=transform_fn,
            X_mean=X_mean,
            X_std=X_std,
        ),
        performance=performance,
        training=TrainingInfo(
            best_history=best_history,
            tour_results=tour_results,
            num_successful_tours=len(tour_results),
        ),
        metadata=Metadata(
            feature_columns=feature_columns,
            target_column=target_column,
            transform_params=df_work.attrs,  # type: ignore[reportArgumentType]
            hyperparameters=Hyperparameters(
                hidden_dims=hidden_dims,
                activation=activation,
                robust_fit=robust_fit,
                penalty_method=penalty_method,
                penalty_lambda=penalty_lambda,
                num_tours=num_tours,
                learning_rate=learning_rate,
                num_epochs=num_epochs,
            ),
        ),
    )


def predict_neural_model(model_result: NeuralFitResult, df: pd.DataFrame) -> Array:
    """Make predictions using a fitted neural model.

    Args:
        model_result: Result from neural_fit function.
        df: New data for prediction.

    Returns:
        Predictions array.
    """
    df_work = model_result["model"]["transform_fn"](df)
    feature_columns = model_result["metadata"]["feature_columns"]
    X_array = jnp.array(df_work[feature_columns].values, dtype=jnp.float32)

    return model_result["model"]["predict_fn"](X_array)


PRNGKey: TypeAlias = Annotated[Any, "PRNGKey"]
"""The jax.PRNGKey used to generate random weights and biases."""

PenaltyMethod: TypeAlias = Literal["squared", "absolute", "weight_decay", "no_penalty"]
"""The type of weight penalty to apply during training to avoid overfitting."""

TransformMethod: TypeAlias = Literal["johnson_su", "johnson_sb"]
"""The method used to transform input parameters to approximate normality."""

ActivationFunction: TypeAlias = Literal["tanh", "relu", "sigmoid"]
"""The activation function used in the neural network layers."""


class History(TypedDict):
    """Losses for the neural network training.

    Attributes:
        train_loss: List of training losses for each epoch.
        val_loss: List of validation losses for each epoch.
    """

    train_loss: list[float]
    val_loss: list[float]


class Params(TypedDict):
    """Parameters for a single layer in the neural network.

    Attributes:
        w: Weights for the layer.
        b: Biases for the layer.
    """

    w: Array
    b: Array


class JohnsonParams(TypedDict):
    """Parameters for Johnson distribution transformations.

    Attributes:
        gamma: Johnson distribution gamma parameter.
        delta: Johnson distribution delta parameter.
        xi: Johnson distribution xi parameter.
        lambda_: Johnson distribution lambda parameter.
    """

    gamma: float
    delta: float
    xi: float
    lambda_: float


class TransformParams(TypedDict):
    """Parameters for covariate transformations.

    Attributes:
        method: Transformation method used
        params: Parameters for the transformation.
    """

    method: TransformMethod
    params: JohnsonParams


class ModelComponents(TypedDict):
    """Components of the trained neural model.

    Attributes:
        params: List of parameters for each layer in the network.
        forward_fn: Forward pass function for the neural network.
        predict_fn: Prediction function that applies the forward pass to new data.
        X_mean: Mean of the input features used for normalization.
        X_std: Standard deviation of the input features used for normalization.
    """

    params: list
    forward_fn: Callable
    predict_fn: Callable
    transform_fn: Callable
    X_mean: Array
    X_std: Array


class PerformanceMetrics(TypedDict):
    """Performance metrics for the fitted model.

    Attributes:
        mse: Mean squared error of predictions.
        mae: Mean absolute error of predictions.
        r_squared: R-squared value of the model.
        best_validation_loss: Best validation loss achieved during training.
    """

    mse: float
    mae: float
    r_squared: float
    best_validation_loss: float


class TourResult(TypedDict):
    """Results from a single training tour.

    Attributes:
        tour: Tour number (0-indexed).
        val_loss: Validation loss at the end of the tour.
        history: Training history for this tour, including train and validation losses.
    """

    tour: int
    val_loss: float
    history: History


class TrainingInfo(TypedDict):
    """Training information and history.

    Attributes:
        best_history: Best training history across all tours.
        tour_results: List of results from each training tour.
        num_successful_tours: Number of tours that completed successfully.
    """

    best_history: History
    tour_results: list[TourResult]
    num_successful_tours: int


class Hyperparameters(TypedDict):
    """Hyperparameters used for training.

    Attributes:
        hidden_dims: Dimensions of hidden layers in the neural network.
        activation: Activation function used in the network.
        robust_fit: Whether to use robust (L1) loss instead of least squares.
        penalty_method: Type of penalty applied to weights during training.
        penalty_lambda: Strength of the penalty applied.
        num_tours: Number of tours (restarts) for training.
        learning_rate: Learning rate for the optimizer.
        num_epochs: Number of epochs per tour for training.
    """

    hidden_dims: tuple[int, ...]
    activation: str
    robust_fit: bool
    penalty_method: PenaltyMethod
    penalty_lambda: float
    num_tours: int
    learning_rate: float
    num_epochs: int


class Metadata(TypedDict):
    """Metadata about the fitted model.

    Attributes:
        feature_columns: List of feature column names used in the model.
        target_column: Name of the target column.
        transform_params: Parameters used for transforming covariates, if any.
        hyperparameters: Hyperparameters used for training the model.
    """

    feature_columns: list[str]
    target_column: str
    transform_params: dict[str, TransformParams] | None
    hyperparameters: Hyperparameters


class NeuralFitResult(TypedDict):
    """Complete result from neural_fit function.

    Attributes:
        model: Components of the trained neural model.
        performance: Performance metrics of the fitted model.
        training: Training information and history.
        metadata: Metadata about the fitted model, including hyperparameters and
            feature/target information.
    """

    model: ModelComponents
    performance: PerformanceMetrics
    training: TrainingInfo
    metadata: Metadata


def _johnson_su_transform(
    x: Array, gamma: float, delta: float, xi: float, lambda_: float
) -> Array:
    """Apply Johnson Su transformation to make data approximately normal.

    Args:
        x: Input array to transform.
        gamma: Johnson Su gamma parameter.
        delta: Johnson Su delta parameter.
        xi: Johnson Su xi parameter.
        lambda_: Johnson Su lambda parameter.

    Returns:
        Transformed array with approximate normal distribution.
    """
    z = (x - xi) / lambda_
    return gamma + delta * jnp.arcsinh(z)


def _johnson_sb_transform(
    x: Array, gamma: float, delta: float, xi: float, lambda_: float
) -> Array:
    """Apply Johnson Sb transformation to make data approximately normal.

    Args:
        x: Input array to transform.
        gamma: Johnson Sb gamma parameter.
        delta: Johnson Sb delta parameter.
        xi: Johnson Sb xi parameter.
        lambda_: Johnson Sb lambda parameter.

    Returns:
        Transformed array with approximate normal distribution.
    """
    z = (x - xi) / lambda_
    # Clip z to avoid numerical issues
    z = jnp.clip(z, 1e-8, 1 - 1e-8)
    return gamma + delta * jnp.log(z / (1 - z))


def _fit_johnson_parameters(
    x: Array, method: TransformMethod = "johnson_su"
) -> JohnsonParams:
    """Fit Johnson distribution parameters using method of moments.

    Args:
        x: Input data array.
        method: Johnson distribution type, either "johnson_su" or "johnson_sb".

    Returns:
        Dictionary containing fitted Johnson distribution parameters.
    """
    x_mean = jnp.mean(x)
    x_var = jnp.var(x)
    x_skew = jnp.mean(((x - x_mean) / jnp.sqrt(x_var)) ** 3)
    x_kurt = jnp.mean(((x - x_mean) / jnp.sqrt(x_var)) ** 4)

    if method == "johnson_su":
        # Simplified parameter estimation for Johnson Su
        delta = 1.0 / jnp.sqrt(jnp.log(((x_kurt - 1) / 2) + 1))
        gamma = -x_skew * delta / 2
        lambda_ = jnp.sqrt(x_var / (jnp.exp(1 / delta**2) - 1))
        xi = x_mean - lambda_ * jnp.sinh(gamma / delta)
    elif method == "johnson_sb":
        # Simplified parameter estimation for Johnson Sb
        delta = 1.0 / jnp.sqrt(jnp.log(((x_kurt - 1) / 2) + 1))
        gamma = -x_skew * delta / 2
        # For Sb, we need bounded support
        x_min, x_max = jnp.min(x), jnp.max(x)
        lambda_ = x_max - x_min
        xi = x_min
    else:
        msg = f"Unknown Johnson method: {method}"
        raise ValueError(msg)

    return JohnsonParams(
        gamma=float(gamma),
        delta=float(delta),
        xi=float(xi),
        lambda_=float(lambda_),
    )


def _transform_covariates(
    df: pd.DataFrame,
    target_col: str,
    transform_method: TransformMethod = "johnson_su",
) -> pd.DataFrame:
    """Transform continuous covariates to approximate normality.

    Args:
        df: Input dataframe.
        target_col: Name of target column to exclude from transformation.
        transform_method: Transformation method to use.

    Returns:
        transformed dataframe with transformation parameters in its attrs.
    """
    df_transformed = df.copy()
    transform_params: dict[str, TransformParams] = {}

    continuous_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in continuous_cols:
        continuous_cols.remove(target_col)

    for col in continuous_cols:
        x = jnp.array(df[col].values)

        if transform_method == "johnson_su":
            params = _fit_johnson_parameters(x, "johnson_su")
            transformed = _johnson_su_transform(x, **params)
        elif transform_method == "johnson_sb":
            params = _fit_johnson_parameters(x, "johnson_sb")
            transformed = _johnson_sb_transform(x, **params)
        else:
            msg = f"Unknown transform method: {transform_method}"
            raise ValueError(msg)

        df_transformed[col] = np.array(transformed)
        transform_params[col] = TransformParams(method=transform_method, params=params)

    df_transformed.attrs.update(transform_params)  # type: ignore[reportArgumentType]
    return df_transformed


def _create_network(
    input_dim: int,
    hidden_dims: tuple[int, ...],
    output_dim: int,
    activation: Literal["tanh", "relu", "sigmoid"] = "tanh",
) -> tuple[Callable, Callable]:
    """Create a neural network function.

    Args:
        input_dim: Number of input features.
        hidden_dims: Tuple of hidden layer dimensions.
        output_dim: Number of output dimensions.
        activation: Activation function to use.

    Returns:
        Tuple of (parameter initialization function, forward function).
    """

    def init_params_fn(key: PRNGKey) -> list[Params]:
        """Initialize network parameters.

        Args:
            key: JAX random key.

        Returns:
            List of parameter dictionaries for each layer.
        """
        keys = jax.random.split(key, len(hidden_dims) + 1)
        params = []

        prev_dim = input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            w = jax.random.normal(keys[i], (prev_dim, hidden_dim)) * jnp.sqrt(
                2.0 / prev_dim
            )
            b = jnp.zeros(hidden_dim)
            params.append({"w": w, "b": b})
            prev_dim = hidden_dim

        # Output layer
        w_out = jax.random.normal(keys[-1], (prev_dim, output_dim)) * jnp.sqrt(
            2.0 / prev_dim
        )
        b_out = jnp.zeros(output_dim)
        params.append({"w": w_out, "b": b_out})

        return params

    def forward(params: list[Params], x: Array) -> Array:
        """Forward pass through the network.

        Args:
            params: Network parameters.
            x: Input array.

        Returns:
            Network output.
        """
        activations = {"tanh": jnp.tanh, "relu": jax.nn.relu, "sigmoid": jax.nn.sigmoid}
        act_fn = activations.get(activation, jnp.tanh)

        for layer_params in params[:-1]:
            x = jnp.dot(x, layer_params["w"]) + layer_params["b"]
            x = act_fn(x)

        # Output layer (no activation)
        x = jnp.dot(x, params[-1]["w"]) + params[-1]["b"]
        return x

    return init_params_fn, forward


def _compute_loss(
    params: list[Params],
    forward_fn: Callable,
    X: Array,
    y: Array,
    penalty_method: PenaltyMethod = "squared",
    penalty_lambda: float = 0.01,
    *,
    robust_fit: bool = False,
) -> float:
    """Compute loss with optional robustness and penalty.

    Args:
        params: Network parameters.
        forward_fn: Forward function.
        X: Input features.
        y: Target values.
        penalty_method: Type of penalty to apply.
        penalty_lambda: Penalty strength parameter.
        robust_fit: Whether to use robust (L1) loss.

    Returns:
        Computed loss value.
    """
    y_pred = forward_fn(params, X).squeeze()

    # Base loss
    if robust_fit:
        # Least absolute deviations
        base_loss = jnp.mean(jnp.abs(y - y_pred))
    else:
        # Least squares
        base_loss = jnp.mean((y - y_pred) ** 2)

    # Penalty term
    penalty = 0.0
    if penalty_method != "no_penalty":
        for layer_params in params:
            w = layer_params["w"]
            if penalty_method == "squared":
                penalty += jnp.sum(w**2)
            elif penalty_method == "absolute":
                penalty += jnp.sum(jnp.abs(w))
            elif penalty_method == "weight_decay":
                penalty += jnp.sum(w**2)  # L2 regularization

    return base_loss + penalty_lambda * penalty  # type: ignore[reportReturnType]


def _train_network(
    X: Array,
    y: Array,
    init_params_fn: Callable[..., list[Params]],
    forward_fn: Callable,
    key: PRNGKey,
    learning_rate: float = 0.01,
    num_epochs: int = 1000,
    penalty_method: PenaltyMethod = "squared",
    penalty_lambda: float = 0.01,
    validation_split: float = 0.2,
    *,
    robust_fit: bool = False,
    progress_bar: bool = True,
) -> tuple[list, History]:
    """Train the neural network with validation.

    Args:
        X: Input features.
        y: Target values.
        init_params_fn: Parameter initialization function.
        forward_fn: Forward pass function.
        key: JAX random key.
        learning_rate: Learning rate for optimizer.
        num_epochs: Number of training epochs.
        penalty_method: Type of penalty to apply.
        penalty_lambda: Penalty strength.
        validation_split: Fraction of data for validation.
        robust_fit: Whether to use robust loss.
        progress_bar: Whether to show a progress bar during training.

    Returns:
        Tuple of (best parameters, training history).
    """
    # Split data for validation
    n_samples = X.shape[0]
    n_val = int(n_samples * validation_split)

    key, split_key = jax.random.split(key)
    indices = jax.random.permutation(split_key, n_samples)

    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    X_train, y_train = X[train_indices], y[train_indices]
    X_val, y_val = X[val_indices], y[val_indices]

    # Initialize parameters and optimizer
    params = init_params_fn(key)
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)  # type: ignore[reportAgumentType]

    # Loss function with penalty
    loss_fn = jax.jit(
        partial(
            _compute_loss,
            forward_fn=forward_fn,
            X=X_train,
            y=y_train,
            robust_fit=robust_fit,
            penalty_method=penalty_method,
            penalty_lambda=penalty_lambda,
        )
    )

    val_loss_fn = jax.jit(
        partial(
            _compute_loss,
            forward_fn=forward_fn,
            X=X_val,
            y=y_val,
            robust_fit=robust_fit,
            penalty_method="no_penalty",
            penalty_lambda=0.0,
        )
    )

    # Training loop
    train_losses = []
    val_losses = []
    best_params = params
    best_val_loss = float("inf")

    grad_fn = jax.grad(loss_fn)

    _tqdm = _noop if not progress_bar else partial(_get_tqdm(), total=num_epochs)
    for _ in (pb := _tqdm(range(num_epochs))):
        # Compute gradients and update parameters
        grads = grad_fn(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params: list[Params] = optax.apply_updates(params, updates)  # type: ignore[reportArgumentType, reportAssignmentType]

        # Compute losses
        train_loss = loss_fn(params)
        val_loss = val_loss_fn(params)

        train_losses.append(float(train_loss))
        val_losses.append(float(val_loss))

        if progress_bar:
            pb.set_postfix(  # type: ignore[reportAttributeAccessIssue]
                train_loss=f"{train_loss:.4f}",
                val_loss=f"{val_loss:.4f}",
            )

        # Early stopping based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params = params

    history: History = {"train_loss": train_losses, "val_loss": val_losses}
    return best_params, history


def _get_tqdm() -> Callable:
    """Get the tqdm function, handling imports."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        from tqdm.autonotebook import tqdm
    return tqdm


def _noop(x: Any) -> Any:  # noqa: ANN401
    return x
