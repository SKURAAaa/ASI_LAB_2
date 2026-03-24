import logging
from typing import Dict, Any, Tuple

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

logger = logging.getLogger(__name__)


def preprocess(data: pd.DataFrame, parameters: Dict[str, Any]) -> pd.DataFrame:
    """
    Czyści dane wejściowe.

    - usuwa wybrane kolumny
    - usuwa outliery
    - wybiera tylko kolumny numeryczne
    - uzupełnia braki wartości

    Args:
        data: Surowe dane z SQLite.
        parameters: Parametry pipeline (nieużywane tutaj).

    Returns:
        Oczyszczony DataFrame.
    """
    logger.info("Start preprocessingu danych")

    data = data.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence'], axis=1, errors='ignore')

    if 'GrLivArea' in data.columns:
        before = len(data)
        data = data[data['GrLivArea'] < 4000]
        logger.info(f"Usunięto {before - len(data)} outlierów")

    data = data.select_dtypes(include=['number'])
    data = data.fillna(0)

    logger.info(f"Dane po preprocessingu: {data.shape}")
    return data


def split_data(
    data: pd.DataFrame,
    parameters: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
           pd.Series, pd.Series, pd.Series]:
    """
    Dzieli dane na train / validation / test.

    Args:
        data: Oczyszczone dane.
        parameters: Parametry zawierające target_column i split.

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    logger.info("Podział danych na zbiory")

    target = parameters["target_column"]
    split_params = parameters["split"]

    X = data.drop(target, axis=1)
    y = data[target]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=split_params["test_size"],
        random_state=split_params["random_state"]
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=split_params["val_ratio"],
        random_state=split_params["random_state"]
    )

    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    return X_train, X_val, X_test, y_train, y_val, y_test


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    parameters: Dict[str, Any]
) -> RandomForestRegressor:
    """
    Trenuje model Random Forest.
    """
    logger.info("Trenowanie modelu")

    model = RandomForestRegressor(
        random_state=parameters["model"]["random_state"]
    )

    model.fit(X_train, y_train)

    logger.info("Model wytrenowany")
    return model


def evaluate_model(
    model: RandomForestRegressor,
    X_val: pd.DataFrame,
    y_val: pd.Series
) -> Dict[str, float]:
    """
    Oblicza metryki modelu (RMSE).

    Args:
        model: Wytrenowany model.
        X_val: Dane walidacyjne (cechy).
        y_val: Dane walidacyjne (target).

    Returns:
        Słownik z metryką RMSE.
    """
    logger.info("Ewaluacja modelu")

    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))

    logger.info(f"RMSE: {rmse:.2f}")

    return {
        "rmse": float(rmse)
    }