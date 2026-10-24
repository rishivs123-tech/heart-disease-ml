"""Hyperparameter tuning module for supervised heart disease classifiers.

Supports GridSearchCV across defined parameter grids in configs/models.yaml.
"""

import logging
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from src.heart_disease.data.preprocess import ALL_FEATURES, create_pipeline
from src.heart_disease.models.build_models import get_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

TUNING_GRIDS = {
    "Logistic Regression": {
        "model__C": [0.01, 0.1, 1.0, 10.0]
    },
    "Random Forest": {
        "model__n_estimators": [100, 200],
        "model__max_depth": [5, 10, None],
        "model__min_samples_split": [2, 5]
    },
    "Support Vector Machine": {
        "model__C": [0.1, 1.0, 5.0],
        "model__gamma": ["scale", "auto"]
    },
    "K-Nearest Neighbors": {
        "model__n_neighbors": [3, 5, 7, 9],
        "model__weights": ["uniform", "distance"]
    },
    "Gradient Boosting": {
        "model__n_estimators": [100, 200],
        "model__learning_rate": [0.05, 0.1],
        "model__max_depth": [2, 3]
    }
}


def tune_model(
    model_name: str,
    strategy: str = "M1",
    train_path: str = "data/processed/train.csv",
    cv_splits: int = 5,
    random_state: int = 42
) -> dict:
    """Execute grid search hyperparameter tuning for a given model and imputation strategy."""
    df_train = pd.read_csv(train_path)
    X_train, y_train = df_train[ALL_FEATURES], df_train["target"]

    base_estimator = get_model(model_name, random_state=random_state)
    pipeline = create_pipeline(strategy, base_estimator)
    param_grid = TUNING_GRIDS.get(model_name, {})

    logger.info("Starting hyperparameter tuning for %s (%s strategy)...", model_name, strategy)
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    grid_search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        refit=True
    )
    grid_search.fit(X_train, y_train)

    logger.info("Best ROC-AUC for %s: %.4f (params: %s)", model_name, grid_search.best_score_, grid_search.best_params_)

    return {
        "model_name": model_name,
        "strategy": strategy,
        "best_score": round(float(grid_search.best_score_), 4),
        "best_params": grid_search.best_params_,
        "best_estimator": grid_search.best_estimator_
    }
