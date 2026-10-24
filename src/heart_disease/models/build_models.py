"""Model initialization and factory functions for the five classifiers.

Default parameters follow standard benchmark settings:
1. Logistic Regression: C=1.0, max_iter=1000, lbfgs solver
2. Random Forest: 200 trees, max_depth=10
3. Support Vector Machine: RBF kernel, C=1.0, probability=True
4. K-Nearest Neighbors: 7 neighbors, uniform weights
5. Gradient Boosting: 200 estimators, learning_rate=0.1, max_depth=2
"""

from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


def get_model(model_name: str, random_state: int = 42, **kwargs) -> BaseEstimator:
    """Instantiate a supervised classifier with reference paper configurations or custom overrides."""
    model_name_lower = model_name.lower().replace(" ", "_").replace("-", "_")

    if model_name_lower in ["lr", "logistic_regression", "logisticregression"]:
        params = {
            "C": 1.0,
            "solver": "lbfgs",
            "max_iter": 1000,
            "random_state": random_state
        }
        params.update(kwargs)
        return LogisticRegression(**params)

    elif model_name_lower in ["rf", "random_forest", "randomforest", "randomforestclassifier"]:
        params = {
            "n_estimators": 200,
            "max_depth": 10,
            "min_samples_split": 2,
            "random_state": random_state
        }
        params.update(kwargs)
        return RandomForestClassifier(**params)

    elif model_name_lower in ["svm", "svc", "support_vector_machine"]:
        params = {
            "kernel": "rbf",
            "C": 1.0,
            "gamma": "scale",
            "probability": True,
            "random_state": random_state
        }
        params.update(kwargs)
        return SVC(**params)

    elif model_name_lower in ["knn", "k_nearest_neighbors", "kneighborsclassifier"]:
        params = {
            "n_neighbors": 7,
            "weights": "uniform",
            "metric": "minkowski"
        }
        params.update(kwargs)
        return KNeighborsClassifier(**params)

    elif model_name_lower in ["gb", "gradient_boosting", "gradientboosting", "gradientboostingclassifier"]:
        params = {
            "n_estimators": 200,
            "learning_rate": 0.1,
            "max_depth": 2,
            "random_state": random_state
        }
        params.update(kwargs)
        return GradientBoostingClassifier(**params)

    else:
        valid_models = ["logistic_regression", "random_forest", "svm", "knn", "gradient_boosting"]
        raise ValueError(f"Unknown model: '{model_name}'. Valid models are: {valid_models}")


def get_all_baseline_models(random_state: int = 42) -> dict[str, BaseEstimator]:
    """Retrieve all five baseline classifiers initialized with the paper's parameters."""
    return {
        "Logistic Regression": get_model("logistic_regression", random_state=random_state),
        "Random Forest": get_model("random_forest", random_state=random_state),
        "Support Vector Machine": get_model("svm", random_state=random_state),
        "K-Nearest Neighbors": get_model("knn", random_state=random_state),
        "Gradient Boosting": get_model("gradient_boosting", random_state=random_state)
    }
