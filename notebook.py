from pathlib import Path
from tempfile import gettempdir

from sklearn.datasets import fetch_openml, load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score


def load_data(seed=42):
    cache_dir = Path(gettempdir()) / "ml_openml_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    openml_cache = cache_dir / "openml"

    try:
        if not openml_cache.exists():
            raise FileNotFoundError("OpenML cache not available locally.")
        X, y = fetch_openml(
            "mnist_784",
            version=1,
            as_frame=False,
            return_X_y=True,
            data_home=str(cache_dir),
        )
        y = y.astype(int)
    except Exception:
        digits = load_digits()
        X, y = digits.data, digits.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=seed,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test


def train_random_forest(X_train, y_train, seed=42):
    model = RandomForestClassifier(random_state=seed)
    model.fit(X_train, y_train)
    return model


def train_adaboost(X_train, y_train, seed=42):
    model = AdaBoostClassifier(random_state=seed)
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)


def run_pipeline(model_type="rf", seed=42):
    X_train, X_test, y_train, y_test = load_data(seed)

    if model_type == "rf":
        model = train_random_forest(X_train, y_train, seed)
    elif model_type == "ab":
        model = train_adaboost(X_train, y_train, seed)
    else:
        raise ValueError("model_type deve ser 'rf' ou 'ab'")

    return evaluate(model, X_test, y_test)
