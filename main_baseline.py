import math
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from config import CSV_PATH, TEST_SIZE, SEED
from models.sk_baselines import (
    build_rf,
    build_xgb,
    build_mlp
)

from evaluation.compare_models import save_results


def load_tabular():

    df = pd.read_csv(CSV_PATH)

    feat_cols = [
        "density",
        "band_gap",
        "lattice_a",
        "lattice_b",
        "lattice_c",
        "avg_eneg",
        "ionic_radius",
        "activation_energy"
    ]

    X = df[feat_cols].values

    y = df["conductivity"].apply(
        lambda v: math.log10(v + 1e-8)
    ).values

    return X, y


def evaluate(y_true, pred):

    return {
        "MAE": mean_absolute_error(y_true, pred),
        "RMSE": math.sqrt(mean_squared_error(y_true, pred)),
        "R2": r2_score(y_true, pred)
    }


def main():

    X, y = load_tabular()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=SEED
    )

    models = {
        "RF": build_rf(SEED),
        "XGB": build_xgb(SEED),
        "MLP": build_mlp(SEED)
    }

    results = []

    for name, model in models.items():

        model.fit(X_train, y_train)

        pred = model.predict(X_test)

        m = evaluate(y_test, pred)

        m["Model"] = name

        results.append(m)

    save_results(results)


if __name__ == "__main__":
    main()