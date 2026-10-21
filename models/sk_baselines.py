# 传统机器学习 baseline。

from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor


def build_rf(seed=42):
    return RandomForestRegressor(
        n_estimators=300,
        random_state=seed
    )


def build_mlp(seed=42):
    return MLPRegressor(
        hidden_layer_sizes=(64, 32),
        max_iter=3000,
        random_state=seed
    )


def build_xgb(seed=42):
    return XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=seed
    )