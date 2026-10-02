"""Random Forest baseline (scikit-learn)."""
from sklearn.ensemble import RandomForestRegressor


def make_rf(n_estimators=300, max_depth=10, min_samples_leaf=5, max_features=0.5, seed=42):
    return RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                 min_samples_leaf=min_samples_leaf, max_features=max_features,
                                 random_state=seed, n_jobs=-1)
