"""Fully-connected neural network (scikit-learn MLPRegressor), architecture from the paper:
input -> 500 -> 100 -> 20 -> 1 with ReLU. Trained with mini-batch Adam, L2 weight decay (alpha),
and chronological early stopping on the validation season."""
import copy
import numpy as np
from sklearn.neural_network import MLPRegressor


class NNRegressor:
    def __init__(self, hidden=(500, 100, 20), lr=1e-3, alpha=1e-2, batch_size=64, max_epochs=60,
                 patience=8, seed=42):
        self.hidden, self.lr, self.alpha, self.batch_size = tuple(hidden), lr, alpha, batch_size
        self.max_epochs, self.patience, self.seed = max_epochs, patience, seed
        self.history = {"train_mse": [], "val_mse": []}

    def fit(self, Xtr, ytr, Xva, yva):
        self.mu, self.sd = ytr.mean(), ytr.std()
        z = (ytr - self.mu) / self.sd
        net = MLPRegressor(hidden_layer_sizes=self.hidden, activation="relu", solver="adam", alpha=self.alpha,
                           learning_rate_init=self.lr, batch_size=self.batch_size, random_state=self.seed)
        rng = np.random.default_rng(self.seed)
        best, best_net, bad = np.inf, None, 0
        for ep in range(self.max_epochs):
            perm = rng.permutation(len(Xtr))
            net.partial_fit(Xtr[perm], z[perm])
            tr = np.mean((net.predict(Xtr) * self.sd + self.mu - ytr) ** 2)
            va = np.mean((net.predict(Xva) * self.sd + self.mu - yva) ** 2)
            self.history["train_mse"].append(tr); self.history["val_mse"].append(va)
            if va < best - 1e-6:
                best, best_net, bad = va, copy.deepcopy(net), 0
            else:
                bad += 1
                if bad >= self.patience:
                    break
        self.net, self.best_epoch, self.best_val = best_net, int(np.argmin(self.history["val_mse"])) + 1, best
        return self

    def predict(self, X):
        return self.net.predict(X) * self.sd + self.mu
