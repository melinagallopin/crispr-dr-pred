import numpy as np
from copy import deepcopy


class OneVsAllClassifier:
    def __init__(self, base_estimator):
        self.base_estimator = base_estimator
        self._is_semi_supervised = True if 'x_u' in base_estimator.fit.__code__.co_varnames else False
        self.estimators = None
        self.n_classes = None

    def fit(self, x_l, y_l, x_u=None, sample_weight=None):
        if (x_u is not None) ^ self._is_semi_supervised:
            raise KeyError('x_u should be not None for semi-supervised base classifier and None otherwise')
        classes = np.unique(y_l)
        self.n_classes = classes.size
        self.estimators = []
        for i in classes:
            y_one_vs_rest = np.array([[0.0, 1.0] if label == i else [1.0, 0.0] for label in y_l])
            estimator = deepcopy(self.base_estimator)
            if x_u is not None:
                estimator.fit(x_l, y_one_vs_rest, x_u, sample_weight=sample_weight)
            else:
                estimator.fit(x_l, y_one_vs_rest, sample_weight=sample_weight)
            self.estimators.append(estimator)

    def predict_proba(self, x):
        return np.array([self.estimators[i].predict_proba(x)[:, 1] for i in range(self.n_classes)]).T
