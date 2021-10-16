from .find_new_clusters import c_find_new_clusters
from sklearn.metrics.pairwise import euclidean_distances
from .one_vs_all_classifier import OneVsAllClassifier
from copy import deepcopy
import numpy as np


def find_new_clusters(x_l, x_u, rejected_inds, n_neighbors=5):
    u = x_u.shape[0]
    l = x_l.shape[0]
    arange_u = np.arange(u)
    x_ = np.concatenate((x_l, x_u))
    dist_matrix = euclidean_distances(x_u, x_)
    dist_matrix[(arange_u, arange_u + l)] = 1e+15
    # adjacency matrix: 1 if one ex. is knn for the other; 0 otherwise
    sorted_inds_by_dist = np.argsort(dist_matrix, axis=1)
    inds = (np.array([[i]*n_neighbors for i in range(u)]).ravel(),
            sorted_inds_by_dist[:, :n_neighbors].ravel())
    # limit to unlabeled data only
    unlab_inds = inds[0][inds[1] >= l], inds[1][inds[1] >= l] - l
    adj_matrix = np.zeros((u, u))
    adj_matrix[unlab_inds] = 1
    rej_matrix = np.zeros((u, u))
    # set to zero all rows/columns from not rejected inds
    rej_matrix[:, rejected_inds] = 1
    adj_matrix = adj_matrix * rej_matrix
    adj_matrix[np.setdiff1d(arange_u, rejected_inds)] = 0
    pairs = np.array((np.where(adj_matrix))).T
    clusters = []
    for pair in pairs.tolist():
        if len(clusters) == 0:
            clusters.append(pair)
        else:
            i = 0
            for cluster in clusters:
                if pair[0] in cluster:
                    if pair[1] not in cluster:
                        cluster.append(pair[1])
                elif pair[1] in cluster:
                    if pair[0] not in cluster:
                        cluster.append(pair[0])
                # it is a new cluster
                else:
                    i += 1
            if i == len(clusters):
                clusters.append(pair)
    # merge clusters if they have intersections
    to_keep = np.array([True] * len(clusters))
    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            if not to_keep[j] or not to_keep[i]:
                continue
            if np.intersect1d(clusters[i], clusters[j]).size != 0:
                clusters[i].extend(np.setdiff1d(clusters[j], clusters[i]))
                to_keep[j] = False
    clusters = np.array(clusters)[to_keep]
    return clusters


class OpenWorldSelfLearning:
    def __init__(self, base_estimator=None, one_vs_all=True, theta=0.7, gamma_rej=0.55, n_neighbors=5,
                 min_cluster_samples=4, cython=True, decreased_pl_weights=True, max_iter=None, random_state=None):
        self.one_vs_all = one_vs_all
        self.theta = theta
        self.gamma_rej = gamma_rej
        self.n_neighbors = n_neighbors
        self.min_cluster_samples = min_cluster_samples
        self.cython = cython
        self.max_iter = max_iter
        if self.max_iter is not None:
            self.max_iter = int(self.max_iter)
        self.decreased_pl_weights = decreased_pl_weights
        self.random_state = random_state
        # validate base_estimator
        self.base_estimator = base_estimator
        self._initialize_base_estimator_()
        self.log_all_iterations = None
        self.y_u_pred = None

    def fit(self, x_l, y_l, x_u):
        """
        :param x_l: Labeled training observations
        :param y_l: Labels
        :param x_u:  Unlabeled training observations
        """
        l = x_l.shape[0]
        u = x_u.shape[0]
        classes = np.unique(y_l)
        init_classes = deepcopy(classes)
        n_classes = classes.size
        # initialize log to store basic info about each iteration of self-learning
        log = LogStorer(x_l, y_l, x_u)
        cond = True
        it = 0
        idx_u = np.arange(u)
        sample_weight = None
        # initialization: supervised model trained on labeled examples only
        if self.one_vs_all:
            estimator = OneVsAllClassifier(base_estimator=self.base_estimator_)
        else:
            estimator = deepcopy(self.base_estimator_)
        if 'x_u' in self.base_estimator_.fit.__code__.co_varnames:
            estimator.fit(x_l, y_l, x_u)
        else:
            estimator.fit(x_l, y_l)
        log.update(estimator, x_l, y_l, x_u, [])
        # start of self-learning
        while cond:
            it += 1
            idx_pl = log.current_iteration['idx_pl']

            # predict confidence matrix (u x K),
            # u - num. of unlab. examples, K - num. of classes in the training set
            # if one_vs_all is False, each rows sums up to 1
            conf_matrix_u = estimator.predict_proba(x_u)

            # pseudo-label examples from old classes and with high confidence
            for i in range(n_classes):
                rest_classes = np.setdiff1d(np.arange(n_classes), [i])
                selection = np.where(np.logical_and(conf_matrix_u[:, i] >= self.theta,
                                                    np.all(conf_matrix_u[:, rest_classes] < self.theta, axis=1)))[0]
                # if nothing to select, continue
                if len(selection) != 0:
                    x_s = x_u[selection, :]
                    # select the examples and pseudo-label them
                    y_s = np.repeat(i, x_s.shape[0])
                    # move them from the unlabeled set to the labeled one
                    x_l = np.concatenate((x_l, x_s))
                    y_l = np.concatenate((y_l, y_s))
                    idx_pl = np.concatenate((idx_pl, idx_u[selection]))
                    x_u = np.delete(x_u, selection, axis=0)
                    idx_u = np.delete(idx_u, selection)
                    conf_matrix_u = np.delete(conf_matrix_u, selection, axis=0)

            # find new classes from rejection
            # new classes are pseudo-labeled and moved to the training set
            rejected_inds = np.where(np.all(conf_matrix_u < self.gamma_rej, axis=1))[0]
            if rejected_inds.size == 0:
                cond = False
                continue
            if self.cython:
                clusters = c_find_new_clusters(np.concatenate((x_l, x_u)), x_l.shape[0], rejected_inds, self.n_neighbors)
            else:
                clusters = find_new_clusters(x_l, x_u, rejected_inds, n_neighbors=self.n_neighbors)
            clusters_chosen = clusters[[len(clusters[i]) >= self.min_cluster_samples for i in range(len(clusters))]]
            if len(clusters_chosen) != 0:
                to_delete = list()
                for i in range(len(clusters_chosen)):
                    n_classes += 1
                    new_class = str(n_classes - 1)
                    classes = np.concatenate((classes, [new_class]))
                    x_s = x_u[clusters_chosen[i]]
                    y_s = np.repeat(new_class, x_s.shape[0])
                    # move them from the unlabeled set to the labeled one
                    x_l = np.concatenate((x_l, x_s))
                    y_l = np.concatenate((y_l, y_s))
                    idx_pl = np.concatenate((idx_pl, idx_u[clusters_chosen[i]]))
                    to_delete.extend(clusters_chosen[i])
                x_u = np.delete(x_u, to_delete, axis=0)
                idx_u = np.delete(idx_u, to_delete)

            # if True, the weight of pseudo-labeled examples is decreased
            if self.decreased_pl_weights:
                u_pl = x_l.shape[0] - l
                sample_weight = np.concatenate((np.repeat(1 / l, l), np.repeat(1 / u_pl, u_pl)))

            # learn a new classifier
            if self.one_vs_all:
                estimator = OneVsAllClassifier(base_estimator=self.base_estimator_)
            else:
                estimator = deepcopy(self.base_estimator_)
            if 'x_u' in self.base_estimator_.fit.__code__.co_varnames:
                estimator.fit(x_l, y_l, x_u, sample_weight=sample_weight)
            else:
                estimator.fit(x_l, y_l, sample_weight=sample_weight)
            # update log
            log.update(estimator, x_l, y_l, x_u, idx_pl)
            # stop if max_iter is reached
            if it == self.max_iter:
                cond = False
            # stop if all unlabeled examples are pseudo-labeled
            if x_u.shape[0] == 0:
                cond = False
        log_all_iterations = log.previous_iterations
        log_all_iterations.append(log.current_iteration)
        self.log_all_iterations = log_all_iterations
        self.y_u_pred = np.repeat(-1, u)
        self.y_u_pred[np.array(log.current_iteration['idx_pl']).astype(int)] = log.current_iteration['y_l'][l:]
        maps = {i: value for i, value in enumerate(init_classes)}
        self.y_u_pred = np.array([maps[val] if val in np.arange(init_classes.size) else val for val in self.y_u_pred])

    def _initialize_base_estimator_(self):
        if self.base_estimator is not None:
            self.base_estimator_ = self.base_estimator
            self._agree_random_state()
        # by default, base_estimator_ is a random forest
        else:
            from sklearn.ensemble import RandomForestClassifier
            rf_model = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=self.random_state)
            self.base_estimator_ = rf_model

    def _agree_random_state(self):
        # if base_estimator_ has random_state attribute
        if hasattr(self.base_estimator_, 'random_state'):
            # and if random_state is not default,
            # replace the rs of base_estimator_ by random_state
            if self.random_state is not None:
                # we raise the warning, if the rd of base_estimator_ is not None initially
                if self.random_state != self.base_estimator_.random_state is not None:
                    raise Warning("random state of base_estimator_ is set to " + str(self.random_state))
                self.base_estimator_.random_state = self.random_state


class LogStorer:
    def __init__(self, x_l, y_l, x_u):
        self.x_l = x_l
        self.y_l = y_l
        self.x_u = x_u
        self.current_iteration = None
        self.previous_iterations = list()

    def update(self, estimator, x_l, y_l, x_u, idx_pl):
        if self.current_iteration is not None:
            self.previous_iterations.append(self.current_iteration)
        self.current_iteration = {
            "estimator": estimator,
            "x_l": x_l,
            "y_l": y_l,
            "x_u": x_u,
            "idx_pl": idx_pl,
        }
