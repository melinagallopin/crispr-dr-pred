import numpy as np
from copy import deepcopy
from joblib import Parallel, delayed, effective_n_jobs
from itertools import repeat
from sklearn.utils.fixes import _joblib_parallel_args
# import pyximport; pyximport.install(setup_args={"include_dirs": np.get_include()},
#                                     language_level="3", reload_support=True)
from .probabilistic_classifier_cython import semi_c_find_best_split_of_all, c_get_prediction
from sklearn.preprocessing import LabelEncoder
# import warnings
# warnings.simplefilter('error', RuntimeWarning)


def prob_error(probs, y_pred):
    probs_copy = deepcopy(probs)
    probs_copy[(np.arange(len(y_pred)), y_pred)] = 0
    return probs_copy.sum(axis=1).mean()


def prob_entropy(group):
    # group is a matrix with K columns
    if group.size == 0:
        return 0
    props = group.mean(axis=0)
    props = props[props != 0]
    return - np.dot(props, np.log(props))


def prob_gini(group):
    # group is a matrix with K columns
    if group.size == 0:
        return 0
    props = group.mean(axis=0)
    return np.dot(props, 1-props)


def prob_misclass(group):
    # group is a matrix with K columns
    if group.size == 0:
        return 0
    props = group.mean(axis=0)
    return 1 - props.max()


def information_gain(y, divide_cond, criterion):
    # left and right are matrices with K columns
    y_left, y_right = y[divide_cond], y[np.logical_not(divide_cond)]
    n_left, n_right, n_parent = y_left.shape[0], y_right.shape[0], y.shape[0]
    return criterion(y) - (n_left * criterion(y_left) + n_right * criterion(y_right)) / n_parent


def variance(x, divide_cond):
    x_left, x_right = x[divide_cond], x[np.logical_not(divide_cond)]
    if x_left.shape[0] == 0 or x_right.shape[0] == 0:
        return 0
    n_left, n_right, n_parent = x_left.shape[0], x_right.shape[0], x.shape[0]
    x_bar = np.mean(x)
    return (n_left * ((x_left.mean()-x_bar)**2) + n_right * ((x_right.mean()-x_bar)**2)) / (n_parent * x.var())


def convert_to_probabilistic_label(y):
    # long time ago, my decision tree has been implemented for probabilistic labels,
    # so this function converts an ordinary label variable to the probabilistic one
    if len(y.shape) == 1:
        n_classes = np.unique(y).size
        y_ = np.zeros((y.shape[0], n_classes))
        y_[(np.arange(y.shape[0]), y)] = 1
        return y_
    else:
        return y


# lam = 0 means that classifier is supervised (only labeled data)
# lam = 1 means that classifier is unsupervised (labels are ignored)
# lam can be a list of size max_depth: then, at the i-th depth lam[i] will be used
class SemisupProbabilisticDecisionTreeClassifier:
    def __init__(self, splitter='best', criterion='entropy', max_depth=None, min_samples_split=2, num_rand_splits=0.5,
                 max_features='sqrt', leaf_uncertainty_threshold=0.9, lam=0, cython=False, random_state=None):
        # self.depth = 0
        self.splitter = splitter
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.num_rand_splits = num_rand_splits
        self.leaf_uncertainty_threshold = leaf_uncertainty_threshold
        self.lam = lam
        self.cython = cython
        self.random_state = random_state
        self.tree_ = None
        self.feature_importances_ = None
        self.criterion_ = None
        self.max_features_ = None
        self.max_depth_ = None
        self.num_rand_splits_ = None
        self.n_classes_ = None
        self.lam_ = None
        self.l = None

    def fit(self, x_l, y_l, x_u=None):
        if self.random_state is not None:
            np.random.seed(self.random_state)
        if x_u is not None:
            x = np.concatenate((x_l, x_u))
        else:
            x = x_l
        y_l_ = convert_to_probabilistic_label(y_l)
        self.l = y_l_.shape[0]
        self.feature_importances_ = np.zeros(x.shape[1])
        self.n_classes_ = y_l_.shape[1]
        if self.criterion == 'gini':
            self.criterion_ = prob_gini
        elif self.criterion == 'misclass':
            self.criterion_ = prob_misclass
        # entropy otherwise
        else:
            self.criterion_ = prob_entropy
        if self.max_features == 'sqrt':
            self.max_features_ = int(np.sqrt(x.shape[1]))
        elif type(self.max_features) == float:
            if 0 < self.max_features <= 1:
                self.max_features_ = int(self.max_features * x.shape[1])
            else:
                raise KeyError("max_features must be within (0,1] if float")
        elif self.max_features is None:
            self.max_features_ = x.shape[1]
        else:
            self.max_features_ = self.max_features
        if self.max_depth is None:
            self.max_depth_ = min(10, x.shape[1] / 10)
        else:
            self.max_depth_ = self.max_depth

        if np.array(self.lam).size != 1:
            if np.array(self.lam).size != self.max_depth_:
                raise KeyError("lam must be either float within [0,1] or a list of size equal to max_depth_")
            self.lam_ = self.lam
        else:
            if not (0 <= self.lam <= 1):
                raise KeyError("lam must be either float within [0,1] or a list of size equal to max_depth_")
            self.lam_ = np.repeat(self.lam, self.max_depth_)

        if type(self.num_rand_splits) == float:
            if 0 < self.num_rand_splits <= 1:
                self.num_rand_splits_ = int(self.num_rand_splits * x.shape[0])
            else:
                raise KeyError("num_rand_splits must be either int or float within (0,1]")
        elif type(self.num_rand_splits) == int:
            self.num_rand_splits_ = self.num_rand_splits
        else:
            raise KeyError("num_rand_splits must be either int or float within (0,1]")

        # initialize the root of the tree
        root = {'feat_idx': None, 'cutoff': None, 'gain': None, 'group_idx': np.arange(x.shape[0]), 'props': None,
                'depth': 0, 'left': None, 'right': None}
        self.tree_ = [root]
        i = 0
        # a stack of not yet processed nodes (their indices in self.tree_)
        stack = [0]
        while len(stack) != 0:
            node_idx = stack.pop()
            subset_l = self.tree_[node_idx]['group_idx'][self.tree_[node_idx]['group_idx'] < self.l]
            self._one_node_split(x[self.tree_[node_idx]['group_idx']], x[subset_l], y_l_[subset_l], node_idx)
            # if the node was found to be split, create children
            if self.tree_[node_idx]['feat_idx'] is not None:
                left_cond = np.where(x[:, self.tree_[node_idx]['feat_idx']] < self.tree_[node_idx]['cutoff'])[0]
                left_child = {'feat_idx': None, 'cutoff': None, 'gain': None,
                              'group_idx': np.intersect1d(self.tree_[node_idx]['group_idx'], left_cond), 'props': None,
                              'depth': self.tree_[node_idx]['depth'] + 1, 'left': None, 'right': None}
                self.tree_.append(left_child)
                i += 1
                stack.append(i)
                self.tree_[node_idx]['left'] = i

                right_cond = np.where(x[:, self.tree_[node_idx]['feat_idx']] >= self.tree_[node_idx]['cutoff'])[0]
                right_child = {'feat_idx': None, 'cutoff': None, 'gain': None,
                               'group_idx': np.intersect1d(self.tree_[node_idx]['group_idx'], right_cond),
                               'props': None, 'depth': self.tree_[node_idx]['depth'] + 1, 'left': None, 'right': None}
                self.tree_.append(right_child)
                i += 1
                stack.append(i)
                self.tree_[node_idx]['right'] = i
        self.feature_importances_ /= np.sum(self.feature_importances_)

    def _one_node_split(self, x, x_l, y_l, node_idx):
        n_lab = y_l.shape[0]
        n_all = x.shape[0]
        if n_lab == 0:
            self.tree_[node_idx]['props'] = np.repeat(1 / self.n_classes_, self.n_classes_)
        else:
            self.tree_[node_idx]['props'] = y_l.mean(axis=0)
        if self.tree_[node_idx]['depth'] >= self.max_depth_:
            return
        lam = self.lam_[self.tree_[node_idx]['depth']]
        # if the number of examples in a leaf less than min_samples_split
        if n_all < self.min_samples_split:
            return
        elif lam == 0 and n_lab < self.min_samples_split:
            return
        # if for (at least) one class the mean posterior probability
        # is greater than leaf_uncertainty_threshold
        # elif np.any(self.tree_[node_idx]['props'] >= self.leaf_uncertainty_threshold):
        #     return
        else:
            if self.cython:
                if y_l.shape[0] == 0:
                    x_l_to_send = np.repeat(-1, x.shape[1]).reshape(1, -1).astype(float)
                    y_l_to_send = np.repeat(-1, self.n_classes_).reshape(1, -1).astype(float)
                else:
                    x_l_to_send = x_l
                    y_l_to_send = y_l
                feat_idx, cutoff, gain = semi_c_find_best_split_of_all(x, x_l_to_send, y_l_to_send, lam,
                                                                       self.criterion, self.max_features_,
                                                                       self.splitter, self.num_rand_splits_,
                                                                       self.random_state)
            else:
                feat_idx, cutoff, gain = self.find_best_split_of_all(x, x_l, y_l, lam)
            # if gain is close to 0, we check if it makes sense to split
            # if isclose(gain, 0, rel_tol=1e-4):
            if np.sum(x[:, feat_idx] < cutoff) == n_all or np.sum(x[:, feat_idx] >= cutoff) == n_all:
                return
            self.tree_[node_idx]['feat_idx'] = feat_idx
            self.tree_[node_idx]['cutoff'] = cutoff
            self.tree_[node_idx]['gain'] = gain
            self.feature_importances_[feat_idx] += gain * x.shape[0]
            return

    def find_best_split_of_all(self, x, x_l, y_l, lam):
        all_feat_inds = np.arange(x.shape[1])
        if self.max_features_ is not None:
            feat_inds = np.random.choice(all_feat_inds, self.max_features_, replace=False)
        else:
            feat_inds = all_feat_inds
        best_feat_splits = np.array([self.find_best_split(x[:, feat_idx], x_l[:, feat_idx], y_l, lam)
                                     for feat_idx in feat_inds])
        idx_max = np.argmax(best_feat_splits[:, 0])
        max_gain = best_feat_splits[idx_max, 0]
        cutoff = best_feat_splits[idx_max, 1]
        chosen_feat_idx = feat_inds[idx_max]
        return chosen_feat_idx, cutoff, max_gain

    def find_best_split(self, feature, sup_feature, y, lam):
        # evaluate information gain
        if lam == 0:
            unsup_gains = np.array([0])
            values = _define_values(sup_feature, self.splitter, self.num_rand_splits_)
        else:
            values = _define_values(feature, self.splitter, self.num_rand_splits_)
            unsup_gains = np.array([variance(feature, feature < value) for value in values])
        if lam == 1:
            sup_gains = np.array([0])
        else:
            if y.size == 0:
                sup_gains = np.array([0])
            else:
                sup_gains = np.array([information_gain(y, sup_feature < value, self.criterion_) for value in values])
        gains = (1 - lam) * sup_gains + lam * unsup_gains
        # find cutoff that maximizes information gain
        idx_max = np.argmax(gains)
        max_gain = gains[idx_max]
        cutoff = values[idx_max]
        return [max_gain, cutoff]

    def predict_proba(self, x):
        if self.cython:
            probs = c_get_prediction(x, self.tree_, self.n_classes_)
        else:
            probs = np.array([self._get_prediction(row) for row in x])
        return probs

    def predict(self, x):
        return self.predict_proba(x).argmax(axis=1)

    def _get_prediction(self, row):
        node = self.tree_[0]
        while True:
            if node['feat_idx'] is None:
                return node['props']
            elif row[node['feat_idx']] < node['cutoff']:
                node = self.tree_[node['left']]
            else:
                node = self.tree_[node['right']]

    def __str__(self):
        return f"{self.__class__.__name__}({', '.join(f'{k}={v}' for k, v in self.__dict__.items() if not k.startswith('_') and not k.endswith('_'))})"

    def __repr__(self):
        return f"{self.__class__.__name__}({', '.join(f'{k}={v}' for k, v in self.__dict__.items() if not k.startswith('_') and not k.endswith('_'))})"


def _define_values(feature, splitter, num_rand_splits):
    feature_values = np.unique(feature)
    # a set of possible cutoffs
    if splitter == 'random':
        values = np.random.uniform(low=feature_values.min(), high=feature_values.max(), size=num_rand_splits)
    # best splitter
    else:
        values = feature_values
    return values


# lam = 0 means that classifier is supervised (only labeled data)
# lam = 1 means that classifier is unsupervised (labels are ignored)
# lam can be a list of size max_depth: then, at the i-th depth lam[i] will be used
class SemisupProbabilisticRandomForestClassifier:
    def __init__(self, n_estimators=100, criterion='entropy', splitter='best', max_depth=None, min_samples_split=2,
                 num_rand_splits=0.5, max_features='sqrt', leaf_uncertainty_threshold=0.9, lam=0, cython=True,
                 oob_score=True, num_bootstrap_samples=None, n_jobs=1, random_state=None):
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.num_rand_splits = num_rand_splits
        self.leaf_uncertainty_threshold = leaf_uncertainty_threshold
        self.lam = lam
        self.cython = cython
        self.oob_score = oob_score
        self.num_bootstrap_samples = num_bootstrap_samples
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.feature_importances_ = None
        self.bootstrap_indices_ = None
        self.oob_decision_function_ = None
        self.estimators_ = None
        self.max_features_ = None
        self.max_depth_ = None
        self.num_bootstrap_samples_ = None
        self.lam_ = None
        self.classes_ = None

    def fit(self, x_l, y_l, x_u, stratify_bootstrap=None, sample_weight=None):
        if len(y_l.shape) == 1:
            le = LabelEncoder()
            inds = le.fit_transform(y_l)
            self.classes_ = le.classes_
            n_classes = self.classes_.size
            y_l_ = np.zeros((inds.shape[0], n_classes))
            y_l_[(np.arange(inds.shape[0]), inds)] = 1
        else:
            y_l_ = y_l
        self.feature_importances_ = np.zeros(x_l.shape[1])
        if self.random_state is not None:
            np.random.seed(self.random_state)
        if self.max_features == 'sqrt':
            self.max_features_ = int(np.sqrt(x_l.shape[1]))
        elif type(self.max_features) == float:
            if 0 < self.max_features <= 1:
                self.max_features_ = int(self.max_features * x_l.shape[1])
            else:
                raise KeyError("max_features must be within (0,1] if float")
        else:
            self.max_features_ = self.max_features
        if self.max_depth is None:
            self.max_depth_ = min(10, x_l.shape[1] / 10)
        else:
            self.max_depth_ = self.max_depth

        if np.array(self.lam).size != 1:
            if np.array(self.lam).size != self.max_depth_:
                raise KeyError("lam must be either float within [0,1] or a list of size equal to max_depth_")
            self.lam_ = self.lam
        else:
            if not (0 <= self.lam <= 1):
                raise KeyError("lam must be either float within [0,1] or a list of size equal to max_depth_")
            self.lam_ = np.repeat(self.lam, self.max_depth_)

        if self.num_bootstrap_samples is None:
            if stratify_bootstrap is None:
                self.num_bootstrap_samples_ = x_l.shape[0]
            else:
                unique_markers, marker_counts = np.unique(stratify_bootstrap, return_counts=True)
                self.num_bootstrap_samples_ = unique_markers.size * marker_counts.min()
        else:
            self.num_bootstrap_samples_ = self.num_bootstrap_samples

        random_states = np.random.randint(np.iinfo(np.int32).max, size=self.n_estimators)
        trees = [self._init_tree(random_state) for random_state in random_states]
        self.bootstrap_indices_ = [_generate_bootstrap_indices(x_l.shape[0], self.num_bootstrap_samples_,
                                                               stratify_bootstrap) for i in range(self.n_estimators)]
        out_of_bags = [np.setdiff1d(np.arange(y_l_.shape[0]), self.bootstrap_indices_[i]) for i in
                       range(self.n_estimators)]

        if self.n_jobs == 1:
            trees = list(map(_learn_tree, trees, repeat(x_l), repeat(y_l_), repeat(x_u), self.bootstrap_indices_,
                             repeat(None)))
        else:
            if self.n_jobs == -1:
                n_jobs = min(self.n_estimators, effective_n_jobs(-1))
            else:
                n_jobs = min(self.n_estimators, self.n_jobs)
            trees = Parallel(n_jobs=n_jobs, **_joblib_parallel_args(prefer='processes'))(
                delayed(_learn_tree)(trees[i], x_l, y_l_, x_u, self.bootstrap_indices_[i], None)
                for i in range(self.n_estimators))

        oob_predictions = [_one_tree_predict(np.zeros(y_l_.shape), trees[i], x_l, out_of_bags[i]) for i in
                           range(self.n_estimators)]
        self.oob_decision_function_ = np.add.reduce(oob_predictions)
        self.oob_decision_function_ /= np.broadcast_to(self.oob_decision_function_.sum(axis=1),
                                                       (y_l_.shape[1], y_l_.shape[0])).T
        self.feature_importances_ = np.add.reduce([tree.feature_importances_ for tree in trees])
        self.feature_importances_ = self.feature_importances_ / self.n_estimators
        # self.oob_score_ = accuracy_score(y, self.oob_decision_function_.argmax(axis=1))
        self.estimators_ = trees

    def _init_tree(self, random_state):
        model = SemisupProbabilisticDecisionTreeClassifier(criterion=self.criterion, splitter=self.splitter,
                                                           max_depth=self.max_depth_,
                                                           min_samples_split=self.min_samples_split,
                                                           num_rand_splits=self.num_rand_splits,
                                                           max_features=self.max_features_, lam=self.lam_,
                                                           leaf_uncertainty_threshold=self.leaf_uncertainty_threshold,
                                                           cython=self.cython, random_state=random_state)
        return model

    def predict(self, x):
        res = self.predict_proba(x).argmax(axis=1)
        if self.classes_ is None:
            return res
        else:
            return self.classes_[res]

    def predict_proba(self, x):
        return np.add.reduce([tree.predict_proba(x) for tree in self.estimators_]) / self.n_estimators

    def __str__(self):
        return f"{self.__class__.__name__}({', '.join(f'{k}={v}' for k, v in self.__dict__.items() if not k.startswith('_') and not k.endswith('_'))})"

    def __repr__(self):
        return f"{self.__class__.__name__}({', '.join(f'{k}={v}' for k, v in self.__dict__.items() if not k.startswith('_') and not k.endswith('_'))})"


def _generate_bootstrap_indices(n_samples, n_bootstrap_samples, stratify_bootstrap):
    if stratify_bootstrap is None:
        return np.random.randint(0, n_samples, n_bootstrap_samples)
    else:
        unique_markers = np.unique(stratify_bootstrap)
        one_marker_bootstrap_size = int(n_bootstrap_samples / unique_markers.size)
        return np.array([np.random.choice(np.where(stratify_bootstrap == marker)[0], one_marker_bootstrap_size)
                         for marker in unique_markers]).ravel()


def _learn_tree(tree, x, y, x_u, bag_indices, out_of_bag_indices):
    if out_of_bag_indices is None:
        tree.fit(x[bag_indices], y[bag_indices], x_u)
    else:
        tree.fit(x[bag_indices], y[bag_indices], x[out_of_bag_indices], y[out_of_bag_indices])
    return tree


def _one_tree_predict(pred_mat, tree, x, out_of_bag_indices):
    pred_mat[out_of_bag_indices] = tree.predict_proba(x[out_of_bag_indices])
    return pred_mat




